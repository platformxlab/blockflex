#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "vssd.h"

static const int BLK_SZ = PAGE_SIZE*4*256;

//Create a regular vssd
//Probably want to keep these constructors separate
vssd_t * alloc_regular_vssd(int chl) {
    
    //Vars we use later in this function
    int i, j, alloc;

    //Allocate the vssd
    vssd_t * ret_vssd = malloc(sizeof(vssd_t));

    //Set the vssd_type
    ret_vssd->vssd_type = regular;

    //Regular vssd wants to create a mapping table
    ret_vssd->mapping_table_v = calloc(CHL_BLK_NUM * chl,sizeof(uint32_t));
    //Also set the current max address to avoid out of bounds
    ret_vssd->max_addr = CHL_BLK_NUM * chl;

    //For debug, which channels did we successfully allocate 
    //Also useful in case we fail to allocate channels
    ret_vssd->allocated_chl = malloc(chl * sizeof(int));
    for(j =0; j < chl; j++) ret_vssd->allocated_chl[j] = -1;

    //Alloc the lists for block management, alloc for allocated blocks, free for those not allocated
    ret_vssd->alloc_block_list = calloc(16,sizeof(node_t *));
    ret_vssd->free_block_list = calloc(16,sizeof(node_t *));

    //Not sure this is needed
    for(i = 0; i < 16; i++) {
        pthread_mutex_init(&(ret_vssd->free_locks[i]),NULL);
        pthread_mutex_init(&(ret_vssd->alloc_locks[i]),NULL);
    }

    //Alloc lba queue
    ret_vssd->lba_queue = malloc(sizeof(Queue *));
    ret_vssd->lba_queue = createQueue(DEVICE_BLK_NUM);

    //Set the space and bandwidth of this vssd
    set_bw(ret_vssd, chl);
    set_sz(ret_vssd, (uint64_t)chl * CHL_BLK_NUM);

    //Only need one ghost slot
    ret_vssd->ghost = calloc(1,sizeof(vssd_t*));

    //Set up the lba queue with corresponding addresses
    for(j = 0; j < CHL_BLK_NUM * chl; j++) enqueue(ret_vssd->lba_queue, j);

    //Go through and allocate channels
    alloc = 0;
    pthread_t threads[16];
    bundle_t * th_info = malloc(16 * sizeof(bundle_t));
    for(i = 0; i < 16; i++) {
        threads[i] = 0;
    }
    int rc;
    for(i = 0; i < 16 && alloc < chl; i++) {
        //Continue if the channel is already allocated
        if ((alloc_chl & (1 << i)) > 0) continue;
        //Set the allocated bit to track allocated channels
        alloc_chl |= (1 << i);
        //Record this as an allocated channel for this vssd 
        ret_vssd->allocated_chl[alloc++] = i;
        th_info[i].a = ret_vssd;
        th_info[i].index = i;
        if ((rc = pthread_create(&threads[i], NULL, prep_chl_v, &th_info[i]))) {
            fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
            return EXIT_FAILURE;
        } 
        //else fprintf(stdout, "Spawned Thread: %d\n", i);
    }

    for(i = 0; i < 16; i++) {
        if (threads[i] == 0) continue;
        pthread_join(threads[i], NULL);
    }
    free(th_info);
    //In the unlikely event of failure, we dealloc and return failure
    if (alloc < chl) {
        free_regular_vssd(ret_vssd);
        //Return failure
        return NULL;
    }
    //Return vssd on success
    return ret_vssd;
}

//Takes the pointer to a vssd and frees it
void free_regular_vssd(vssd_t * v) {

    //TODO check if we have a ghost vssd and reclaim it first

    int i;
    //Iterate over all channels and free those allocated to us
    for(i = 0; i < v->bandwidth; i++) {

        int f_chl = v->allocated_chl[i];
        if (f_chl == -1) break;

        //Remove from allocated channels
        alloc_chl ^= (1 << f_chl);

        //Free the lists
        free_blk_list_v(v->alloc_block_list[f_chl]);
        free_blk_list_v(v->free_block_list[f_chl]);
    }

    //TODO do we need to do something about home/ghosts here?
    //Free the remaining malloc'ed items in the vssd
    free(v->mapping_table_v);
    free(v->allocated_chl);
    free(v->alloc_block_list);
    free(v->free_block_list);
    free(v->ghost);
    while(dequeue(v->lba_queue) >= 0);
    free(v->lba_queue);
    free(v);
}

//Grabs a pba, lba, maps them and returns the lba
//This function is generic across vssd types
uint32_t alloc_block_v(vssd_t * v, int chl_id) {

    uint32_t ppa_addr;
    int lba_addr;

    //lock the list
    pthread_mutex_lock(&v->free_locks[chl_id]);
    //Grab the block
    node_t * free_blk = pop_list_v(&v->free_block_list[chl_id]);
    pthread_mutex_unlock(&v->free_locks[chl_id]);
    if (free_blk) ppa_addr = free_blk->ppa;
    else {
        printf("Cannot allocate free block\n");
        exit(0);
    }

    //Grab a logical block address
    lba_addr = dequeue(v->lba_queue);
    if (lba_addr == -1) {
        printf("ERROR: cannot get LBA\n");
        exit(0);
    }

    //Map them together
    v->mapping_table_v[lba_addr] = ppa_addr;

    //add to alloc block list, don't need to make a new node since we keep the old one
    add_list_v(&v->alloc_block_list[chl_id], free_blk);

    //Return the lba
    return lba_addr;
}

vssd_t * alloc_pghost_vssd(vssd_t * ghost) {
    //TODO assuming well formed ghost vssd?
    int i;

    vssd_t * ret_vssd = malloc(sizeof(vssd_t));
    ret_vssd->vssd_type = pghost;

    uint32_t chl = get_bw(ghost);
    uint64_t sz = get_sz(ghost);

    ret_vssd->allocated_chl = malloc(chl * sizeof(int));
    for(i = 0; i < chl; i++) ret_vssd->allocated_chl[i] = ghost->allocated_chl[i];

    //Alloc the lists for block management, alloc for allocated blocks, free for those not allocated
    ret_vssd->alloc_block_list = calloc(16,sizeof(node_t *));
    ret_vssd->free_block_list = calloc(16,sizeof(node_t *));

    //Alloc lba queue, we fill in a sec
    ret_vssd->lba_queue = malloc(sizeof(Queue *));
    ret_vssd->lba_queue = createQueue(DEVICE_BLK_NUM);

    //Setup mapping table and allocate the logical block addresses accordingly
    ret_vssd->mapping_table_v = calloc(sz,sizeof(uint32_t));
    ret_vssd->max_addr = (uint32_t)sz;
    //Fill the lba queue 
    for(i = 0; i < sz; i++) enqueue(ret_vssd->lba_queue, i);

    set_bw(ret_vssd, chl);
    set_sz(ret_vssd, sz);
    set_dur(ret_vssd, get_dur(ghost));

    //TODO This will probably be resized
    ret_vssd->ghost = calloc(1,sizeof(vssd_t*));
    ret_vssd->ghosts = 1;
    set_ghost(ret_vssd, ghost);

    //Transfer blocks
    for(i = 0; i < ghost->bandwidth; i++) {

        int f_chl = ghost->allocated_chl[i];

        while(1) {
            //Grab block from ghost vssd
            node_t * f_block = pop_list_v(&ghost->free_block_list[f_chl]);
            if (!f_block) break;

            //Give to pghost vssd
            add_list_v(&ret_vssd->free_block_list[f_chl], f_block);
        }
    }
    return ret_vssd;
}

//TODO This is not shrinking the mapping table at all, I dont see that this
//will have neough additionas and removals that warrants the fact that we should implement shrinkage
void remove_ghost_vssd(vssd_t * pghost, vssd_t * ghost) {
    uint32_t i,j,rc;
    //transfer blocks
    //Handle block transfers in parallel
    bundle_t * th_info = malloc(get_bw(ghost) * sizeof(bundle_t));
    pthread_t threads[16];
    for(i = 0; i < get_bw(ghost); i++) {
        th_info[i].a = pghost;
        th_info[i].b = ghost;
        th_info[i].index = ghost->allocated_chl[i];
        if ((rc = pthread_create(&threads[i], NULL, ret_chl_v, &th_info[i]))) {
            fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
            return EXIT_FAILURE;
        } 
        //else fprintf(stdout, "Spawned Thread: %d\n", i);
    }
    for(i = 0; i < get_bw(ghost); i++) {
        pthread_join(threads[i], NULL);
    }
    free(th_info);
    //Shrink the allocated channel list
    int * temp = malloc(get_bw(pghost) - get_bw(ghost));
    j = 0;
    for(i = 0; i < get_bw(pghost); i++) {
        if (pghost->allocated_chl[i] == -1)  j++;
        else temp[i-j] = pghost->allocated_chl[i];
    }
    free(pghost->allocated_chl);
    pghost->allocated_chl = temp;

    //Shrink the ghost list
    j = 0;
    for(i = 0; i < pghost->ghosts; i++) {
        if (pghost->ghost[i] == ghost)  j++;
        else pghost->ghost[i-j] = pghost->ghost[i];
    }
    if(--pghost->ghosts > 1) {
        pghost->ghost = realloc(pghost->ghost, --pghost->ghosts);
    } else {
        set_ghost(pghost, NULL);
    }

    //We are not going to shrink the size of the mapping table right now.
    set_ghost(ghost, NULL);

    //update size metadata, etc
    set_bw(pghost, get_bw(pghost) - get_bw(ghost));
    set_sz(pghost, get_sz(pghost) - get_sz(ghost));
    //Option 2: Sweep the remaining vssds to recompute the minimum for the duration
    int min = 1 << 30;
    for(i = 0; i < pghost->ghosts && pghost->ghost[i] != NULL; i++) {
        min = min < get_dur(pghost->ghost[i]) ? min: get_dur(pghost->ghost[i]);
    }
    set_dur(pghost, min);
    //Not returning anything
}
void * harvest_chl_v(void * args) {
    bundle_t * info = (bundle_t *) args;
    vssd_t * reg = info->a;
    vssd_t * gh = info->b;
    int blks = info->blks;
    int chl = info->index;
    pthread_mutex_lock(&reg->free_locks[chl]);
    while(blks > 0) {
        //Take one block from the free list
        node_t * f_block = pop_list_v(&reg->free_block_list[chl]);
        if (!f_block) {
            break;
        }
        //Add it to the ghost free list
        add_list_v(&gh->free_block_list[chl], f_block);
        blks--;
    }
    pthread_mutex_unlock(&reg->free_locks[chl]);
}


void * prep_chl_v(void * args) {
    bundle_t * info = (bundle_t *)args;
    vssd_t * ret_vssd = info->a;
    int i,j,k,l;
    i = info->index;
    //Setup up some tracking information
    for(j=0; j < CFG_NAND_LUN_NUM; j++) {
        for(k=0; k < CFG_NAND_BLOCK_NUM; k++) {
            //if(k >= CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK) {
            //    superblk_bad[chl_id][j][k-(CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK)] = 0;
            //    superblk_ers_cnt[chl_id][j][k-(CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK)] = 0;
            //}
            for(l=0; l < CFG_NAND_PLANE_NUM; l++) {
                blk_status[i][j][k][l] = 0;
                blk_ers_cnt[i][j][k][l] = 0;
                blk_bad[i][j][k][l] = 0;
            }
        }
    }

    //allocate the channel, reads badblock file, preps blocks, then adds to the free list 
#ifdef DEBUG
    erase_all_blk_chl_v_debug(i,ret_vssd,1);
#else
    erase_all_blk_chl_v_debug(i,ret_vssd,0);
#endif
}

void erase_blks_v(vssd_t * v) {
    bundle_t * th_info = malloc(get_bw(v) * sizeof(bundle_t));
    pthread_t threads[16];
    int i,rc;
    for(i = 0; i < get_bw(v); i++) {
        th_info[i].a = v;
        th_info[i].index = v->allocated_chl[i];
        if ((rc = pthread_create(&threads[i], NULL, ret_blks_v, &th_info[i]))) {
            fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
            return EXIT_FAILURE;
        } 
        //else fprintf(stdout, "Spawned Thread: %d\n", i);
    }
    for(i = 0; i < get_bw(v); i++) {
        pthread_join(threads[i], NULL);
    }
    free(th_info);
}

void * ret_blks_v(void * args) {
    bundle_t * info = (bundle_t *) args;
    int cur_chl = info->index;
    vssd_t * v = info->a;
    //Erase and return allocated blocks
    while(1) {
        //Grab block from ghost vssd
        node_t * f_block = pop_list_v(&v->alloc_block_list[cur_chl]);
        if (!f_block) break;

        //Erase the block
        erase_blk_v(f_block->ppa);

        //Return to home vssd
        add_list_v(&v->free_block_list[cur_chl], f_block);
    }
}

void * ret_chl_v(void * args) {
    //int cur_chl = ghost->allocated_chl[i];
    bundle_t * info = (bundle_t *)args;
    int cur_chl = info->index;
    vssd_t * pghost = info->a;
    vssd_t * ghost = info->b;
    int j;

    //Return the free blocks directly
    while(1) {
        node_t * f_block = pop_list_v(&pghost->free_block_list[cur_chl]);
        if (!f_block) {
            break;
        }
        //Add it to the parent free list
        add_list_v(&ghost->free_block_list[cur_chl], f_block);
    }

    //Erase and return allocated blocks
    while(1) {
        //Grab block from ghost vssd
        node_t * f_block = pop_list_v(&pghost->alloc_block_list[cur_chl]);
        if (!f_block) break;

        //Erase the block
        erase_blk_v(f_block->ppa);

        //Return to home vssd
        add_list_v(&ghost->free_block_list[cur_chl], f_block);
    }

    //Mark this channel as no longer part of the parent ghost vssd
    for (j = 0; j < get_bw(pghost); j++) {
        if (cur_chl == pghost->allocated_chl[j]) {
            pghost->allocated_chl[j] = -1;
            break;
        }
    }
}

void add_ghost_vssd(vssd_t * pghost, vssd_t * ghost) {
    uint32_t i;
    //transfer blocks
    for(i = 0; i < get_bw(ghost); i++) {
        int cur_chl = ghost->allocated_chl[i];
        while(1) {
            node_t * f_block = pop_list_v(&ghost->free_block_list[cur_chl]);
            if (!f_block) {
                break;
            }
            //Add it to the parent free list
            add_list_v(&pghost->free_block_list[cur_chl], f_block);
        }
    }
    //TODO update metadata
    pghost->allocated_chl = realloc(pghost->allocated_chl, get_bw(pghost) + get_bw(ghost));
    
    //Append the new ghost to the list
    add_ghost(pghost, ghost);
    //Set the ghost to point to the pghost
    set_ghost(ghost, pghost);
    
    //Update mapping table
    pghost->mapping_table_v = realloc(pghost->mapping_table_v, pghost->max_addr + get_sz(ghost));

    //Expand the lba list
    for(i = pghost->max_addr; i < pghost->max_addr + get_sz(ghost); i++) {
        enqueue(pghost->lba_queue, i);
    }

    //Update the maximum address
    pghost->max_addr += get_sz(ghost);
       
    //Update sz, bw, dur
    set_sz(pghost, get_sz(pghost) + get_sz(ghost));
    set_bw(pghost, get_bw(pghost) + get_bw(ghost));
    set_dur(pghost, get_dur(pghost) < get_dur(ghost) ? get_dur(pghost) : get_dur(ghost));
}


/* reg_vssd: Regular vssd we are removing from
 * chl: number of channels we are harvesting from
 * blks: number of blocks
 * dur: duration to tag the vssd with
 */
vssd_t * alloc_ghost_vssd(vssd_t * reg_vssd, int chl, int blks, int dur) {
    //TODO allocate vars needed
    int i;
    //Sanity checks that we can harvest from this vssd before we start
    //if (reg_vssd->vssd_type != regular) return NULL;
    if (reg_vssd->bandwidth < chl) return NULL;
    if (reg_vssd->lba_queue->size < blks) return NULL;

    //Harvest the blocks from the allocated channels
    vssd_t * ret_vssd = malloc(sizeof(vssd_t));
    ret_vssd->vssd_type = ghost;

    //Allocate the allocated chl list
    ret_vssd->allocated_chl = malloc(chl * sizeof(int));
    for(i = 0; i < chl; i++) ret_vssd->allocated_chl[i] = reg_vssd->allocated_chl[i];

    //Alloc the lists for block management, alloc for allocated blocks, free for those not allocated
    ret_vssd->alloc_block_list = calloc(16,sizeof(node_t *));
    ret_vssd->free_block_list = calloc(16,sizeof(node_t *));

    //Alloc lba queue but we do not fill it yet
    ret_vssd->lba_queue = malloc(sizeof(Queue *));
    ret_vssd->lba_queue = createQueue(DEVICE_BLK_NUM);

    //Set the mapping table to null for now
    ret_vssd->mapping_table_v = NULL;

    //Set the metadata
    set_sz(ret_vssd, (uint64_t)blks);
    set_bw(ret_vssd, chl);
    set_dur(ret_vssd, dur);
    
    //Needed if we attach this to a pghost
    ret_vssd->ghost = calloc(1,sizeof(vssd_t*));

    //Set the home/ghost pointers
    set_ghost(reg_vssd, ret_vssd);
    set_home(ret_vssd, reg_vssd);


    //TODO this will do nothing for now
    int fail = 0;
    int blk_per = blks/chl;
    int rc;
    pthread_t threads[16];
    bundle_t * th_info = malloc(16 * sizeof(bundle_t));
    for(i = 0; i < chl; i++) {
        th_info[i].a = reg_vssd;
        th_info[i].b = ret_vssd;
        th_info[i].index = ret_vssd->allocated_chl[i];
        th_info[i].blks = blk_per;
        if ((rc = pthread_create(&threads[i], NULL, harvest_chl_v, &th_info[i]))) {
            fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
            return EXIT_FAILURE;
        } 
    }
    for(i = 0; i < chl; i++) {
        pthread_join(threads[i], NULL);
    }
    free(th_info);
    //Check for success
    //if (fail) {
    //    //create dealloc ghost function to be called here
    //    free_ghost_vssd(ret_vssd);
    //    return NULL;
    //}
    return ret_vssd;
}

void free_pghost_vssd(vssd_t * v) {
    //TODO iterate over all of the component children, remove them and 
    //then free them before freeing yourself
    while(get_ghost(v) != NULL) {
        vssd_t * ghost = get_ghost(v);
        remove_ghost_vssd(v, ghost);
        free_ghost_vssd(ghost);
    }
    free(v->ghost);
    free(v->mapping_table_v);
    free(v->allocated_chl);
    while(dequeue(v->lba_queue) >= 0);
    free(v->lba_queue);
    //Block lists should be empty so neeed to transfer or remove blocks there
    free(v->alloc_block_list);
    free(v->free_block_list);
    free(v);
}

void free_ghost_vssd(vssd_t * v) {
    //get the home vssd from the home pointer
    //Make this optimized with multithreading?
    vssd_t * home = get_home(v);

    int i;
    //Iterate over the free lists to return blocks
    for(i = 0; i < v->bandwidth; i++) {
        int f_chl = v->allocated_chl[i];
        while(1) {

            //Grab block from ghost vssd
            node_t * f_block = pop_list_v(&v->free_block_list[f_chl]);
            if (!f_block) break;

            //Return to home vssd
            add_list_v(&home->free_block_list[f_chl], f_block);
        }
        while(1) {
            //Grab block from ghost vssd
            node_t * f_block = pop_list_v(&v->alloc_block_list[f_chl]);
            if (!f_block) break;

            //Erase the block
            erase_blk_v(f_block->ppa);

            //Return to home vssd
            add_list_v(&home->free_block_list[f_chl], f_block);
        }
    }
    //Free stuff
    if (v->mapping_table_v) free(v->mapping_table_v);
    free(v->alloc_block_list);
    free(v->free_block_list);
    while(dequeue(v->lba_queue) >= 0);
    free(v->lba_queue);
    free(v);
}

//Set the home pointer of the vssd
void set_home(vssd_t * v, vssd_t * home) {
    v->home = home;
}
//Get the home pointer of the vssd
vssd_t * get_home(vssd_t * v) {
    return v->home;
}
//Append a new child ghost
void add_ghost(vssd_t * v, vssd_t * ghost) {
    int cur_ghosts = ++v->ghosts;
    v->ghost = realloc(v->ghost, cur_ghosts);
    v->ghost[cur_ghosts-1] = ghost;
}
//Set the ghost pointer of the vssd
void set_ghost(vssd_t * v, vssd_t * ghost) {
    v->ghost[0] = ghost;
}
//Get the ghost pointer of the vssd
vssd_t * get_ghost(vssd_t * v) {
    return v->ghost[0];
}
//Get the ghost pointer by index of the vssd
vssd_t * get_ghost_ind(vssd_t * v, int ind) {
    return v->ghost[ind];
}
//Set the duration of the vssd
void set_dur(vssd_t * v, uint32_t dur) {
    v->duration = dur;
}
//Get the duration of the vssd
uint32_t get_dur(vssd_t * v) {
    return v->duration;
}
//Set the bandwidth of the vssd
void set_bw(vssd_t * v, uint32_t bw) {
    v->bandwidth = bw;
}
//Get the bandwidth of the vssd
uint32_t get_bw (vssd_t * v) {
    return v->bandwidth;
}
//Set the sz of the vssd
void set_sz(vssd_t * v, uint64_t sz) {
    v->space = sz;
}
//Get the sz of the vssd
uint64_t get_sz(vssd_t * v) {
    return v->space;
}


//==================================================
//         HELPER FUNCTIONS
//==================================================
//Copied helper function
//Free the entire list
int free_blk_list_v(node_t* head) {
    node_t* current = NULL;
    node_t * temp = NULL;
    current = head;
    while(current != NULL) {
        temp = current;
        current = current->next;
        temp->next = NULL;
        free(temp);
    }
    return 0;
}
//Get the length of the list
int list_len_v(node_t * head) {
    node_t * current = head;
    int cnt = 0;
    while(current != NULL) {
        cnt++;
        current = current->next;
    }
    return cnt;
}
//Return the head of the list and remove it 
node_t * pop_list_v(node_t ** head) {
    node_t * current = NULL;
    if (*head == NULL) return NULL;
    else {
        current = *head;
        *head = current->next;
        current->next = NULL;
        return current;
    }
}
//Head is the head of the list
//node is the node_t being added
int add_list_v(node_t ** head, node_t* node) {
    node->next = NULL;
    node_t* current = NULL;
    if(*head == NULL) {
        *head = node;	
    }
    else{
        if((*head)->data <= node->data) {
            node->next = *head;
            *head = node;
        }
        else{
            current = *head;
            while(current->next != NULL && current->next->data > node->data){
                current = current->next;
            }
            node->next = current->next;
            current->next = node;
        }
    }
    return 0;
}
