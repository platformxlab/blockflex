#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <sys/time.h>

#include "channel.h"
#include "bflex.h"
#include "queue.h"
#include "blklist.h"
#include "vssd.h"

//inline double TimeGap( struct timeval* t_start, struct timeval* t_end ) {
//  return (((double)(t_end->tv_sec - t_start->tv_sec) * 1000.0) +
//          ((double)(t_end->tv_usec - t_start->tv_usec) / 1000.0));
//}

u16 *badbin = NULL;


int alloc_channel(int chl_id) {
    int i = 0, j = 0, k = 0;
    /* allocated channel #id */
    if(is_allocated(chl_id) == FUN_SUCCESS) {
        return -1;
    }	

    set_chl_allocated(chl_id);

    for(i=0; i < CFG_NAND_LUN_NUM; i++)
        for(j=0; j < CFG_NAND_BLOCK_NUM; j++) {
            if(j >= CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK) {
                superblk_bad[chl_id][i][j-(CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK)] = 0;
                superblk_ers_cnt[chl_id][i][j-(CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK)] = 0;
            }

            for(k=0; k < CFG_NAND_PLANE_NUM; k++) {
                blk_status[chl_id][i][j][k] = 0;
                blk_ers_cnt[chl_id][i][j][k] = 0;
                blk_bad[chl_id][i][j][k] = 0;
            }
        }

    erase_all_blk_chl(chl_id);


    init_mapping_table(chl_id);

    return FUN_SUCCESS;
}

int alloc_channel_v(int chl_id) {
    int i = 0, j = 0, k = 0;

    for(i=0; i < CFG_NAND_LUN_NUM; i++)
        for(j=0; j < CFG_NAND_BLOCK_NUM; j++) {
            if(j >= CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK) {
                superblk_bad[chl_id][i][j-(CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK)] = 0;
                superblk_ers_cnt[chl_id][i][j-(CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK)] = 0;
            }

            for(k=0; k < CFG_NAND_PLANE_NUM; k++) {
                blk_status[chl_id][i][j][k] = 0;
                blk_ers_cnt[chl_id][i][j][k] = 0;
                blk_bad[chl_id][i][j][k] = 0;
            }
        }
    erase_all_blk_chl(chl_id);
    return 0;
}


int free_channel(int chl_id) {
    /* free channel #id */
    set_chl_free(chl_id);
    if (chl_allocated == 0) {
        free(badbin);
    }
    while(dequeue(lba_queue[chl_id]) >= 0);

    free_alloc_list(chl_id);
    free_free_list(chl_id);
    free_dead_list(chl_id);

    free_alloc_list_superblk(chl_id);
    free_free_list_superblk(chl_id);
    free_dead_list_superblk(chl_id);


    return FUN_SUCCESS;
}


int init_mapping_table(int chl_id) {
    int i = 0;

    if(mapping_table == NULL) {
        mapping_table = malloc(DEVICE_BLK_NUM * sizeof(u32));
        memset(mapping_table, 0xFF, DEVICE_BLK_NUM * sizeof(u32));
    }
    else {
        memset((u32 *)mapping_table + chl_id * CHL_BLK_NUM, 0, CHL_BLK_NUM * sizeof(u32)); 
    }

    if(lba_queue[chl_id] == NULL){
        lba_queue[chl_id] = createQueue(DEVICE_BLK_NUM);
        for(i=0; i<CFG_NAND_BLOCK_NUM * CFG_NAND_PLANE_NUM * CFG_NAND_LUN_NUM; i++) {
            enqueue(lba_queue[chl_id], i+chl_id*(CFG_NAND_BLOCK_NUM * CFG_NAND_PLANE_NUM * CFG_NAND_LUN_NUM));
            //printf("enqueue block chl_id =%d, i=%d\n", chl_id, front(lba_queue[chl_id]));
            //printf("enqueue block chl_id =%d, i=%d\n", chl_id, lba_queue[chl_id]->elements[lba_queue[chl_id]->tail]);
        }
    }


    return FUN_SUCCESS;	
}


int set_mapping_entry(u32 lba, u32 ppa) {

    u32 *m_addr;

    m_addr = (u32 *)mapping_table + lba;
    *m_addr = ppa;

    return FUN_SUCCESS;

}
u32 get_ppa_addr_v(vssd_t * v, u32 lba) {

    u32 m_addr;

    if(lba < 0 || lba >= v->max_addr) {
        printf("lba address error: %d out of range\n", lba);
        return -1;
    }

    m_addr = v->mapping_table_v[lba];
    return m_addr;
}

u32 get_ppa_addr(u32 lba) {

    u32 *m_addr;

    if(lba >= DEVICE_BLK_NUM) {
        printf("lba address error: %d out of range\n", lba);
        return -1;
    }

    m_addr = (u32 *)mapping_table + lba;
    return *m_addr;
}

u32 get_lba_addr(u32 ppa) {
    u32 *m_addr;
    u32 lba = 0;

    while(lba < DEVICE_BLK_NUM) {
        m_addr = (u32 *)mapping_table + lba;

        if(*m_addr == ppa) {
            return lba;
        }
        lba++;
    }
    //get rid of warning
    return -1;
}



u32 alloc_block(int chl_id) {
    /* allocate a block from channel #chl_id */
    //int i_blk = 0;
    //int i_lun = 0;
    //int i_pln = 0; 
    u32 ppa_addr = 0;
    int lba_addr = 0;
    //int r_blk = 0;
    //int r_lun = 0;
    //int r_pln = 0;
    //int min = 0;
    //struct timeval start, end;

    //srand(time(NULL));

    //gettimeofday(&start, NULL);


    node_t *free_blk = del_free_list(chl_id);
    if(free_blk)
        ppa_addr = free_blk->ppa;
    else {
        printf("cannot get free block\n");
        exit(0);
    }


    //gettimeofday(&end, NULL);

    //printf("allocation overhead %f\n", TimeGap(&start, &end));

    //blk_status[chl_id][r_lun][r_blk][r_pln] = 1;



    //ppa_addr =   chl_id | 
    //	     (r_lun << (CH_BITS+EP_BITS+PL_BITS)) |
    //	     (r_pln << (CH_BITS+EP_BITS)) |
    //             (r_blk << (CH_BITS+EP_BITS+PL_BITS+LN_BITS+PG_BITS));


    lba_addr = dequeue(lba_queue[chl_id]);

    //printf("lba:%d ppa:%p\n", lba_addr, ppa_addr);

    if(lba_addr == -1) {
        printf("Error: cannot get LBA\n");
        exit(0);
    }

    set_mapping_entry(lba_addr, ppa_addr);
    add_alloc_list(chl_id, free_blk);

    return lba_addr;
}


int free_block(int chl_id, u32 lba) {
    /* release the block */
    int devid = 0;
    int nsid = 1;
    int qid = 1;

    int i_lun = 0; 
    int i_blk = 0; 
    int i_pln = 0;
    int i_chl = 0;

    //int r_hot_lun = 0, r_cold_lun = 0;
    //int r_hot_blk = 0, r_cold_blk = 0;
    //int r_hot_pln = 0, r_cold_pln = 0;
    //int min = 0, max = 0;
    //u32 hot_ppa, cold_ppa;
    //u32 cold_lba;
    //int page_num = 0;
    //u32 ppa_new;
    //int i = 0;

    /* erase the freed block */
    u32 ppa_addr;
    ppa_addr = get_ppa_addr(lba);

    i_blk = (ppa_addr >> (CH_BITS + EP_BITS + PL_BITS + LN_BITS + PG_BITS)) & 0xFFFF;
    i_lun = (ppa_addr >> (CH_BITS + EP_BITS + PL_BITS)) & 0x03;
    i_pln = (ppa_addr >> (CH_BITS + EP_BITS)) & 0x03;
    i_chl =  ppa_addr & 0x0F; 

    ersppa_sync(devid, nsid, ppa_addr, qid, 0); 

    //blk_status[i_chl][i_lun][i_blk][i_pln] = 0;
    blk_ers_cnt[i_chl][i_lun][i_blk][i_pln]++;
    enqueue(lba_queue[chl_id], lba);
    set_mapping_entry(lba, 0xFFFFFFFF);

    node_t* alloc_blk = find_alloc_list(chl_id, ppa_addr);

    alloc_blk->data = blk_ers_cnt[i_chl][i_lun][i_blk][i_pln];

    if(alloc_blk != NULL) {
        add_free_list(chl_id, alloc_blk);
    }

    //add_free_list(chl_id, ppa_addr, blk_ers_cnt[i_chl][i_lun][i_blk][i_pln]);

    /* switch the cold block with hot block */
    /*	
        switch_cnt++;	
        if(switch_cnt == SWITCH_THRESHOLD) {
        printf("start the swapping\n");
    // find the coldest block
    for(i_lun=0; i_lun < CFG_NAND_LUN_NUM; i_lun++) {
    for(i_blk=0; i_blk < CFG_NAND_BLOCK_NUM; i_blk++) {
    for(i_pln=0; i_pln < CFG_NAND_PLANE_NUM; i_pln++) {
    if(blk_status[chl_id][i_lun][i_blk][i_pln] == 1) {
    //blk_status[chl_id][i_lun][i_blk][i_pln] = 1;
    r_cold_blk = i_blk;
    r_cold_lun = i_lun;
    r_cold_pln = i_pln;
    min = blk_ers_cnt[chl_id][i_lun][i_blk][i_pln];
    goto FIND_COLD_BLK;
    }
    }
    }
    }  
FIND_COLD_BLK:

if(i_lun >= CFG_NAND_LUN_NUM)
return NULL;

for(i_lun=r_cold_lun; i_lun < CFG_NAND_LUN_NUM; i_lun++) {
for(i_blk=r_cold_blk; i_blk < CFG_NAND_BLOCK_NUM; i_blk++){
for(i_pln = r_cold_pln+1; i_pln < CFG_NAND_PLANE_NUM; i_pln++){
if(blk_status[chl_id][i_lun][i_blk][i_pln] == 1) {
if(blk_ers_cnt[chl_id][i_lun][i_blk][i_pln] < min){
r_cold_lun = i_lun;
r_cold_blk = i_blk;
r_cold_pln = i_pln;
min = blk_ers_cnt[chl_id][i_lun][i_blk][i_pln];
}
}
}
}
}

// find the hottest block	    
for(i_lun=0; i_lun < CFG_NAND_LUN_NUM; i_lun++) {
for(i_blk=0; i_blk < CFG_NAND_BLOCK_NUM; i_blk++) {
for(i_pln=0; i_pln < CFG_NAND_PLANE_NUM; i_pln++) {
if((blk_status[chl_id][i_lun][i_blk][i_pln] == 0) && (blk_bad[chl_id][i_lun][i_blk][i_pln] == 0)) {
    //blk_status[chl_id][i_lun][i_blk][i_pln] = 1;
    r_hot_blk = i_blk;
    r_hot_lun = i_lun;
    r_hot_pln = i_pln;
    max = blk_ers_cnt[chl_id][i_lun][i_blk][i_pln];
    goto FIND_HOT_BLK;
    }
    }
    }
    } 

FIND_HOT_BLK:

for(i_lun=r_hot_lun; i_lun < CFG_NAND_LUN_NUM; i_lun++) {
for(i_blk=r_hot_blk; i_blk < CFG_NAND_BLOCK_NUM; i_blk++){
for(i_pln = r_hot_pln+1; i_pln < CFG_NAND_PLANE_NUM; i_pln++){
if((blk_status[chl_id][i_lun][i_blk][i_pln] == 0) && (blk_bad[chl_id][i_lun][i_blk][i_pln] == 0)) {
if(blk_ers_cnt[chl_id][i_lun][i_blk][i_pln] > max){
r_hot_lun = i_lun;
r_hot_blk = i_blk;
r_hot_pln = i_pln;
max = blk_ers_cnt[chl_id][i_lun][i_blk][i_pln];
}
}
}
}
}

// set the status
blk_status[chl_id][r_cold_lun][r_cold_blk][r_cold_pln] = 0;
blk_status[chl_id][r_hot_lun][r_hot_blk][r_hot_pln] = 1;

char* buf = (char*)malloc(sizeof(char)*CFG_NAND_PAGE_SIZE*CFG_NAND_PAGE_NUM);
char* metabuf = (char*)malloc(16*CFG_NAND_PAGE_NUM);

// get the physical address of hot and cold blocks

hot_ppa =   chl_id |
(r_hot_lun << (CH_BITS+EP_BITS+PL_BITS)) |
(r_hot_pln << (CH_BITS+EP_BITS)) |
(r_hot_blk << (CH_BITS+EP_BITS+PL_BITS+LN_BITS+PG_BITS));

cold_ppa =   chl_id |
(r_cold_lun << (CH_BITS+EP_BITS+PL_BITS)) |
(r_cold_pln << (CH_BITS+EP_BITS)) |
(r_cold_blk << (CH_BITS+EP_BITS+PL_BITS+LN_BITS+PG_BITS));

cold_lba = get_lba_addr(cold_ppa);
printf("swap: hotppa=%p, coldppa=%p cold_lba = %d\n", hot_ppa, cold_ppa, cold_lba);



// read data from cold block
page_num = 0;
ppa_new = cold_ppa;

printf("start reading data block\n");

for(i = 0; i < CFG_NAND_PAGE_NUM; i++) {
    read_data_ppa(0, 1, ppa_new, 1, 3, buf + i * CFG_NAND_PAGE_SIZE, metabuf + i * 16);
    page_num++;
    ppa_new = (ppa_new & (~(0xFF << (CH_BITS + EP_BITS + PL_BITS + LN_BITS)))) |
        ((page_num) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));

}

// write data to hot block
page_num = 0;
ppa_new = hot_ppa;

for(i = 0; i < CFG_NAND_PAGE_NUM; i++) {
    write_data_ppa(0, 1, ppa_new, 1, 3, buf + i * CFG_NAND_PAGE_SIZE, metabuf + i * 16);
    page_num++;
    ppa_new = (ppa_new & (~(0xFF << (CH_BITS + EP_BITS + PL_BITS + LN_BITS)))) |
        ((page_num) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));

}

set_mapping_entry(cold_lba, hot_ppa);	 // map cold block's lba to new ppa
ersppa_sync(devid, nsid, cold_ppa, qid, 0);  //erase cold block


}

*/

return FUN_SUCCESS;
}



u32 alloc_superblock(int chl_id) {
    /* allocate a block from channel #chl_id */
    //int i_blk = 0;
    //int i_lun = 0;
    //int i_pln = 0; 
    u32 ppa_addr = 0;
    int lba_addr = 0;
    //int r_blk = 0;
    //int r_lun = 0;
    //int r_pln = 0;
    //int min = 0;
    //struct timeval start, end;

    //srand(time(NULL));

    //gettimeofday(&start, NULL);


    node_t *free_blk = del_free_list_superblk(chl_id);
    if(free_blk)
        ppa_addr = free_blk->ppa;
    else {
        printf("cannot get free block\n");
        exit(0);
    }


    //gettimeofday(&end, NULL);

    //printf("allocation overhead %f\n", TimeGap(&start, &end));



    lba_addr = dequeue(lba_queue[chl_id]);

    //printf("lba:%d ppa:%p\n", lba_addr, ppa_addr);

    if(lba_addr == -1) {
        printf("Error: cannot get LBA\n");
        exit(0);
    }

    set_mapping_entry(lba_addr, ppa_addr);
    add_alloc_list_superblk(chl_id, free_blk);

    return lba_addr;
}


int free_superblock(int chl_id, u32 lba) {
    /* release the block */
    int devid = 0;
    int nsid = 1;
    int qid = 1;

    int i_lun = 0; 
    int i_blk = 0; 
    //int i_pln = 0;
    int i_chl = 0;

    //int r_hot_lun = 0, r_cold_lun = 0;
    //int r_hot_blk = 0, r_cold_blk = 0;
    //int r_hot_pln = 0, r_cold_pln = 0;
    //int min = 0, max = 0;
    //u32 hot_ppa, cold_ppa;
    //u32 cold_lba;
    //int page_num = 0;
    //u32 ppa_new;
    //int i = 0;
    int pl_val = 0;

    /* erase the freed block */
    u32 ppa_addr;
    ppa_addr = get_ppa_addr(lba);

    i_blk = (ppa_addr >> (CH_BITS + EP_BITS + PL_BITS + LN_BITS + PG_BITS)) & 0xFFFF;
    i_lun = (ppa_addr >> (CH_BITS + EP_BITS + PL_BITS)) & 0x03;
    //i_pln = (ppa_addr >> (CH_BITS + EP_BITS)) & 0x03;
    i_chl =  ppa_addr & 0x0F; 

    for(pl_val = 0; pl_val = CFG_NAND_PLANE_NUM; pl_val++) {
        ersppa_sync(devid, nsid, ppa_addr | (pl_val << (CH_BITS + EP_BITS)), qid, 0);
    }


    //blk_status[i_chl][i_lun][i_blk][i_pln] = 0;
    superblk_ers_cnt[i_chl][i_lun][i_blk]++;
    enqueue(lba_queue[chl_id], lba);
    set_mapping_entry(lba, 0xFFFFFFFF);

    node_t* alloc_blk = find_alloc_list_superblk(chl_id, ppa_addr);

    alloc_blk->data = superblk_ers_cnt[i_chl][i_lun][i_blk];

    if(alloc_blk != NULL) {
        add_free_list_superblk(chl_id, alloc_blk);
    }


    return FUN_SUCCESS;
}



int read_page_v(vssd_t * v, u32 lba, int page_id, char *buf, char *metabuf) {
    /* read a page from block #lba */
    if(!buf || !metabuf) {
        return ERROR;
    }

    u32 ppa_addr;
    ppa_addr = get_ppa_addr_v(v,lba);
    ppa_addr = ppa_addr | ((page_id) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));

    return read_data_ppa(0, 1, ppa_addr, 1, 3, buf, metabuf);
}

int read_page(u32 lba, int page_id, char *buf, char *metabuf) {
    /* read a page from block #lba */
    if(!buf || !metabuf) {
        return ERROR;
    }

    u32 ppa_addr;
    ppa_addr = get_ppa_addr(lba);
    ppa_addr = ppa_addr | ((page_id) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));

    return read_data_ppa(0, 1, ppa_addr, 1, 3, buf, metabuf);
}


int read_data(u32 lba, int page_id, char *buf, char *metabuf, int page_num) {
    /* read #page_num pages from #page_id at block #lba */
    if(!buf || !metabuf) {
        return ERROR;
    }

    u32 paddr = get_ppa_addr(lba);

    int pnum = page_id;
    int i = 0;

    //pnum = (paddr >> 10) & 0xFF;
    if (page_num > (CFG_NAND_PAGE_NUM - page_id) )
        return ERROR;

    paddr = paddr | ((pnum) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));

    for(i=0; i<page_num; i++){
        read_data_ppa(0, 1, paddr, 1, 3, buf + i*CFG_NAND_PAGE_SIZE, metabuf + i*16);
        pnum++;
        paddr = (paddr & (~(0xFF << (CH_BITS + EP_BITS + PL_BITS + LN_BITS)))) | 
            ((pnum) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));	

    }
    return FUN_SUCCESS;

}

int read_block(u32 lba, char *buf, char *metabuf) {
    /* read a block */
    if(!buf || !metabuf) {
        return ERROR;
    }	

    int page_num = 0;
    int i = 0;
    u32 paddr = get_ppa_addr(lba);

    for(i = 0; i < CFG_NAND_PAGE_NUM; i++) {
        read_data_ppa(0, 1, paddr, 1, 3, buf + i * CFG_NAND_PAGE_SIZE, metabuf + i * 16);
        page_num++;
        paddr = (paddr & (~(0xFF << (CH_BITS + EP_BITS + PL_BITS + LN_BITS)))) | 
            ((page_num) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));	

    }

    return FUN_SUCCESS;
}

int write_block_v(vssd_t * v, u32 lba, char *buf, char *metabuf) {
    /* write a block */
    if(!buf || !metabuf) {
        return ERROR;
    }
    int page_num = 0;
    int i = 0;
    u32 paddr = get_ppa_addr_v(v, lba);

    for(i = 0; i < CFG_NAND_PAGE_NUM; i++) {
        write_data_ppa(0, 1, paddr, 1, 3, buf + i * CFG_NAND_PAGE_SIZE, metabuf + i * 16);
        page_num++;
        paddr = (paddr & (~(0xFF << (CH_BITS + EP_BITS + PL_BITS + LN_BITS)))) | 
            ((page_num) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));	

    }

    return FUN_SUCCESS;
}



int write_block(u32 lba, char *buf, char *metabuf) {
    /* write a block */
    if(!buf || !metabuf) {
        return ERROR;
    }
    int page_num = 0;
    int i = 0;
    u32 paddr = get_ppa_addr(lba);

    for(i = 0; i < CFG_NAND_PAGE_NUM; i++) {
        write_data_ppa(0, 1, paddr, 1, 3, buf + i * CFG_NAND_PAGE_SIZE, metabuf + i * 16);
        page_num++;
        paddr = (paddr & (~(0xFF << (CH_BITS + EP_BITS + PL_BITS + LN_BITS)))) | 
            ((page_num) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));	

    }

    return FUN_SUCCESS;
}


int write_page(u32 lba, int page_id, char *buf, char *metabuf) {
    /* write a page */
    if(!buf || !metabuf) {
        return ERROR;
    }

    u32 ppa_addr;
    ppa_addr = get_ppa_addr(lba);
    ppa_addr = ppa_addr | ((page_id) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));


    return write_data_ppa(0, 1, ppa_addr, 1, 3, buf, metabuf);
}


int write_data(u32 lba, int page_id, char *buf, char *metabuf, int page_num) {
    /* write multiple pages in a block */
    if(!buf || !metabuf) {
        return ERROR;
    }
    int pnum = page_id;
    int i = 0;
    u32 paddr = get_ppa_addr(lba);

    //pnum = (paddr >> 10) & 0xFF;
    if (page_num > (CFG_NAND_PAGE_NUM - page_id) )
        return ERROR;

    paddr = paddr | ((pnum) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));


    for(i=0; i<page_num; i++) {
        write_data_ppa(0, 1, paddr, 1, 3, buf + i*CFG_NAND_PAGE_SIZE, metabuf + i*16);
        pnum++;
        paddr = (paddr & (~(0xFF << (CH_BITS + EP_BITS + PL_BITS + LN_BITS)))) | 
            ((pnum) << (CH_BITS + EP_BITS + PL_BITS + LN_BITS));	
    }

    return FUN_SUCCESS;

} 


int write_data_ppa(int devid, u32 nsid, u32 ppa_addr, u16 qid, int nlb, char *buf, char *metabuf) {
    /* basic function for writing pages */
    int ret = 0;
    int fd = 0;
    int cmd = BFLEX_IOCTL_PPA_SYNC;
    struct nvme_ppa_command cmd_para;
    char name[DEV_NAME_SIZE];
    u16 instance = 0;

    memset(&cmd_para, 0 , sizeof(cmd_para));
    cmd_para.apptag = qid;  //default qid
    cmd_para.nsid = nsid;    //default nsid
    cmd_para.appmask = ADDR_FIELDS_SHIFT_EP; // default EP mode
    cmd_para.control = NVME_SINGLE_PLANE;
    cmd_para.nlb = nlb;

    if(nlb > 16) {
        cmd_para.appmask = ADDR_FIELDS_SHIFT_CH;
    }

    if (cmd_para.appmask == ADDR_FIELDS_SHIFT_CH) {
        cmd_para.control = NVME_SINGLE_PLANE; // CH mode, we must be SINGLE_PLANE due to PPA list size limited to 64
    } else if ((cmd_para.nlb + 1) > CFG_NAND_PLANE_NUM * CFG_DRIVE_EP_NUM) {
        cmd_para.control = NVME_QUART_PLANE;  // EP mode, we have PPA for more than one plane
    }

    cmd_para.start_list = (u64)ppa_addr;
    cmd_para.opcode = nvme_cmd_wrppa;
    cmd_para.prp1 = (s64)buf;
    cmd_para.metadata = (s64)metabuf;

    snprintf(name, DEV_NAME_SIZE, "%s%d", BFLEX_DEV, instance);
    fd = open(name, O_RDWR);
    if (0 > fd) {
        perror("open bflex0");
        ret = ERROR;
        return ret;
    }

    ret = ioctl(fd, cmd, &cmd_para);
    //parse_cmd_returnval(cmd, ret, instance, cmd_para.apptag);

    close(fd);

    return ret;
}


int read_data_ppa(int devid, u32 nsid, u32 ppa_addr, u16 qid, int nlb, char *buf, char *metabuf) {

    int ret = 0;
    int fd = 0;
    int cmd = BFLEX_IOCTL_PPA_SYNC;
    struct nvme_ppa_command cmd_para;
    char name[DEV_NAME_SIZE];
    u16 instance = 0;

    memset(&cmd_para, 0 , sizeof(cmd_para));
    cmd_para.apptag = qid;  //default
    cmd_para.nsid = nsid;  //default
    cmd_para.appmask = ADDR_FIELDS_SHIFT_EP; // default EP mode
    cmd_para.nlb = nlb;

    cmd_para.start_list = (u64)ppa_addr;
    cmd_para.opcode = nvme_cmd_rdppa;

    snprintf(name, DEV_NAME_SIZE, "%s%d", BFLEX_DEV, instance);
    fd = open(name, O_RDWR);
    if (0 > fd) {
        perror("open bflex0");
        ret = ERROR;
        return ret;
    }

    cmd_para.prp1 = (s64)buf;
    cmd_para.metadata = (s64)metabuf;

    ret = ioctl(fd, cmd, &cmd_para);
    //parse_cmd_returnval(cmd, ret, instance, cmd_para.apptag);

    //printf("%s\n", buf);
    //printf("metadata\n%s\n", metabuf);
    close(fd);	
    return FUN_SUCCESS;

}



int is_allocated(int chl_id) {

    if((chl_allocated & (1 << chl_id)) > 0) {
        return FUN_SUCCESS;
    }
    else
        return FUN_FAILURE;
}



void set_chl_allocated(int chl_id) {

    chl_allocated = chl_allocated | (1 << chl_id);
    return;
}



void set_chl_free(int chl_id) {

    chl_allocated = chl_allocated & (~(1 << chl_id));
    return; 
}


void erase_blk_v(uint32_t ppa_addr) {
    int devid = 0;
    u32 nsid = 1;
    u16 qid = 1;

    ersppa_sync(devid, nsid, ppa_addr, qid, 0);
}

int erase_all_blk_chl(int chl_id) {

    int i_blk = 0;
    int result = 1; 
    //long para = 0; 
    //int fd = 0;
    //int cmd = BFLEX_IOCTL_PPA_SYNC; 
    //char *endptr; 
    //int opt = 0;
    //struct nvme_ppa_command cmd_para; 
    //char name[DEV_NAME_SIZE];
    int devid = 0;
    //u16 instance = 0; 
    u32 nsid = 1;
    u16 qid = 1;
    u16 chmask_sw = 0xffff;
    //u16 *badbin = NULL;
    u32 file_len;

    int lun_val, pl_val;
    int flag = GOOD_PPA;
    //int perfect_block = GOOD_PPA;
    u32 ppa_addr = 0;

    u32 channel; 
    u16 chmask_hw;
    u16 channel_mask;

    //printf("erase_all_blk_chl is called\n");	
    // read badblock bitmap
    //TODO update this!! 
    char* path = concat(BLOCKFLEX_DIR, "bb_ben.bin");
    badbin = (u16*)parse_read_file(path, &file_len);
    free(path);	
    if(badbin == NULL) {
        printf("read file failed\n");
        return ERROR;
    }

    if(file_len < CFG_NAND_BLOCK_NUM * CFG_NAND_LUN_NUM * CFG_NAND_CHANNEL_NUM / 8) {
        printf("file length is too short. \n");
        result = -EIO;
        free(badbin);
        exit(0);
    }

    read_nvme_reg32(devid, CHANNEL_COUNT, &channel);
    if(channel == 4) {
        chmask_hw = CHANNEL_MASK_4;
    } else if(channel == 8) {
        chmask_hw = CHANNEL_MASK_8;
    } else {
        chmask_hw = CHANNEL_MASK_16;
    }
    channel_mask = chmask_sw & chmask_hw;


    //struct timeval start, end;
    srand(time(NULL));

    //gettimeofday( &start, NULL);

    for(i_blk = 0; i_blk < CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK; i_blk++) {

        for(lun_val = 0; lun_val < CFG_NAND_LUN_NUM; lun_val++) {
            /*
               ppa_addr =   chl_id |
               (lun_val << (CH_BITS+EP_BITS+PL_BITS)) |
               (i_blk << (CH_BITS+EP_BITS+PL_BITS+LN_BITS+PG_BITS));

               flag = skip_ppa(ppa_addr, badbin, channel_mask);

               if (flag == BAD_PPA) {
            //PRINT("%d, ", chl_id);
            //blk_bad[chl_id][lun_val][i_blk][0] = 1;
            continue;
            } else {
            for(pl_val = 0; pl_val < CFG_NAND_PLANE_NUM; pl_val++) {
            ersppa_sync(devid, nsid, ppa_addr | (pl_val << (CH_BITS+EP_BITS)), qid, 0); 
            }
            }
            */

            for(pl_val = 0; pl_val < CFG_NAND_PLANE_NUM; pl_val++) {
                ppa_addr =   chl_id |
                    (lun_val << (CH_BITS+EP_BITS+PL_BITS)) |
                    (i_blk << (CH_BITS+EP_BITS+PL_BITS+LN_BITS+PG_BITS)) |
                    (pl_val << (CH_BITS+EP_BITS));

                flag = skip_ppa(ppa_addr, badbin, channel_mask);
                if(flag == BAD_PPA) {
                    blk_bad[chl_id][lun_val][i_blk][pl_val] = 1;
                    continue;
                }
                else {
                    ersppa_sync(devid, nsid, ppa_addr, qid, 0);

                    node_t* node = malloc(sizeof(node_t));
                    node->ppa = ppa_addr;
                    node->data = 0;

                    add_free_list(chl_id, node);
                }

            }

        }	

    }

    for(i_blk = CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK; i_blk < CFG_NAND_BLOCK_NUM; i_blk++) {
        for(lun_val = 0; lun_val < CFG_NAND_LUN_NUM; lun_val++) {
            ppa_addr = chl_id | 
                (lun_val << (CH_BITS + EP_BITS + PL_BITS)) |
                (i_blk << (CH_BITS + EP_BITS + PL_BITS + LN_BITS + PG_BITS));
            flag = skip_ppa(ppa_addr, badbin, channel_mask);

            if(flag == BAD_PPA) {
                superblk_bad[chl_id][lun_val][i_blk - (CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK)] = 1;
                continue;
            }
            else {
                for(pl_val = 0; pl_val = CFG_NAND_PLANE_NUM; pl_val++) {
                    ersppa_sync(devid, nsid, ppa_addr | (pl_val << (CH_BITS + EP_BITS)), qid, 0);
                }

                node_t* node = malloc(sizeof(node_t));
                node->ppa = ppa_addr;
                node->data = 0;

                add_free_list_superblk(chl_id, node);	

            }			

        }
    }


    //gettimeofday( &end, NULL);

    //printf("erase time %f millisec\n", TimeGap(&start, &end));

    return result;	
}


int erase_all_blk_chl_v(int chl_id, vssd_t * v) {
    return erase_all_blk_chl_v_debug(chl_id, v, 0);
}

int erase_all_blk_chl_v_debug(int chl_id, vssd_t * v, int debug){

    int i_blk = 0;
    int result = 1; 
    //long para = 0; 
    //int fd = 0;
    //int cmd = BFLEX_IOCTL_PPA_SYNC; 
    //char *endptr; 
    //int opt = 0;
    //struct nvme_ppa_command cmd_para; 
    //char name[DEV_NAME_SIZE];
    int devid = 0;
    //u16 instance = 0; 
    u32 nsid = 1;
    u16 qid = 1;
    u16 chmask_sw = 0xffff;
    //u16 *badbin = NULL;
    u32 file_len;

    int lun_val, pl_val;
    int flag = GOOD_PPA;
    //int perfect_block = GOOD_PPA;
    u32 ppa_addr = 0;

    u32 channel; 
    u16 chmask_hw;
    u16 channel_mask;

    //printf("erase_all_blk_chl is called\n");	
    // read badblock bitmap
    //TODO update this!! 
    char* path = concat(BLOCKFLEX_DIR, "mark_toshA19.bin");
    badbin = (u16*)parse_read_file(path, &file_len);	
    free(path);
    if(badbin == NULL) {
        printf("read file failed\n");
        return ERROR;
    }

    if(file_len < CFG_NAND_BLOCK_NUM * CFG_NAND_LUN_NUM * CFG_NAND_CHANNEL_NUM / 8) {
        printf("file length is too short. \n");
        result = -EIO;
        free(badbin);
        exit(0);
    }

    read_nvme_reg32(devid, CHANNEL_COUNT, &channel);
    if(channel == 4) {
        chmask_hw = CHANNEL_MASK_4;
    } else if(channel == 8) {
        chmask_hw = CHANNEL_MASK_8;
    } else {
        chmask_hw = CHANNEL_MASK_16;
    }
    channel_mask = chmask_sw & chmask_hw;


    //struct timeval start, end;
    srand(time(NULL));

    //gettimeofday( &start, NULL);

    for(i_blk = 0; i_blk < CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK; i_blk++) {

        for(lun_val = 0; lun_val < CFG_NAND_LUN_NUM; lun_val++) {
            for(pl_val = 0; pl_val < CFG_NAND_PLANE_NUM; pl_val++) {
                ppa_addr =   chl_id |
                    (lun_val << (CH_BITS+EP_BITS+PL_BITS)) |
                    (i_blk << (CH_BITS+EP_BITS+PL_BITS+LN_BITS+PG_BITS)) |
                    (pl_val << (CH_BITS+EP_BITS));

                flag = skip_ppa(ppa_addr, badbin, channel_mask);
                if(flag == BAD_PPA) {
                    blk_bad[chl_id][lun_val][i_blk][pl_val] = 1;
                    continue;
                }
                else {
                    if(!debug)
                        ersppa_sync(devid, nsid, ppa_addr, qid, 0);

                    node_t* node = malloc(sizeof(node_t));
                    node->ppa = ppa_addr;
                    node->data = 0;

                    //add_free_list(chl_id, node);
                    add_list_v(&v->free_block_list[chl_id], node);
                }

            }

        }	

    }

    //for(i_blk = CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK; i_blk < CFG_NAND_BLOCK_NUM; i_blk++) {
    //    for(lun_val = 0; lun_val < CFG_NAND_LUN_NUM; lun_val++) {
    //        ppa_addr = chl_id | 
    //            (lun_val << (CH_BITS + EP_BITS + PL_BITS)) |
    //            (i_blk << (CH_BITS + EP_BITS + PL_BITS + LN_BITS + PG_BITS));
    //        flag = skip_ppa(ppa_addr, badbin, channel_mask);

    //        if(flag == BAD_PPA) {
    //            superblk_bad[chl_id][lun_val][i_blk - (CFG_NAND_BLOCK_NUM - RESERVED_SUPERBLK)] = 1;
    //            continue;
    //        }
    //        else {
    //            for(pl_val = 0; pl_val = CFG_NAND_PLANE_NUM; pl_val++) {
    //                ersppa_sync(devid, nsid, ppa_addr | (pl_val << (CH_BITS + EP_BITS)), qid, 0);
    //            }

    //            node_t* node = malloc(sizeof(node_t));
    //            node->ppa = ppa_addr;
    //            node->data = 0;

    //            add_free_list_superblk(chl_id, node);	

    //        }			

    //    }
    //}


    //gettimeofday( &end, NULL);

    //printf("erase time %f millisec\n", TimeGap(&start, &end));

    return result;	
}
