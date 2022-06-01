#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <getopt.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include "bflex.h"
#include "usage.h"
#include "blklist.h"
#include "trace.h"
#include "vssd.h"
// #include "../ocssd/ocssd_queue.h"

//stdount, #include "channel.h"

//Block size is 16 KB page size * 256 pages = 4096 KB
#define BLK_NUM 30
#define FUN_SUCCESS 0
//First possible start channel is 0, last possible is 15
#define START_CHL 0
#define N_CHL 1
#define M_CHL 16
#define CHL_BW 70
#define CHL_SZ 64
inline double TimeGap( struct timeval* t_start, struct timeval* t_end ) {
    return (((double)(t_end->tv_sec - t_start->tv_sec) * 1000.0) +
            ((double)(t_end->tv_usec - t_start->tv_usec) / 1000.0));
}
#define MAX(x,y) (((x) >= (y)) ? (x) : (y))
#define MIN(x,y) (((x) <= (y)) ? (x) : (y))

static int ch_mapping[2][16];
static pthread_mutex_t r_lock[16];
static pthread_mutex_t g_lock[16];

inline void help_dist(int tot_chl, int target, int start, int ind) {
    int rem = tot_chl;
    int rem_chl = target;
    while(rem_chl > 0) {
        ch_mapping[ind][start] = rem/rem_chl;
        rem-=ch_mapping[ind][start++];
        rem_chl--;
    }
}

char* concat(const char *s1, const char *s2)
{
    char *result = malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

typedef struct _thread_data_t {
    int index;
    int* chls;
    int ch_num;
    vssd_t * vssd;
    uint32_t * mapping;
    trace_t* t;
    uint32_t * read_bw;
    uint32_t * write_bw;
    uint32_t * iops;
    uint32_t * alloc;
} thread_data_t;
static const int PAGE_SZ = PAGE_SIZE*4;
static const int BLK_SZ = PAGE_SIZE*4*256;
static const int BLK_SZ_META = META_SIZE*4*256;
static const uint32_t MAX_SECTOR = 1 << 27; // (1 << 32) / PAGE_SZ; or 1 << 18;
static const int ops = 250;
static int end_val = 4000;
#ifdef DEBUG 
    static const int interval = 15; 
    int iter = 4; 
#else
    static const int interval = 180; 
    int iter = 10; // TODO: tune this SMALL=10 LARGE=40 
#endif
char *buf;
char *metabuf;
char *readbuf;
char *readmetabuf;

mqd_t mqfd, mqfd_0, mqfd_1;
int sz_fd, bw_fd;
char* sz_shared_mem, *bw_shared_mem;
char* path;


void * run_trace_realtime(void * args) {

    thread_data_t * data = (thread_data_t *) args;
    vssd_t * v = data->vssd;
    int chl_num = data->ch_num;
    uint32_t * mapping = data->mapping;
    int *chls = data->chls;
    // trace_t * t = data->t;
    //wrr_data * wrr = data->wrr;
    //For timing
    struct timeval start, end;
    int i,j,k;

    //Prep some blocks for reading
    int read_prep = 50; 
    uint32_t * prepped_reads = malloc(read_prep * 1024 * sizeof(uint32_t));
    memset(prepped_reads, 0, read_prep * 1024 * sizeof(uint32_t));
    int read_wrap = read_prep * 1024;
    int cur_read = 0;

    //Prep the read blocks
    //Do so for each channel
    for(k = 0; k < chl_num; k++) {
        int index = chls[k];
        for(i = 0; i < read_prep; i++) {
            uint32_t lba = alloc_block_v(v, index);
            for(j = 0; j < 1024; j++) {
                int cur_ind = i * 1024 + j;
                prepped_reads[cur_ind] = lba * 1024 + j;
            }
            write_block_v(v,lba, buf, metabuf);
        }
    }
    //Finish prepping read blocks
    event_t * next_cur;
    //counter for the total number of events processed, total is for the number of writes
    //used for sanity checking
    int counter = 0;
    gettimeofday(&start, NULL);
    double diff;
    int hist_index = 0;
    int iops = 0;
    int reads = 0;
    int writes = 0;
    int new_writes = 0;
    int prev_join = hist_index;
    int cur_ind = 0;
    int cur_chl = chls[cur_ind];
    int cur_page = 0;
    uint32_t cur_lba = alloc_block_v(v, cur_chl);
    //Main LOOP
top:
    while(1) {
        //Grab more work
        Req req;
        struct timespec ts;
        ts.tv_nsec = 1;
        int status = -1;
        int to_print = 1;
        while(status == -1){
            if (v->vssd_type == regular) mqfd = mqfd_1;
            else mqfd = mqfd_0;
            status = mq_receive(mqfd, (char*) &req, MAX_MSG, 0);
            if (status == -1) {
                perror("mq_receive failure\n");
            }
            // else {
            //     printf("mq_receive successful\n");
            //     printf("OCSSD received request <id, mode, off, len, chn, tid>: <%lu, %d, %lu, %u, %d, %d> \n", req.vssd_id, req.mode, req.offset, req.length, cur_chl, data->index);
            // }
        }

        if (counter % 50 == 0) {
            gettimeofday(&end, NULL);
            diff = TimeGap(&start, &end);
            int cur_index = (int)(diff/1000);
            if (cur_index != prev_join) {
                data->read_bw[prev_join] = reads;
                data->write_bw[prev_join] = writes;
                data->iops[prev_join] = iops;
                data->alloc[prev_join] = new_writes;
                if (cur_index >= interval) {
                    // if (v->vssd_type == regular) printf("Regular finished\n");
                    // else printf("Harvest finished\n");
                    pthread_exit(NULL);
                }
                reads = 0;
                writes = 0;
                iops = 0;
                new_writes = 0;
                prev_join = cur_index;
            }
        }
        counter++;
        iops++;
        //Process the event, have type, offset, size of interest
        if (req.mode == 0) {
            //Grab these locally for later so we don't affect the master copy
            uint32_t size = req.length / PAGE_SZ;
            uint32_t off = req.offset / PAGE_SZ;
            if(size < 0 || size > 100000){
                printf("Interesting read <%lu, %u>\n", off, size);
            }
            reads += size;
            //Read each component page
            while(size > 0) {
                if(off >= MAX_SECTOR){
                    printf("Offset %u exceeds the sector boundary %u\n", off, MAX_SECTOR);
                    size--;
                    off++;
                    continue;
                }
                uint32_t lba = mapping[off];
                if(lba == 0){
                    //Grab the mapped page
                    lba = prepped_reads[cur_read];
                    cur_read = (cur_read + 1) % read_wrap;
                    //Get the block and page indices from our mapped value, need these for the read func
                }
                uint32_t block_ind = lba/1024;
                uint32_t page_ind = lba%1024;
                if (v->vssd_type == harvest) {
#ifndef DEBUG
                    read_page_v(v, block_ind, page_ind, readbuf, readmetabuf);
#endif
                } else {
                    // node_t * temp = malloc(sizeof(node_t));
                    // temp->ppa = cur_lba;
                    // temp->rw = 0;
                    // temp->data = page_ind;
                    if(v->vssd_type == pghost) {
                        pthread_mutex_lock(&g_lock[cur_chl]);
                        //if (pthread_mutex_trylock(&g_lock[cur_chl])) {
                            //add_list_v(&g_list[cur_chl], temp); 
#ifndef DEBUG 
                            read_page_v(v,block_ind, page_ind, readbuf, readmetabuf);
#endif
                            pthread_mutex_unlock(&g_lock[cur_chl]);
                        //} else {
                            // reads-=size;
                            // free(temp);
                        //    continue;
                        //}
                    } else {
                        pthread_mutex_lock(&r_lock[cur_chl]);
                        //add_list_v(&r_list[cur_chl], temp); 
#ifndef DEBUG 
                        read_page_v(v,block_ind, page_ind, readbuf, readmetabuf);
#endif
                        pthread_mutex_unlock(&r_lock[cur_chl]);
                    }
                }
                //One page read completed
                size--;
                off++;
            }
        } else {
            //Grab these locally for later so we don't affect the master copy
            uint32_t size = req.length / PAGE_SZ;
            uint32_t off = req.offset / PAGE_SZ;
            if(size < 0 || size > 100000){
                printf("Interesting write <%lu, %u>\n", off, size);
            }
            //Now we handle the mappings
            while(size > 0) {
                if(off >= MAX_SECTOR){
                    printf("Offset %u exceeds the sector boundary %u\n", off, MAX_SECTOR);
                    size--;
                    // cur_page++;
                    off++;
                    continue;
                }
                if(mapping[off] == 0) new_writes++;
                mapping[off] = cur_lba * 1024 + cur_page;
                cur_page++;
                //check if we have filled a block, in this case we submit a block write
                if (cur_page == 1024) {
                    if (v->vssd_type == harvest) {
#ifndef DEBUG 
                        write_block_v(v,cur_lba, buf, metabuf);
#endif
                    }else {
                        // node_t * temp = malloc(sizeof(node_t));
                        // temp->ppa = cur_lba;
                        // temp->rw = 1;
                        if(v->vssd_type == pghost) {
                            pthread_mutex_lock(&g_lock[cur_chl]);
                            // if (pthread_mutex_trylock(&g_lock[cur_chl])) {
                                // add_list_v(&g_list[cur_chl], temp);
#ifndef DEBUG 
                                write_block_v(v,cur_lba, buf, metabuf);
#endif
                                pthread_mutex_unlock(&g_lock[cur_chl]);
                           // } else {
                                // free(temp);
                           //     cur_page--;
                            //    continue;
                           // }
                        } else {
                            pthread_mutex_lock(&r_lock[cur_chl]);
                            //add_list_v(&r_list[cur_chl], temp); 
#ifndef DEBUG 
                            write_block_v(v,cur_lba, buf, metabuf);
#endif
                            pthread_mutex_unlock(&r_lock[cur_chl]);
                        }
                    }
                    writes += 1024;
                    // printf("Write size: %d; Current LBA: %u\n", writes, cur_lba);
                    cur_lba = alloc_block_v(v, cur_chl);
                    cur_page = 0;
                }
                size--;
                off++;
            }
        }
        //Go to the next event
        // next_cur = t->head[cur_chl]->next;
        // free(t->head[cur_chl]);
        // t->head[cur_chl] = next_cur;
        cur_ind = (cur_ind + 1) % chl_num;
        cur_chl = chls[cur_ind];
    }
    //fprintf(stdout, "Thread %d is done and exiting at %d\n", data->index, hist_index);
    //fflush(stdout);
    pthread_exit(NULL);
}


int main(int argc, char* argv[]) {
    int chl_r = 8;
    int chl_h = 8;
    int chl_g = chl_r/2;
    //int chl_g = 0;
    vssd_t *r1, *h1, *g1, *p1;
    //ghost borrows the trace from "harvest"
    trace_t * t_r, * t_h;
    //struct timeval start, end;
    buf = (char*)malloc(BLK_SZ);
    metabuf = (char*)malloc(BLK_SZ_META);
    readbuf = (char*)malloc(BLK_SZ);
    readmetabuf = (char*)malloc(BLK_SZ_META);

    int i,j;
    for(i=0; i<BLK_SZ; i++)  buf[i] = 'x';
    for(i=0; i<BLK_SZ_META; i++) metabuf[i] = 'm';
    
    r1 = alloc_regular_vssd(chl_r);
    h1 = alloc_regular_vssd(chl_h);
    //Should be maximum inter
    //r1->vssd_type = harvest;
    //Needed to avoid priority queue
    h1->vssd_type = harvest;

    //Needed to harvest a meaningful amount
    int min = 1 << 30;
    for(i = 0; i < chl_r; i++) {
        int val = list_len_v(r1->free_block_list[i]);
        //printf("%d\n", val);
        if (val < min) min = val;
    }
    //No better dur for the moment 
    int bogus_dur = 100;
    g1 = alloc_ghost_vssd(r1, chl_g, chl_g * (min/2), bogus_dur);
    p1 = alloc_pghost_vssd(g1);

    //for(i = 0; i < chl_r; i++) {
    //    int val = list_len_v(r1->free_block_list[i]);
    //    printf("%d\n", val);
    //}

    printf("Allocation finishes\n");
    ssize_t status;
    struct mq_attr attr = {.mq_maxmsg=MAX_MSG, .mq_msgsize = MAX_MQUEUE_MSG_SIZE};

    // read only
    mqfd_0 = mq_open(QUEUE_NAME_0, O_RDONLY|O_CREAT, PMODE, &attr);
    if(mqfd_0 == -1) {
        perror("Parent mq_open failure");
        exit(0);
    }
    mqfd_1 = mq_open(QUEUE_NAME_1, O_RDONLY|O_CREAT, PMODE, &attr);
    if(mqfd_1 == -1) {
        perror("Parent mq_open failure");
        exit(0);
    }

    //Set number of threads
    int thr_r = 0;
    int thr_h = 8;
    int thr_g = 0;
    int next_r = 0;
    int next_g = 0;
    if (argc == 2) {
        //in_file_r = argv[1];
    }
    else if (argc == 3) {
        //in_file_r = argv[1];
        //end_val = atoi(argv[2]);
        next_r = atoi(argv[1]);
        next_g = atoi(argv[2]);
    }

    srand(time(NULL));
    path = concat(BLOCKFLEX_DIR, "sz_inputs.txt");
    if ((sz_fd = open(path, O_RDWR|O_CREAT, S_IRWXU)) == -1)
    {
        printf("unable to open shared file sz_inputs.txt\n");
        return 0;
    }
    dprintf(sz_fd, "%010d %011.5lf %011.5lf %011.5lf %011.5lf", 0, (double)rand()/rand(), (double)rand()/rand(), (double)rand()/rand(), (double)rand()/rand());
    sz_shared_mem = (char*) mmap(NULL, 8, PROT_READ | PROT_WRITE, MAP_SHARED, sz_fd, 0);
    
    path = concat(BLOCKFLEX_DIR, "bw_inputs.txt");
    if ((bw_fd = open(path, O_RDWR|O_CREAT, S_IRWXU)) == -1)
    {
        printf("unable to open shared file bw_inputs.txt\n");
        return 0;
    }
    dprintf(bw_fd, "%010d %010d %010d %010d %010d %010d %010d", 0, rand(), rand(), rand(), rand(), rand(), rand());
    bw_shared_mem = (char*) mmap(NULL, 8, PROT_READ | PROT_WRITE, MAP_SHARED, bw_fd, 0);
    free(path);

    fprintf(stdout, "Configuration is: %d regular channels, %d harvest channels, %d ghost channels\n", chl_r, chl_h, chl_g);
    fprintf(stdout, "Initial configuration is: %d regular threads, %d harvest threads, %d ghost threads\n", thr_r, thr_h, thr_g);
    fprintf(stdout, "Harvesting configuration is: %d regular threads, %d harvest threads, %d ghost threads\n", next_r, thr_h, next_g);
    fflush(stdout);

    //Worst case is 32 since we could have harvesting for each channel, unlikely to actually hit tho
    thread_data_t thr_data[32];
    int rc;
    pthread_t threads[32];

    //Mapping tables for each
    //reduced memory consumption
    uint32_t * mapping_r = malloc(MAX_SECTOR* sizeof(uint32_t));
    memset(mapping_r, 0, MAX_SECTOR * sizeof(uint32_t));
    uint32_t * mapping_h = malloc(MAX_SECTOR* sizeof(uint32_t));
    memset(mapping_h, 0, MAX_SECTOR * sizeof(uint32_t));
    uint32_t * mapping_g = malloc(MAX_SECTOR* sizeof(uint32_t));
    memset(mapping_g, 0, MAX_SECTOR * sizeof(uint32_t));

    
    int * tot_reads_r = calloc(interval * iter,sizeof(int));
    int * tot_writes_r = calloc(interval * iter,sizeof(int));
    int * tot_reads_g = calloc(interval * iter,sizeof(int));
    int * tot_writes_g = calloc(interval * iter,sizeof(int));
    int * tot_reads_h = calloc(interval * iter,sizeof(int));
    int * tot_writes_h = calloc(interval * iter,sizeof(int));
    int cnt = 0;
    double total_alloc = 0, prev_alloc = 0;
    for(i = 0; i < 16; i++) {
        pthread_mutex_init(&r_lock[i],NULL);
        pthread_mutex_init(&g_lock[i],NULL);
        //g_list[i] = NULL;
        //r_list[i] = NULL;
    }

    while(cnt < iter) {
        fprintf(stdout, "Starting Iteration: %d\n", cnt);
        fflush(stdout);
        int rchl = 0;
        int hchl = 0;
        // int thr_h = h_vals[cnt];
        if (cnt >= iter / 2) {
            thr_r = next_r;
            thr_g = next_g;
            path = concat(BLOCKFLEX_DIR, "start_harvest");
            int fd = open(path, O_WRONLY | O_APPEND | O_CREAT, 0644);
            close(fd);
            free(path);
            fprintf(stdout, "Starting Harvesting: %d\n", cnt);
            fflush(stdout);
        }
        help_dist(chl_r, thr_r, 0, 0);
        help_dist(chl_h, thr_h, thr_r, 1);
        //TODO erase the written blocks between runs so we dont run into space issues
        for(i = 0; i < thr_r + thr_h + thr_g; i++) {
            thr_data[i].index = i;
            thr_data[i].read_bw = calloc(interval, sizeof(int));
            thr_data[i].write_bw = calloc(interval, sizeof(int));
            thr_data[i].iops = calloc(interval, sizeof(int));
            thr_data[i].alloc = calloc(interval, sizeof(int));
            if (i < thr_r) {
                //We are a 'regular' VM thread
                thr_data[i].ch_num = ch_mapping[0][i];
                thr_data[i].chls = malloc(ch_mapping[0][i] * sizeof(int));
                for(j = 0; j < ch_mapping[0][i]; j++) {
                    thr_data[i].chls[j] = rchl++;
                }
                thr_data[i].mapping = mapping_r;
                thr_data[i].vssd = r1;
                // thr_data[i].t = t_r;
                //thr_data[i].wrr = &rr_data[i];
            }
            else if (i < thr_r + thr_h) {
                //We are a 'harvest' VM thread
                thr_data[i].ch_num = ch_mapping[1][i];
                thr_data[i].chls = malloc(ch_mapping[1][i] * sizeof(int));
                for(j = 0; j < ch_mapping[1][i]; j++) {
                    thr_data[i].chls[j] = chl_r + hchl;
                    hchl++;
                }
                thr_data[i].mapping = mapping_h;
                thr_data[i].vssd = h1;
                // thr_data[i].t = t_h;
            } else {
                //We are a ghost thread
                //For now lets always one to one map them
                thr_data[i].ch_num = 1;
                thr_data[i].chls = malloc(1 * sizeof(int));
                thr_data[i].chls[0] = p1->allocated_chl[i-thr_r-thr_h];
                thr_data[i].mapping = mapping_g;
                thr_data[i].vssd = p1;
                // thr_data[i].t = t_h;
            }
            if ((rc = pthread_create(&threads[i], NULL, run_trace_realtime, &thr_data[i]))) {
                fprintf(stderr, "error: pthread_create, rc: %d\n", rc);
                return EXIT_FAILURE;
            }
        }
        for(i = 0; i < thr_r + thr_h + thr_g; i++) {
            pthread_join(threads[i], NULL);
        }
        for(i = 0; i < thr_r + thr_h + thr_g; i++) {
            for(j = 0; j < interval; j++) {
                if (i < thr_r) {
                    tot_reads_r[cnt*interval+j] += thr_data[i].read_bw[j];
                    tot_writes_r[cnt*interval+j] += thr_data[i].write_bw[j];
                }
                else {
                    tot_reads_h[cnt*interval+j] += thr_data[i].read_bw[j];
                    tot_writes_h[cnt*interval+j] += thr_data[i].write_bw[j];
                }
            }
        }

        // Write to predictor
        // int max min avg delta of the alloc storage (in channels) 
        int max_bw = 0, min_bw = INT32_MAX, avg_bw = 0;
        int max_iops = 0, min_iops = INT32_MAX, avg_iops = 0;
        double max_alloc = 0, min_alloc = INT32_MAX, avg_alloc = 0, delta_alloc;
        for(j = 0; j < interval; j++) {
            int bw = 0, iops = 0;
            double alloc = 0;
            for(i = thr_r; i < thr_h + thr_g; i++) {
                bw += (thr_data[i].read_bw[j] + thr_data[i].write_bw[j]) * PAGE_SZ / CHL_BW / 1024 / 1024;
                iops += thr_data[i].iops[j];
                alloc += (double)thr_data[i].alloc[j] * PAGE_SZ / CHL_SZ / 1024.0 / 1024.0 / 1024.0;
            }
            max_bw = MAX(bw, max_bw);
            min_bw = MIN(bw, min_bw);
            avg_bw += bw;
            max_iops = MAX(iops, max_iops);
            min_iops = MIN(iops, min_iops);
            avg_iops += iops;
            total_alloc += alloc;
            max_alloc = MAX(total_alloc, max_alloc);
            min_alloc = MIN(total_alloc, min_alloc);
            avg_alloc += total_alloc;
        }
        avg_iops = avg_iops / interval;
        avg_bw = avg_bw / interval;
        avg_alloc = avg_alloc / interval;
        delta_alloc = total_alloc - prev_alloc;
        prev_alloc = total_alloc;
        char temp0[76], temp1[57];
        sprintf(temp0, "%010d %010d %010d %010d %010d %010d %010d", cnt+1, max_bw, min_bw, avg_bw, max_iops, min_iops, avg_iops);
        printf("%s\n", temp0);
        memcpy(bw_shared_mem, temp0, 76);

        sprintf(temp1, "%010d %011.5lf %011.5lf %011.5lf %011.5lf", cnt+1, max_alloc, min_alloc, avg_alloc, delta_alloc);
        printf("%s\n", temp1);
        memcpy(sz_shared_mem, temp1, 57);

        for(i = 0; i < thr_r + thr_h + thr_g; i++) {
            free(thr_data[i].read_bw);
            free(thr_data[i].write_bw);
            free(thr_data[i].chls);
            free(thr_data[i].iops);
            free(thr_data[i].alloc);
        }

        //Erase blocks before the next iteration
        erase_blks_v(h1);
        erase_blks_v(r1);
        erase_blks_v(p1);
        cnt++;
    }
    //for(i = 0; i < thr_r; i++) {
    //    pthread_cancel(wrr_threads[i]);
    //}
    fprintf(stdout, "PRINTING REGULAR\n");
    for(j = 0; j < iter * interval; j++) {
        fprintf(stdout, "%d %d\n", tot_reads_r[j], tot_writes_r[j]);
    }
    fprintf(stdout, "PRINTING HARVEST\n");
    for(j = 0; j < iter * interval; j++) {
        fprintf(stdout, "%d %d\n", tot_reads_h[j], tot_writes_h[j]);
    }
    fflush(stdout);
    // free_trace(t_r);
    // free_trace(t_h);
    free_regular_vssd(r1);
    free(buf);
    free(tot_reads_h);
    free(tot_writes_h);
    free(tot_reads_r);
    free(tot_writes_r);
    free(mapping_r);
    free(mapping_h);
    free(metabuf);
    free(readbuf);
    free(readmetabuf);

    path = concat(BLOCKFLEX_DIR, "end_harvest");
    int fd = open(path, O_WRONLY | O_APPEND | O_CREAT, 0644);
    free(path);
    close(fd);
    close(sz_fd);
    close(bw_fd);
    //fprintf(stdout, "Everything done and freed\n");
    //fflush(stdout);
    return 0;
} 
