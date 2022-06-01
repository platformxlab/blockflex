#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "powerfault.h"
#include <sys/stat.h>
#include "../ocssd_queue.h"

/*config environment variables*/
const char *pfe_io_block_size_environ="PFE_IO_BLOCK_SIZE";
/*default values */
int pfe_io_block_size = 4096;



/*alloc a rand mask w/ the same lenght as cmd data buf*/
char* pfe_alloc_rand_mask(uint32_t length){
    char *buf = NULL;
    buf = (char *)calloc(length, 1);
    char *offset = buf;
    int rand_num = 0;
    int i = 0;
    for(i = 0; i < length/sizeof(int); ++i){
        rand_num = rand();	
        memcpy(offset, &rand_num, sizeof(int));
        offset += sizeof(int);
    }
    return buf; 
}

void pfe_free_rand_mask(char *buf){
    free(buf);
    return;
}

/*check if the LBA falls in the corruption range*/
int pfe_addr_in_range(uint64_t offset){
    if(offset >= PFE_CORRUPT_LOWERBOUND && 
       offset <= PFE_CORRUPT_UPPERBOUND) {
        return 1;
    } else {
        return 0;
    }
}

uint64_t pfe_keep_in_range(uint64_t id, uint64_t lower, uint64_t upper){
    if(id < lower) 
        return lower;
    else if(id > upper)
        return upper;
    else
        return id;
}

//is id in range [lower, upper)
int pfe_is_in_range(uint64_t id, uint64_t lower, uint64_t upper){
    if(id >= lower && id < upper) {
        return 1;
    } else {
        return 0;
    }
}



void pfe_print_scsi_cmd(struct scsi_cmd *cmd){
	printf("PFE:PFE: pfe_print_scsi_cmd(struct scsi_cmd *cmd)\n");
    //printf("PFE: sizeof(struct scsi_cmd) == %d\n", sizeof(struct scsi_cmd));
	//printf("PFE: *cmd == %p\n", cmd);
    //printf("PFE: cmd->c_target == %p (struct target *)\n", cmd->c_target);
    printf("PFE: cmd->c_target->name == %s \n", cmd->c_target->name);
    printf("PFE: cmd->c_target->tid == %d \n", cmd->c_target->tid);
    //printf("PFE: &(cmd->c_hlist) == %p (struct list_head)\n", &(cmd->c_hlist));
    //printf("PFE: &(cmd->qlist) == %p (struct list_head)\n", &(cmd->qlist));
    printf("PFE: cmd->dev_id == %"PRIu64" (uint64_t)\n", cmd->dev_id);//=1
    //printf("PFE: cmd->dev == %p (struct scsi_lu *)\n", cmd->dev);
    printf("PFE: cmd->state == %"PRIu64" (unsigned long)\n", cmd->state);
	printf("PFE: cmd->data_dir == %d (enum data_direction)\n", cmd->data_dir);
    //printf("PFE: cmd->in_sdb.resid == %d (int)\n", cmd->in_sdb.resid);
    printf("PFE: cmd->in_sdb.length == %"PRIu64" (uint32_t)\n", cmd->in_sdb.length);
    printf("PFE: cmd->in_sdb.buffer == %p (uint64_t)\n", cmd->in_sdb.buffer);
    printf("PFE: cmd->out_sdb.resid == %d (int)\n", cmd->out_sdb.resid);
    printf("PFE: cmd->out_sdb.length == %"PRIu64" (uint32_t)\n", cmd->out_sdb.length);
    printf("PFE: cmd->out_sdb.buffer == %p (uint64_t)\n", cmd->out_sdb.buffer);
	printf("PFE: cmd->cmd_itn_id == %"PRIu64" (uint64_t)\n", cmd->cmd_itn_id);
	printf("PFE: cmd->offset == %"PRIu64" (uint64_t)\n", cmd->offset);
	printf("PFE: cmd->tl == %u (uint32_t)\n", cmd->tl);
	//printf("PFE: cmd->scb == %p (uint8_t *)\n", cmd->scb);
    printf("PFE: cmd->scb[0] == %x\n", cmd->scb[0]);
    printf("PFE: cmd->scb_len == %d (int)\n", cmd->scb_len);
    //printf("PFE: &(cmd->lun[0]) == %p (uint8_t lun[8])\n", &(cmd->lun[0]));
    //printf("PFE: cmd->lun[0] == %x (uint8_t lun[8])\n", cmd->lun[0]);//=0
    //printf("PFE: cmd->lun[1] == %x (uint8_t lun[8])\n", cmd->lun[1]);//=1
    //printf("PFE: cmd->lun[2] == %x (uint8_t lun[8])\n", cmd->lun[2]);//=0
    //printf("PFE: cmd->attribut == %d (int)\n", cmd->attribute);//=32
	printf("PFE: cmd->tag == %"PRIu64" (uint64_t)\n", cmd->tag);
    printf("PFE: cmd->result == %d (int)\n", cmd->result);
	//printf("PFE: cmd->mreq == %p (struct mgmt_req *)\n", cmd->mreq);
	//printf("PFE: &(cmd->sense_buffer[0]) == %p (unsigned char)\n", &(cmd->sense_buffer[0]));
	//printf("PFE: cmd->sense_buffer[0] == %x (unsigned char)\n", cmd->sense_buffer[0]);
    //printf("PFE: cmd->sense_len == %d (int)\n", cmd->sense_len);//=0
    //printf("PFE: &(cmd->bs_list) == %p (struct list_head)\n", &(cmd->bs_list));
    //printf("PFE: cmd->it_nexus == %p (struct it_nexus *)\n", cmd->it_nexus);
    //printf("PFE: cmd->itn_lu_info == %p (struct it_nexus_lu_info *)\n", cmd->itn_lu_info);
    //
  fflush(stdout);

    
    return;
}

void pfe_print_scsi_cdb(uint8_t *scb){
    printf("PFE:PFE: pfe_print_scsi_cdb(uint8_t *scb)\n");
	switch (scb[0]) {
        case WRITE_10:
            printf("PFE: scb[0] == %x (Op Code)\n", scb[0]);
            //printf("PFE: scb[1] == %x \n", scb[1]);
            //printf("PFE: scb[2] == %x (LBA MSB)\n", scb[2]);//MSB
            //printf("PFE: scb[3] == %x (LBA)\n", scb[3]);
            //printf("PFE: scb[4] == %x (LBA)\n", scb[4]);
            //printf("PFE: scb[5] == %x (LBA LSB)\n", scb[5]);//LSB
            uint8_t four_bytes[4];
            four_bytes[0] = scb[5];//little endian
            four_bytes[1] = scb[4];
            four_bytes[2] = scb[3];
            four_bytes[3] = scb[2];
            uint32_t *four_bytes_p = four_bytes;
            //printf("PFE: scb[2-5] == %u (32-bit LBA, in 512-sectors)\n", *four_bytes_p);
            printf("PFE: scb[2-5] == %u (32-bit LBA, in bytes)\n", (*four_bytes_p)*512);
            //printf("PFE: scb[6] == %x \n", scb[6]);
            //printf("PFE: scb[7] == %x (Transfer length: MSB)\n", scb[7]);//MSB
            //printf("PFE: scb[8] == %x (Transfer length: LSB)\n", scb[8]);//LSB
            uint8_t two_bytes[2];
            two_bytes[0] = scb[8]; //little-endian
            two_bytes[1] = scb[7];
            uint16_t *two_bytes_p = two_bytes;
            //printf("PFE: scb[7-8] == %u (16-bit Transfer length, in 512-sectors\n", *two_bytes_p);
            printf("PFE: scb[7-8] == %u (16-bit Transfer length, in bytes)\n", (*two_bytes_p)*512);
            //printf("PFE: scb[9] == %x (Control)\n", scb[9]);
            break;
        default:
            printf("PFE: scb[0] == %x (Op Code)\n", scb[0]);
            printf("PFE: Not a WRITE_10 cmd!\n");
            break;
    }

    return;
}


//print header of a logged IO
void pfe_print_header(pfe_io_header_t *header){

    printf("PFE: IO Header\n");
    printf("PFE: header.id = %"PRIu64"\n", header->id);
    printf("PFE: header.ts = %"PRIu64"\n", header->ts);
    printf("PFE: header.offset = %"PRIu64"\n", header->offset);
    printf("PFE: header.length = %"PRIu64"\n", header->length);
    printf("PFE: header.datalog_offset = %"PRIu64"\n", header->datalog_offset);
    printf("PFE: header.cmd_type = %"PRIu64"\n", header->cmd_type);
    printf("PFE: header.data_dir = %"PRIu64"\n", header->data_dir);
    fflush(stdout);

    return;
}
char* concat(const char *s1, const char *s2)
{
    char *result = malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

//log this write cmd
//the most important function for recording
int pfe_log_io_req(struct scsi_cmd *cmd, uint64_t offset,\
        uint64_t length, char *buf){

    struct stat sbuf;   
    char* path = concat(USR_DIR, "iscsi/start_log");
    if(stat(path, &sbuf) != 0){
        free(path);
        return 0;
    }
    free(path);

    fprintf(stderr, "PFE:PFE: pfe_log_io_req(), pfe_io_id = %d\n", pfe_io_id);

    /*
     * typedef struct pfe_io_header {
     *      uint64_t id;
     *      uint64_t ts;
     *      uint64_t offset;
     *      uint64_t length;
     *      uint64_t datalog_offset;
     *      uint64_t cmd_type;
     *      uint64_t data_dir;
     *      //uint64_t marker; 
     *      uint64_t tid; //cmd->c_target->tid
     * } pfe_io_header_t;
     */

    struct timeval t;
    gettimeofday(&t, 0); 
    uint64_t ts = t.tv_sec * 1000000ull + t.tv_usec;

    //make sure io_id and datalog_offset are updated together
    pthread_mutex_lock(&pfe_io_logs_mutex);

    // struct stat sbuf;
    // fstat(pfe_io_header_fd, &sbuf);
    // off_t size = sbuf.st_size;
    // printf("%ld\n", size);


    pfe_io_header_t io_header;
    io_header.id = pfe_io_id;//read shared var
    io_header.ts = ts;
    io_header.offset = offset;
    io_header.length = length;
    io_header.datalog_offset = pfe_io_datalog_offset;//read shared var
    io_header.cmd_type = (uint64_t)cmd->scb[0];
    io_header.data_dir = (uint64_t)cmd->data_dir;
    //io_header.marker = 101010101010;
    io_header.tid = (uint64_t)cmd->c_target->tid;

    ++pfe_io_id;//next io id
    pfe_io_datalog_offset += length;//next io data start here
    pthread_mutex_unlock(&pfe_io_logs_mutex);

    int ret = 0;

    //pfe_print_header(&io_header);

    //log header of this io req
    ret = pwrite(pfe_io_header_fd, &io_header, sizeof(pfe_io_header_t), sizeof(pfe_io_header_t)*io_header.id);
    if(ret < 0){
        printf("PFE: log io header failed! ret = %d \n", ret);
        return ret;
    }
    //update total io cnt in the 0 header
    ret = pwrite(pfe_io_header_fd, &io_header.id, sizeof(uint64_t), 0);
    if(ret < 0){
        printf("PFE: update super_head failed! ret = %d \n", ret);
        return ret;
    }
    //fsync(pfe_io_header_fd);
    printf("PFE: logged IO# %"PRIu64" header to fd %d\n", io_header.id, pfe_io_header_fd);

    //log data of this write io req
    if(buf != NULL){
        ret = pwrite(pfe_io_data_fd, buf, length, io_header.datalog_offset);
        if(ret < 0){
            printf("PFE: log io data failed! ret = %d \n", ret);
            return ret;
        }
    }
    fsync(pfe_io_data_fd);
    printf("PFE: logged IO# %"PRIu64" data to fd %d\n", io_header.id, pfe_io_header_fd);

    return 0;

}

//get total cnt of recorded IO
uint64_t pfe_get_io_cnt(int io_header_fd){

    pfe_io_header_t io_header;
    int ret;

    //read total io cnt
    ret = pread(io_header_fd, &io_header, sizeof(pfe_io_header_t), 0);
    if(ret < 0){
        printf("PFE: ERROR in reading io_header log !!!\n");
        return -1; 
    }

    //first head's first uint64_t
    printf("PFE: total # of IO in log: %"PRIu64" \n", io_header.id);

    return io_header.id;
}

//split large IO to multiple 4KB writes
//create another new header_log
//data_log doesn't need change
uint64_t pfe_split_io_log(int io_header_fd, int io_header_split_fd){

    uint64_t ret = 0;
    uint64_t tot_io_cnt = 0;
    uint64_t tot_splitted_io_cnt = 0;
    uint64_t i = 0;

    pfe_io_header_t io_header_orig;
    pfe_io_header_t io_header_split;

    tot_io_cnt =  pfe_get_io_cnt(io_header_fd);

    //init first header in splitted io header log (64 B) (to store total io cnt later)
    char * super_head = calloc(sizeof(pfe_io_header_t), 1);
    ret = pwrite(io_header_split_fd, super_head, sizeof(pfe_io_header_t), 0);
    if(ret < 0){
        printf("PFE: init sup_head failed! ret = %d \n", ret);
        return ret;
    }

    /*
     * typedef struct pfe_io_header {
     *      uint64_t id;
     *      uint64_t ts;
     *      uint64_t offset;
     *      uint64_t length;
     *      uint64_t datalog_offset;
     *      uint64_t cmd_type;
     *      uint64_t data_dir;
     *     // uint64_t marker; 
     *      uint64_t tid; 
     * } pfe_io_header_t;
     */

    //read each header
    for(i = 1; i <= tot_io_cnt; ++i){//0 is super head

        //this IO's offset in the header log
        uint64_t cur_offset = sizeof(pfe_io_header_t) * (i);//skip 0 super_header
        //read this IO's header
        ret = pread(io_header_fd, &io_header_orig, sizeof(pfe_io_header_t), cur_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread #%"PRIu64" header from io_header log !!!\n", i);
            return -1; 
        }
        //pfe_print_header(&io_header_orig);

        //unchanged fields:
        io_header_split.ts = io_header_orig.ts;
        io_header_split.cmd_type = io_header_orig.cmd_type;
        io_header_split.data_dir = io_header_orig.data_dir;
        //io_header_split.marker = io_header_orig.marker;
        io_header_split.tid = io_header_orig.tid;

        //split to 4KB if > 4KB 
        if(io_header_orig.length > pfe_io_block_size){

            uint64_t n_split = io_header_orig.length / pfe_io_block_size; //# of 4KB blocks
            uint64_t n_split_remain = io_header_orig.length % pfe_io_block_size;

            int j;
            for(j = 0; j < n_split; ++j) {
                ++tot_splitted_io_cnt;
                io_header_split.id = tot_splitted_io_cnt;
                io_header_split.offset = io_header_orig.offset + pfe_io_block_size * j;
                io_header_split.length = pfe_io_block_size;
                io_header_split.datalog_offset = io_header_orig.datalog_offset + pfe_io_block_size * j;

                //log header of this io 
                ret = pwrite(io_header_split_fd, &io_header_split, sizeof(pfe_io_header_t), sizeof(pfe_io_header_t)*io_header_split.id);
                if(ret < 0){
                    printf("PFE: log io header failed! ret = %d \n", ret);
                    return ret;
                }
            }
            if(n_split_remain != 0){
                ++tot_splitted_io_cnt;
                io_header_split.id = tot_splitted_io_cnt;
                io_header_split.offset = io_header_orig.offset + pfe_io_block_size * n_split;
                io_header_split.length = n_split_remain;//remain bytes
                io_header_split.datalog_offset = io_header_orig.datalog_offset + pfe_io_block_size * n_split;
                
                //log header of this io 
                ret = pwrite(io_header_split_fd, &io_header_split, sizeof(pfe_io_header_t), sizeof(pfe_io_header_t)*io_header_split.id);
                if(ret < 0){
                    printf("PFE: log io header failed! ret = %d \n", ret);
                    return ret;
                }
            }
        
        }
        else{//< 4KB, don't split
            ++tot_splitted_io_cnt;
            io_header_split.id = tot_splitted_io_cnt;
            io_header_split.offset = io_header_orig.offset;
            io_header_split.length = io_header_orig.length;
            io_header_split.datalog_offset = io_header_orig.datalog_offset;
        
            //log header of this io 
            ret = pwrite(io_header_split_fd, &io_header_split, sizeof(pfe_io_header_t), sizeof(pfe_io_header_t)*io_header_split.id);
            if(ret < 0){
                printf("PFE: log io header failed! ret = %d \n", ret);
                return ret;
            }
        }
    }

    //record total splitted io cnt in the 0 header
    ret = pwrite(io_header_split_fd, &tot_splitted_io_cnt, sizeof(uint64_t), 0);
    if(ret < 0){
        printf("PFE: update super_head failed! ret = %d \n", ret);
        return ret;
    }

    return ret;
}

//simple replay all recorded IO without failure emulation
uint64_t pfe_replay_simple(int io_header_fd, int io_data_fd, int disk_fd,\
        uint64_t first_id, uint64_t last_id){

    uint64_t ret = 0;
    uint64_t tot_io_cnt = 0;
    uint64_t replayed_cnt = 0;
    uint64_t i = 0;
    pfe_io_header_t io_header;

    tot_io_cnt = pfe_get_io_cnt(io_header_fd);

    /*
     * typedef struct pfe_io_header {
     *      uint64_t id;
     *      uint64_t ts;
     *      uint64_t offset;
     *      uint64_t length;
     *      uint64_t datalog_offset;
     *      uint64_t cmd_type;
     *      uint64_t data_dir;
     *      uint64_t tid; 
     * } pfe_io_header_t;
     */

    //read each header
    for(i = first_id; i <= last_id; ++i){
        ++replayed_cnt;

        //this IO's offset in the header log
        uint64_t cur_offset = sizeof(pfe_io_header_t) * (i);//skip 0 super_header
        //read this IO's header
        ret = pread(io_header_fd, &io_header, sizeof(pfe_io_header_t), cur_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread #%"PRIu64" header from io_header log !!!\n", i);
            return -1; 
        }
        //pfe_print_header(&io_header);
        
        //buf for this IO's data
        char *data_buf = calloc(io_header.length, 1); 
        if(data_buf == NULL){
            printf("PFE: ERROR at calloc !!!\n");
            return -1;
        }
        //read data of this IO from datalog
        ret = pread(io_data_fd, data_buf, io_header.length, io_header.datalog_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread data #%"PRIu64", \
                    offset #%"PRIu64" at io_data log!!!\n", i, io_header.datalog_offset);
            return -1; 
        }

        //replay the IO to disk 
        ret = pwrite(disk_fd, data_buf, io_header.length, io_header.offset);
        if(ret < 0){
            printf("PFE: ERROR at replay data #%"PRIu64", \
                    offset #%"PRIu64" on disk file!!!\n", i, io_header.offset);
            return -1; 
        }
        free(data_buf);
    }

    printf("PFE: total replayed IO # = %"PRIu64"\n", replayed_cnt);

    return ret;
}

uint64_t pfe_do_bitcorrupt(int disk_fd, char *data_buf, uint64_t length, uint64_t offset){

    printf("PFE: do_bitcorrupt at block# %"PRIu64" (offset %"PRIu64", length %"PRIu64") !\n", offset/pfe_io_block_size, offset, length);
    uint64_t ret = 0;
    char *pfe_buf = NULL;
    pfe_buf = pfe_alloc_rand_mask(length);

    ret = pwrite64(disk_fd, pfe_buf, length, offset);//write rand bits

    pfe_free_rand_mask(pfe_buf);

    return ret;
}

uint64_t pfe_do_shornwrite(int disk_fd, char *data_buf, uint64_t length, uint64_t offset){

    printf("PFE: do_shornwrite at block# %"PRIu64" (offset %"PRIu64", length %"PRIu64") !\n", offset/pfe_io_block_size, offset, length);
    uint64_t ret = 0;
    uint32_t new_data_size = PFE_SHORN_NEW_DATA_SIZE;//TODO: rand select

    if(length < new_data_size)
        new_data_size = length; //"shorn write" whole record
    
    char *pfe_buf = NULL;
    pfe_buf = calloc(new_data_size, 1);
    memcpy(pfe_buf, data_buf , new_data_size);            

    ret = pwrite64(disk_fd, pfe_buf, new_data_size, offset);
    ret = length; //pretend return from a successful pwrite

    free(pfe_buf);

    return ret;
}

uint64_t pfe_do_flywrite(int disk_fd, char *data_buf, uint64_t length, uint64_t offset){

    printf("PFE: do_flywrite at block# %"PRIu64" (offset %"PRIu64", length %"PRIu64") !\n", offset/pfe_io_block_size, offset, length);
    uint64_t ret = 0;
    uint64_t pfe_new_offset = offset + PFE_FLY_OFFSET ; //TODO: get offset from input
    ret = pwrite64(disk_fd, data_buf, length, pfe_new_offset);

    return ret;
}    

uint64_t pfe_do_unserialwrite(int disk_fd, char *data_buf, uint64_t length, uint64_t offset){

    printf("PFE: do_unserialwrite at block# %"PRIu64" (offset %"PRIu64", length %"PRIu64") !\n", offset/pfe_io_block_size, offset, length);
    //simply skip this write
    //pretend return from a successful pwrite
    return length;
}

//record error blk# 
//will hang or errno 5 when reading these blocks
//for emulating metadata corrupt. or return err bit corrupt 
uint64_t pfe_do_errblk(int ret_err_fd, uint64_t length, uint64_t offset, uint64_t *cnt){

    printf("PFE: do_recordblk at block# %"PRIu64" (offset %"PRIu64") !\n", offset/pfe_io_block_size, offset);
    uint64_t ret = 0;
    uint64_t err_fd_offset = 0;
    uint64_t blk_num = offset/pfe_io_block_size;

    ++(*cnt);
    err_fd_offset = (*cnt)*sizeof(uint64_t);

    //just log the blk#
    ret = pwrite(ret_err_fd, &blk_num, sizeof(uint64_t), err_fd_offset);
    if(ret < 0){
        printf("PFE: do_recordblk failed! ret = %d, errno = %d\n", ret, errno);
        return ret;
    }
    printf("PFE: write at io_err_blk_log: err_fd_offset = %"PRIu64", blk_num = %"PRIu64"\n", err_fd_offset, blk_num);

    //first int stores total blk cnt
    ret = pwrite(ret_err_fd, cnt, sizeof(uint64_t), 0);
    if(ret < 0){
        printf("PFE: do_recordblk failed! ret = %d, errno = %d\n", ret, errno);
        return ret;
    }
    printf("PFE: write at io_err_blk_log: err_fd_offset = 0, cnt = %"PRIu64"\n", *cnt);

    //don't need to write the data to disk
    //just pretend return from a successful pwrite
    return length;
}

uint64_t pfe_do_normalwrite(int disk_fd, char *data_buf, uint64_t length, uint64_t offset){

    uint64_t ret = 0;
    ret = pwrite64(disk_fd, data_buf, length, offset);

    return ret;
}    

//replay with shorn writes
uint64_t pfe_replay_shorn(int io_header_fd, int io_data_fd, int disk_fd,\
        uint64_t first_id, uint64_t last_id){
    
    printf("PFE:Replay with Shorn Write: first_io = %"PRIu64", last_io = %"PRIu64"\n", first_id, last_id);

    pfe_io_header_t io_header;
    uint64_t replayed_cnt = 0;
    uint64_t ret = 0;
    uint64_t i = 0;
    
    //based on observed pattern
    //only the last op will be affected
    uint64_t shorn_id = last_id;

    //read from start_id'th IO 
    for(i = first_id; i <= last_id; ++i){
        ++replayed_cnt;

        //this IO's offset in the header log
        uint64_t cur_offset = sizeof(pfe_io_header_t) * (i);
        //read this IO's header
        ret = pread(io_header_fd, &io_header, sizeof(pfe_io_header_t), cur_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread header #%"PRIu64" from io_header log !!!\n", i);
            return -1; 
        }
        //pfe_print_header(&io_header);
        
        //buf for this IO's data
        char *data_buf = calloc(io_header.length, 1); 
        if(data_buf == NULL){
            printf("PFE: ERROR at calloc !!!\n");
            return -1;
        }
        //read data of this IO from datalog to data_buf
        ret = pread(io_data_fd, data_buf, io_header.length, io_header.datalog_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread data #%"PRIu64", \
                    offset #%"PRIu64" at io_data log!!!\n", i, io_header.datalog_offset);
            return -1; 
        }

        uint64_t offset = io_header.offset;
        uint64_t length = io_header.length;

        //length, offset, data all availbe by now, start emu
        if(i == shorn_id) {//shorn this one 
            ret = pfe_do_shornwrite(disk_fd, data_buf, length, offset);
            printf("PFE: Emulated shorn write at IO# %"PRIu64"!\n", i);
        }
        else{//don't shorn 
            ret = pfe_do_normalwrite(disk_fd, data_buf, length, offset);
        }

        if(ret != length){
            printf("PFE: ERROR at replay data #%"PRIu64", \
                    offset #%"PRIu64" on disk file!!!\n", i, io_header.offset);
            return -1; 
        }

        free(data_buf);
    }

    printf("PFE: total replayed IO # = %"PRIu64"\n", replayed_cnt);

    return 0;
}


//replay with simple bit corruption (just corrupt data, don't consider return errno) 
uint64_t pfe_replay_bit(int io_header_fd, int io_data_fd, int disk_fd,\
        uint64_t first_id, uint64_t last_id){
    
    printf("PFE:Replay with Bit Corruption: first_io = %"PRIu64", last_io = %"PRIu64"\n", first_id, last_id);

    pfe_io_header_t io_header;
    uint64_t replayed_cnt = 0;
    uint64_t ret = 0;
    uint64_t i = 0;
    
    uint64_t first_corrupt_id = last_id - PFE_FIRST_CORRUPT_BEFORE_CUT;//first op to corrupt
    uint64_t last_corrupt_id = first_corrupt_id + PFE_CORRUPT_RANGE;//# of consecutive op to corrupt
    //make sure the id is in the replayable range//not necessary
    //first_corrupt_id = pfe_keep_in_range(first_corrupt_id, first_id, last_id);
    //last_corrupt_id = pfe_keep_in_range(last_corrupt_id, first_id, last_id);
    

    //read from start_id'th IO 
    for(i = first_id; i <= last_id; ++i){
        ++replayed_cnt;

        //this IO's offset in the header log
        uint64_t cur_offset = sizeof(pfe_io_header_t) * (i);
        //read this IO's header
        ret = pread(io_header_fd, &io_header, sizeof(pfe_io_header_t), cur_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread header #%"PRIu64" from io_header log !!!\n", i);
            return -1; 
        }
        //pfe_print_header(&io_header);
        
        //buf for this IO's data
        char *data_buf = calloc(io_header.length, 1); 
        if(data_buf == NULL){
            printf("PFE: ERROR at calloc !!!\n");
            return -1;
        }
        //read data of this IO from datalog to data_buf
        ret = pread(io_data_fd, data_buf, io_header.length, io_header.datalog_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread data #%"PRIu64", \
                    offset #%"PRIu64" at io_data log!!!\n", i, io_header.datalog_offset);
            return -1; 
        }

        uint64_t offset = io_header.offset;
        uint64_t length = io_header.length;

        //length, offset, data all availbe by now, start emu
        if(pfe_is_in_range(i, first_corrupt_id, last_corrupt_id)) {//id in corrupt range 
            ret = pfe_do_bitcorrupt(disk_fd, data_buf, length, offset);
            printf("PFE: Emulated bit corruption at IO# %"PRIu64"!\n", i);
        }
        else{//don't corrupt 
            ret = pfe_do_normalwrite(disk_fd, data_buf, length, offset);
        }

        if(ret != length){
            printf("PFE: ERROR at replay data #%"PRIu64", \
                    offset #%"PRIu64" on disk file!!!\n", i, io_header.offset);
            return -1; 
        }

        free(data_buf);
    }

    printf("PFE: total replayed IO # = %"PRIu64"\n", replayed_cnt);

    return 0;
}


//replay with flying writes
uint64_t pfe_replay_fly(int io_header_fd, int io_data_fd, int disk_fd,\
        uint64_t first_id, uint64_t last_id){
    
    printf("PFE:Replay with Flying Writes: first_io = %"PRIu64", last_io = %"PRIu64"\n", first_id, last_id);

    pfe_io_header_t io_header;
    uint64_t replayed_cnt = 0;
    uint64_t ret = 0;
    uint64_t i = 0;
   
    //single flying op
    uint64_t fly_id = last_id - PFE_FIRST_FLY_BEFORE_CUT;//first op to fly 

    //read from start_id'th IO 
    for(i = first_id; i <= last_id; ++i){
        ++replayed_cnt;

        //this IO's offset in the header log
        uint64_t cur_offset = sizeof(pfe_io_header_t) * (i);
        //read this IO's header
        ret = pread(io_header_fd, &io_header, sizeof(pfe_io_header_t), cur_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread header #%"PRIu64" from io_header log !!!\n", i);
            return -1; 
        }
        //pfe_print_header(&io_header);
        
        //buf for this IO's data
        char *data_buf = calloc(io_header.length, 1); 
        if(data_buf == NULL){
            printf("PFE: ERROR at calloc !!!\n");
            return -1;
        }
        //read data of this IO from datalog to data_buf
        ret = pread(io_data_fd, data_buf, io_header.length, io_header.datalog_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread data #%"PRIu64", \
                    offset #%"PRIu64" at io_data log!!!\n", i, io_header.datalog_offset);
            return -1; 
        }

        uint64_t offset = io_header.offset;
        uint64_t length = io_header.length;

        //length, offset, data all availbe by now, start emu
        if(i == fly_id) {//fly this one 
            ret = pfe_do_flywrite(disk_fd, data_buf, length, offset);
            printf("PFE: Emulated flying write at IO# %"PRIu64"!\n", i);
        }
        else{//don't fly
            ret = pfe_do_normalwrite(disk_fd, data_buf, length, offset);
        }

        if(ret != length){
            printf("PFE: ERROR at replay data #%"PRIu64", \
                    offset #%"PRIu64" on disk file!!!\n", i, io_header.offset);
            return -1; 
        }

        free(data_buf);
    }

    printf("PFE: total replayed IO # = %"PRIu64"\n", replayed_cnt);

    return 0;
}


//replay with unserializable writes 
uint64_t pfe_replay_unserial(int io_header_fd, int io_data_fd, int disk_fd,\
        uint64_t first_id, uint64_t last_id, char * pattern_buf, uint32_t pattern_size){
    
    printf("PFE:Replay with Unserializable Writes: first_io = %"PRIu64", last_io = %"PRIu64"\n", first_id, last_id);

    pfe_io_header_t io_header;
    uint64_t replayed_cnt = 0;
    uint64_t ret = 0;
    uint64_t i = 0;
	uint64_t remained_io = 0;
	uint32_t pattern_i = 0;
	char pattern_flag;
	uint32_t dropped_cnt = 0;
   
    //currently only support one vulnerability window to drop writes
    //uint64_t first_unserial_id = last_id - PFE_FIRST_UNSERIAL_BEFORE_CUT;//first op to drop 
    //uint64_t last_unserial_id = first_unserial_id + PFE_UNSERIAL_RANGE;

    //read from start_id'th IO 
    for(i = first_id; i <= last_id; ++i){
        ++replayed_cnt;

        //this IO's offset in the header log
        uint64_t cur_offset = sizeof(pfe_io_header_t) * (i);
        //read this IO's header
        ret = pread(io_header_fd, &io_header, sizeof(pfe_io_header_t), cur_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread header #%"PRIu64" from io_header log !!!\n", i);
            return -1; 
        }
        //pfe_print_header(&io_header);
        
        //buf for this IO's data
        char *data_buf = calloc(io_header.length, 1); 
        if(data_buf == NULL){
            printf("PFE: ERROR at calloc !!!\n");
            return -1;
        }
        //read data of this IO from datalog to data_buf
        ret = pread(io_data_fd, data_buf, io_header.length, io_header.datalog_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread data #%"PRIu64", \
                    offset #%"PRIu64" at io_data log!!!\n", i, io_header.datalog_offset);
            return -1; 
        }

        uint64_t offset = io_header.offset;
        uint64_t length = io_header.length;

        //length, offset, data all available now, start emu
		
		//pattern for this op
		remained_io = last_id - i;
		if(remained_io < pattern_size) { //this op has enter the pattern
			uint32_t pattern_idx = pattern_size - remained_io - 1;//this op's idx in pattern
			pattern_flag = pattern_buf[pattern_idx];
		}
		//if(pattern_i < pattern_size) {
		//	pattern_flag = pattern_buf[pattern_i];
		//	pattern_i++;
		//}
		else{
			pattern_flag = '1';
		}

		if(pattern_flag == '0'){
        //if(pfe_is_in_range(i, first_unserial_id, last_unserial_id)) {//id in drop range 
            ret = pfe_do_unserialwrite(disk_fd, data_buf, length, offset);
            printf("PFE: Emulated unserializable writes at IO# %"PRIu64"!\n", i);
			dropped_cnt ++;
        }
        else{//don't drop 
            ret = pfe_do_normalwrite(disk_fd, data_buf, length, offset);
        }

        if(ret != length){
            printf("PFE: ERROR at replay data #%"PRIu64", \
                    offset #%"PRIu64" on disk file!!!\n", i, io_header.offset);
            return -1; 
        }

        free(data_buf);
    }

    printf("PFE: total replayed IO # = %"PRIu64"\n", replayed_cnt);
    printf("PFE: total dropped IO # = %"PRIu64"\n", dropped_cnt);

    return 0;
}


//replay with recording err block#s (metadata corruption / advanced bit corruption) 
uint64_t pfe_replay_errblk(int io_header_fd, int io_data_fd, int disk_fd,\
        uint64_t first_id, uint64_t last_id, int err_blk_fd){
    
    printf("PFE:Replay with Recording Err Blocks: first_io = %"PRIu64", last_io = %"PRIu64"\n", first_id, last_id);

    pfe_io_header_t io_header;
    uint64_t replayed_cnt = 0;
    uint64_t ret = 0;
    uint64_t i = 0;
    uint64_t err_cnt = 0;
   
    //currently only support one vulnerability window
    uint64_t first_err_id = last_id - PFE_FIRST_ERR_BEFORE_CUT;//first op to drop 
    uint64_t last_err_id = first_err_id + PFE_ERR_RANGE;

    //read from start_id'th IO 
    for(i = first_id; i <= last_id; ++i){
        ++replayed_cnt;

        //this IO's offset in the header log
        uint64_t cur_offset = sizeof(pfe_io_header_t) * (i);
        //read this IO's header
        ret = pread(io_header_fd, &io_header, sizeof(pfe_io_header_t), cur_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread header #%"PRIu64" from io_header log !!!\n", i);
            return -1; 
        }
        //pfe_print_header(&io_header);
        
        //buf for this IO's data
        char *data_buf = calloc(io_header.length, 1); 
        if(data_buf == NULL){
            printf("PFE: ERROR at calloc !!!\n");
            return -1;
        }
        //read data of this IO from datalog to data_buf
        ret = pread(io_data_fd, data_buf, io_header.length, io_header.datalog_offset);
        if(ret < 0){
            printf("PFE: ERROR at pread data #%"PRIu64", \
                    offset #%"PRIu64" at io_data log!!!\n", i, io_header.datalog_offset);
            return -1; 
        }

        uint64_t offset = io_header.offset;
        uint64_t length = io_header.length;

        //length, offset, data all availbe by now, start emu
        if(pfe_is_in_range(i, first_err_id, last_err_id)) {//id in drop range 
            ret = pfe_do_errblk(err_blk_fd, length, offset, &err_cnt);
            printf("PFE: Emulated recorind err block at IO# %"PRIu64" (offset = %"PRIu64", blk# = %"PRIu64"!\n", i, offset, offset/pfe_io_block_size);
            printf("PFE: err_blk_cnt = %"PRIu64"\n", err_cnt);
        }
        else{//
            ret = pfe_do_normalwrite(disk_fd, data_buf, length, offset);
        }

        if(ret != length){
            printf("PFE: ERROR at replay data #%"PRIu64", \
                    offset #%"PRIu64" on disk file!!!\n", i, io_header.offset);
            return -1; 
        }

        free(data_buf);
    }

    printf("PFE: total replayed IO # = %"PRIu64"\n", replayed_cnt);
    printf("PFE: total err_blk_cnt = %"PRIu64"\n", err_cnt);

    return 0;
}

//replay with emulated failure
//start from the start_id'th IO (should be 0 normally)
//end at end_id'th
//simple address-based failure pattern
uint64_t pfe_replay_failure(int io_header_fd, int io_data_fd, int disk_fd, int fail_type, uint64_t start_id, uint64_t end_id, int err_blk_fd){

    uint64_t ret = 0;
    uint64_t tot_io_cnt = 0;
    //uint64_t replayed_cnt = 0;
    uint64_t first_id = start_id; //0 super_header
    uint64_t last_id = end_id;
    uint64_t i;
    pfe_io_header_t io_header;

    tot_io_cnt = pfe_get_io_cnt(io_header_fd);

    if(tot_io_cnt < end_id){//don't replay beyond recorded IO cnt
        last_id = tot_io_cnt;
    }

    if(first_id < 1){//first replayable IO id is 1
        first_id = 1;
    }

    printf("PFE:PFE: REPLAY_w_FAILURE\n");
    printf("PFE: total # of IO recorded = %"PRIu64"\n", tot_io_cnt);
    printf("PFE: start replay from IO#= %"PRIu64"\n", start_id);
    printf("PFE: end replay at IO#= %"PRIu64"\n", last_id);
    printf("PFE: fail type = %d\n", fail_type);
    /*
     * typedef struct pfe_io_header {
     *      uint64_t id;
     *      uint64_t ts;
     *      uint64_t offset;
     *      uint64_t length;
     *      uint64_t datalog_offset;
     *      uint64_t cmd_type;
     *      uint64_t data_dir;
     *  //    uint64_t marker; 
     *      uint64_t tid; 
     * } pfe_io_header_t;
     */

    switch(fail_type){
        case PFE_SHORN_WRITES:{
             ret = pfe_replay_shorn(io_header_fd, io_data_fd, disk_fd,\
                                    first_id, last_id);
             if(ret == -1){
                printf("PTE: pfe_replay_shorn failed!");
                return -1;
             }
            break;
        }
        case PFE_BIT_CORRUPTION:{
             ret = pfe_replay_bit(io_header_fd, io_data_fd, disk_fd,\
                                    first_id, last_id);
             if(ret == -1){
                printf("PTE: pfe_replay_bit failed!");
                return -1;
             }
            break;
        }
        case PFE_FLYING_WRITES:{
             ret = pfe_replay_fly(io_header_fd, io_data_fd, disk_fd,\
                                    first_id, last_id);
             if(ret == -1){
                printf("PTE: pfe_replay_fly failed!");
                return -1;
             }
            break;
        }
        case PFE_UNSERIALIZABLE_WRITES:{
			 char * pattern_buf;
			 uint32_t pattern_size;
		 	 pattern_buf = pfe_build_unser_pattern_cache(&pattern_size);
			 printf("PFE: pattern_size = %u\n", pattern_size);
			 /*int i;
			 for (i = 0; i < pattern_size; i++){
				printf("PFE: pattern_buf[%d]: %c\n", i, pattern_buf[i]); 
			 }*/
             ret = pfe_replay_unserial(io_header_fd, io_data_fd, disk_fd,\
                                    first_id, last_id, pattern_buf, pattern_size);
             if(ret == -1){
                printf("PTE: pfe_replay_unserial failed!");
                return -1;
             }
		 	 pfe_free_unser_pattern_cache(pattern_buf);
            break;
        }

        case PFE_RET_ERRNO5:
             //fall through
        case PFE_RET_HANG:
             //fall through
        case PFE_METADATA_CORRUPTION:{
             //record bad blk # during replay
             ret = pfe_replay_errblk(io_header_fd, io_data_fd, disk_fd,\
                                    first_id, last_id, err_blk_fd);
             if(ret == -1){
                printf("PTE: pfe_replay_errblk failed!");
                return -1;
             }
            break;
        }

        case PFE_DEAD_DEVICE:
            //nothing to do just fall through
        case PFE_NORMAL:
            //fall through
        default:
            printf("PFE: do simple replay w/o failure.\n"); 
            pfe_replay_simple(io_header_fd, io_data_fd, disk_fd, first_id, last_id);
    }

    return 0;
}


//read err blk#s to memory
void pfe_build_errblks_cache(int err_blk_fd){

    //pfe_errblks_cache_t * err_blk_cache;
    uint64_t tot_blk_cnt = 0;

    int ret = pread(err_blk_fd, &tot_blk_cnt, sizeof(uint64_t), 0);
    if(ret != sizeof(uint64_t)){
        printf("PFE: ERROR in reading err_blk !!!\n");
    }
    printf("PFE: tot_blk_cnt = %"PRIu64"\n", tot_blk_cnt);

    uint64_t length = sizeof(uint64_t)*(tot_blk_cnt);//+1: the tot_cnt
    pfe_errblks_tgtd.buf = calloc(length,1); 
    pfe_errblks_tgtd.length = length;
    pfe_errblks_tgtd.n_blks = tot_blk_cnt;

    //uint64_t offset = sizeof(uint64_t); //0 is the tot_cnt, so start from 1 uint64_t 
    //int i = 0;
    /*for(i = 1; i <= tot_blk_cnt; ++i){//0 is the tot_cnt, so start from 1 
        offset = sizeof(uint64_t) * i;
        pread(err_blk_fd, &cache_buf, sizeof(uint64_t), offset);
         
    }*/

    ret = pread(err_blk_fd, pfe_errblks_tgtd.buf, length, sizeof(uint64_t));
    if(ret != length){
        printf("PFE: ERROR in reading err_blk !!!\n");
    }

    return;
}


void pfe_free_errblks_cache(){
    free(pfe_errblks_tgtd.buf);
    return;
}

//check if the req cover any err blk
int pfe_is_errblks(uint64_t offset, uint64_t length){

    uint64_t n_err_blks = pfe_errblks_tgtd.n_blks;
    uint64_t first_blk = offset/pfe_io_block_size;
    uint64_t after_last_blk = (offset+length)/pfe_io_block_size;
    uint64_t err_blk_num = 0;
    uint64_t buf_offset = 0;

    int i = 0;
    for(i = 0; i < n_err_blks; ++i){
        buf_offset = sizeof(uint64_t) * i;
        //pread(pfe_err_blk_fd, &err_blk_num, sizeof(uint64_t), buf_offset);
        memcpy(&err_blk_num, pfe_errblks_tgtd.buf + buf_offset, sizeof(uint64_t));
        //printf("PFE: err_blk_num = %"PRIu64"\n", err_blk_num);
        if(first_blk <= err_blk_num && err_blk_num < after_last_blk) {
            printf("PFE: read err_blk %"PRIu64": req offset %"PRIu64" \\
                    (blk %"PRIu64"), length %"PRIu64" (%"PRIu64" blks)\n",
                    err_blk_num, offset, first_blk, length, length/pfe_io_block_size);
            return 1;
        }
    }

    return 0;
}

//return -1 (errno 5) or hang at this read cmd 
            //ret = pread64(fd, scsi_get_in_buffer(cmd), length, offset);
int pfe_read(int fd, char * buf, uint64_t length, uint64_t offset,\
        struct scsi_cmd *cmd){
    int ret = 0;

    if(pfe_is_errblks(offset, length)){
       ret = -1; 
       //ret = pread64(fd, buf, length, offset);
    }
    else{//do normal read
       ret = pread64(fd, buf, length, offset);
    }

    return ret;
}


char * pfe_build_unser_pattern_cache(uint32_t * pattern_size){
	char * pattern_buf;
	char * pattern_file = getenv("UNSERIAL_PATTERN_FILE"); 
	printf(pattern_file);
	printf("\n");
	FILE * fp = fopen(pattern_file, "r");
	char line[3];// '0' ' \n'  or  '1' '\n', fgets read N-1 = 2 bytes 
	uint32_t line_cnt = 0;
	uint32_t op_num = 0;
	//get ops count in pattern
	if(fp != NULL){
		while(fgets(line, sizeof(line), fp) != NULL){
			//fputs(line, stdout);	
			//printf("pattern: %c\n", line[0]);
			line_cnt ++;
		}
		fclose(fp);
		printf("PFE: ops in pattern = %u\n", line_cnt);
	}
	else{
		perror(pattern_file);	
		exit(EXIT_FAILURE);
	}

	//alloc buf for pattern
	*pattern_size = line_cnt;
	pattern_buf = calloc(line_cnt, 1);
	//printf("pattern_size: %d\n ", line_cnt);
	fp = fopen(pattern_file, "r");
	uint32_t i = 0;
	if(fp != NULL){
		while(fgets(line, sizeof(line), fp) != NULL){
			pattern_buf[i] = line[0];
			//printf("pattern_buf[%d]]]: %c\n", i, pattern_buf[i]);
			i++;
		}
		fclose(fp);
	}
	else{
		perror(pattern_file);	
		exit(EXIT_FAILURE);
	}
	
	return pattern_buf;
}

void pfe_free_unser_pattern_cache(char * pattern_buf){
	free(pattern_buf);
}
