/* mai: power fault emulator project*/
#ifndef __POWERFAULT_H
#define __POWERFAULT_H

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>

#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/time.h>
#include <stdbool.h>


#include "list.h"
#include "util.h"
#include "tgtd.h"
#include "scsi.h"
#include "spc.h"
#include "bs_thread.h"
#include "target.h"

//enable record for replay 
//1 enabled; 0 disabled
//enable record will disable failure emulation in tgt
#define ENABLE_RECORD 1

//types of failures
#define PFE_NORMAL 0
#define PFE_BIT_CORRUPTION 1 
#define PFE_SHORN_WRITES 2 
#define PFE_FLYING_WRITES 3 
#define PFE_UNSERIALIZABLE_WRITES 4 
#define PFE_METADATA_CORRUPTION 5 
#define PFE_DEAD_DEVICE 6
#define PFE_RET_ERRNO5 7
#define PFE_RET_HANG 8

//failure type to emulate in tgt
//only meaningful when recording is disabled
//#define PFE_FAILURE_TYPE 0


///////////////////////// paramters for simple emulation of failures
//addr range for address-based courrption 
#define PFE_CORRUPT_LOWERBOUND 1000 //min LBA to corrupt; skip block#0
#define PFE_CORRUPT_UPPERBOUND (4096*3-1) //max LBA to corrupt; corrupt block#1, bock#2
//metatdata corruption test need a larger range
//since read in larger chunk 
//#define PFE_CORRUPT_LOWERBOUND (4096*4) //min LBA to corrup
//#define PFE_CORRUPT_UPPERBOUND (4096*20-1) //max LBA to corrupt


//param for shorn writes
#define PFE_SHORN_NEW_DATA_SIZE (512*7) //size of new data (must < rec size)

//param for bit corruption
#define PFE_FIRST_CORRUPT_BEFORE_CUT 0 //count from the last(cut) op; e.g., 10 means the 10th to the last op is corupted
#define PFE_CORRUPT_RANGE 1 //consecutive ops to corrupt, 10, 11, 12

//param for flying writes
#define PFE_FIRST_FLY_BEFORE_CUT  10 //count from the last op 
#define PFE_FLY_OFFSET (4096*2) //add fixed bytes to offset; 

//param for unserializable writes
#define PFE_FIRST_UNSERIAL_BEFORE_CUT 9 //count from the last(cut) op; e.g., the 10th to the last op is corupted
#define PFE_UNSERIAL_RANGE 8 //consecutive ops to corrupt, 10, 11, 12

//param for returning err code (metadata corruption / advanced bit corruption)
#define PFE_FIRST_ERR_BEFORE_CUT 10 //count from the last(cut) op; e.g., the blk written at 10th to the last op will ret err code when read
#define PFE_ERR_RANGE 3 //consecutive ops to ret err, 10, 11, 12

/////////////////////////


//#define PFE_BLOCK_SIZE 4096 //splitting large IO to this size
#define PFE_BLOCK_SIZE 512 //splitting large IO to this size

extern const char *pfe_io_block_size_environ;
extern int pfe_io_block_size;

inline static void config_pfe_io_block_size() {
    char *tmp = NULL;
    tmp = getenv(pfe_io_block_size_environ);
    if (tmp == NULL) {
        fprintf(stderr, "config_pfe_io_block_size:%s doesn't exist,"
                "use default value %d.\n",
                pfe_io_block_size_environ, pfe_io_block_size);
    } else {
        fprintf(stderr, "config_pfe_io_block_size:%s exist,"
                "use its value %s.\n",
                pfe_io_block_size_environ, tmp);
       pfe_io_block_size = atoi(tmp);
    }   
}






//metadata for each recorded io
typedef struct pfe_io_header {
    uint64_t id; //IO#
    uint64_t ts; //timestamp
    uint64_t offset;
    uint64_t length;
    uint64_t datalog_offset;
    uint64_t cmd_type;
    uint64_t data_dir;
    //uint64_t marker; 
    uint64_t tid; //cmd->c_target->tid
} pfe_io_header_t;


typedef struct pfe_errblks_cache {
    char * buf;
    uint64_t length; //size in bytes
    uint64_t n_blks; // # of 4KB blks
} pfe_errblks_cache_t;


extern int pfe_io_header_fd;
extern int pfe_io_data_fd;
extern int pfe_err_blk_fd;
extern uint64_t pfe_io_id;
extern uint64_t pfe_io_datalog_offset;
extern pthread_mutex_t pfe_io_logs_mutex;
extern int pfe_fail_type_tgtd;
extern pfe_errblks_cache_t pfe_errblks_tgtd;
extern int pfe_enable_record;


char* pfe_alloc_rand_mask(uint32_t length);
void pfe_free_rand_mask(char *buf);

int pfe_addr_in_range(uint64_t offset);

void pfe_print_scsi_cmd(struct scsi_cmd *cmd);
void pfe_print_scsi_cdb(uint8_t *scb);
void pfe_print_header(pfe_io_header_t *header);

int pfe_log_io_req(struct scsi_cmd *cmd, uint64_t offset, uint64_t length, char *buf);
uint64_t pfe_split_io_log(int io_header_fd, int io_header_split_fd);

uint64_t pfe_get_io_cnt(int io_header_fd);
uint64_t pfe_replay_failure(int io_header_fd, int io_data_fd, int disk_fd, int fail_type, uint64_t start_id, uint64_t end_id, int err_blk_fd);

uint64_t pfe_replay_simple(int io_header_fd, int io_data_fd, int disk_fd, uint64_t first_id, uint64_t last_id);
uint64_t pfe_replay_bit(int io_header_fd, int io_data_fd, int disk_fd, uint64_t first_id, uint64_t last_id);
uint64_t pfe_replay_shorn(int io_header_fd, int io_data_fd, int disk_fd, uint64_t first_id, uint64_t last_id);
uint64_t pfe_replay_fly(int io_header_fd, int io_data_fd, int disk_fd, uint64_t first_id, uint64_t last_id);
uint64_t pfe_replay_unserial(int io_header_fd, int io_data_fd, int disk_fd, uint64_t first_id, uint64_t last_id, char * pattern_buf, uint32_t pattern_size);

uint64_t pfe_do_bitcorrupt(int disk_fd, char *data_buf, uint64_t length, uint64_t offset);
uint64_t pfe_do_shornwrite(int disk_fd, char *data_buf, uint64_t length, uint64_t offset);
uint64_t pfe_do_flywrite(int disk_fd, char *data_buf, uint64_t length, uint64_t offset);
uint64_t pfe_do_unserialwrite(int disk_fd, char *data_buf, uint64_t length, uint64_t offset);
uint64_t pfe_do_normalwrite(int disk_fd, char *data_buf, uint64_t length, uint64_t offset);

void pfe_build_errblks_cache(int err_blk_fd);
void pfe_free_errblks_cache(void);

char * pfe_build_unser_pattern_cache(uint32_t * pattern_size);
void pfe_free_unser_pattern_cache(char * pattern_buf);

#endif

