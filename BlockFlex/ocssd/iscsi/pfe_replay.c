/*replay IO based on io logs*/

#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/time.h>

#include "powerfault.h"
int pfe_io_header_fd = -1;
int pfe_io_data_fd = -1;
int pfe_err_blk_fd = -1;
uint64_t pfe_io_id = 1; //cnt from 1
uint64_t pfe_io_datalog_offset = 0;
pthread_mutex_t pfe_io_logs_mutex = PTHREAD_MUTEX_INITIALIZER;
int pfe_fail_type_tgtd = 0;
pfe_errblks_cache_t pfe_errblks_tgtd;
int pfe_enable_record = 0;


static struct option const long_options[] = {
	{"disk-file", required_argument, 0, 'f'},
	{"io-header-log", required_argument, 0, 'h'},
	{"io-data-log", required_argument, 0, 'd'},
	{"fail-type", required_argument, 0, 't'},
	{"start-io", required_argument, 0, 's'}, //first/start io to replay
	{"end-io", required_argument, 0, 'e'}, //last/end io to replay
	{"io-header-split-log", required_argument, 0, 'l'},
	{"pfe-err-blk", required_argument, 0, 'p'},//log storing io ret err block#
	{"split-io", required_argument, 0, 'b'},//1: split large io to 4KBs
	{0, 0, 0, 0},
};

int main(int argc, char **argv)
{
    //char * headerlog_file = "../pfe_scripts/pfe_io_header_log";
    //char * datalog_file = "../pfe_scripts/pfe_io_data_log";
    //char * disk_file = "/home/zhengm/iscsi-target-dev4M.re-fail";
   

     config_pfe_io_block_size(); //set pfe_io_block_size

    char * disk_file = NULL;
    char * headerlog_file = NULL;
    char * headerlog_split_file = NULL;
    char * datalog_file = NULL;
    char * err_blk_file = NULL;
    int fail_type = 0; //normal
    uint64_t start_io = 0;//first io to replay
    uint64_t end_io = 0;//last io to replay
    int split_io = 0;//1: split IO; 0: don't split
    int ret = 0;

    int op_ch = 0;
    int op_index = 0;

    while(1){
        op_ch = getopt_long(argc, argv, "f:h:d:t:s", \
                long_options, &op_index);
               
        if(op_ch == -1)//no more options
            break;
        
        switch(op_ch){
            case 'f'://disk_file
                disk_file = optarg;
                break;
            case 'h'://header log
                headerlog_file = optarg;
                break;
            case 'l'://header split log
                headerlog_split_file = optarg;
                break;
            case 'd'://data log
                datalog_file = optarg;
                break;
            case 'p'://ret err blocks log
                err_blk_file = optarg;
                break;
            case 't'://failure type
                fail_type = strtol(optarg, NULL, 0);
                break;
            case 's'://first/start io
                start_io = strtol(optarg, NULL, 0);
                break;
            case 'e'://last/end io
                end_io = strtol(optarg, NULL, 0);
                break;
            case 'b'://last/end io
                split_io = strtol(optarg, NULL, 0);
                break;
            case '?':
               /* getopt_long already printed an error message. */
               break;
            default:
               abort();
        }
    }

    //Print any remaining command line arguments (not options)
    if(optind < argc){
        printf("non-option ARGV-elements: ");
        while (optind < argc)
            printf ("%s ", argv[optind++]);
        putchar ('\n');
    }

    printf("PTE:Replay: disk_file = %s\n", disk_file);
    printf("PTE:Replay: io_header_file = %s\n", headerlog_file);
    printf("PTE:Replay: io_data_file = %s\n", datalog_file);
    printf("PTE:Replay: io_header_split_file = %s\n", headerlog_split_file);
    printf("PTE:Replay: io_err_blk_file = %s\n", err_blk_file);
    printf("PTE:Replay: fail_type = %d\n", fail_type);
    printf("PTE:Replay: start_io from input = %"PRIu64"\n", start_io);
    printf("PTE:Replay: end_io from input = %"PRIu64"\n", end_io);
    printf("PTE:Replay: split_io = %d\n", split_io);
    

    int io_header_fd = open(headerlog_file, O_RDONLY);
    if(io_header_fd < 0){
        printf("PFE: ERROR in opening %s !!!\n", headerlog_file);
        return -1; 
    }
    int io_data_fd = open(datalog_file, O_RDONLY);
    if(io_data_fd < 0){
        printf("PFE: ERROR in opening %s !!!\n", datalog_file);
        return -1; 
    }
    int disk_fd = open(disk_file, O_RDWR | O_SYNC);
    if(disk_fd < 0){
        printf("PFE: ERROR in opening %s !!!\n", disk_file);
        return -1; 
    }
 
    int io_header_split_fd = open(headerlog_split_file, O_CREAT |  O_RDWR , 00666);
    if(io_header_split_fd < 0){
        printf("PFE: ERROR in opening %s !!!\n", headerlog_split_file);
        return -1; 
    }

    int io_err_blk_fd = open(err_blk_file, O_RDWR | O_TRUNC | O_CREAT, 00666);
    if(io_err_blk_fd < 0){
        printf("PFE: ERROR in opening %s !!!\n", err_blk_file);
        return -1; 
    }

    //simple replay all recorded io without injecting emulated failures
    //ret = pfe_replay_simple(io_header_fd, io_data_fd, disk_fd);
    
    if(split_io){
        //split all recorded IO to 4KB writes
        //only need to change header log
        ret = pfe_split_io_log(io_header_fd, io_header_split_fd);
        fsync(io_header_split_fd);
        //replay using splitted io header log
        ret = pfe_replay_failure(io_header_split_fd, io_data_fd, disk_fd, fail_type, start_io, end_io, io_err_blk_fd);

    }else{
        //replay with orignial io header log 
        ret = pfe_replay_failure(io_header_fd, io_data_fd, disk_fd, fail_type, start_io, end_io, io_err_blk_fd);
    }

    if(ret < 0){
        printf("PFE: pfe replay FAILED !!!\n");
        return -1; 
    }
    

    close(io_header_fd);
    close(io_data_fd);
    close(disk_fd);
    close(io_header_split_fd);
    close(io_err_blk_fd);

    return 0;
}
