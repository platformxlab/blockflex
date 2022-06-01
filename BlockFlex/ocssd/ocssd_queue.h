#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/wait.h>
#include <mqueue.h>

#define PMODE 0655
#define MAX_MQUEUE_MSG_SIZE 8192
#define MAX_MSG 8192
#define QUEUE_NAME_0 "/harvest0"
#define QUEUE_NAME_1 "/harvest1"
#define ROOT_DIR
#define BLOCKFLEX_DIR "/home/osdi22ae/BlockFlex/blockflex/" // CHANGE THIS PATH
#define USR_DIR "/home/osdi22ae/BlockFlex/ocssd/" // CHANGE THIS PATH

extern mqd_t mqfd_1, mqfd_0, mqfd;

typedef struct Req
{
  uint64_t vssd_id;
  int mode;
  uint64_t offset;
  uint32_t length;
  char data[1024];
} Req;

char* concat(const char *s1, const char *s2); 