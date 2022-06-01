#include "ocssd_queue.h"
#include <time.h>

int main()
{
   ssize_t status;
   int count0 = 0, count1 = 0;
   struct mq_attr attr = {.mq_maxmsg=MAX_MSG, .mq_msgsize = MAX_MQUEUE_MSG_SIZE};

   mqd_t mqfd = mq_open(QUEUE_NAME_0, O_RDONLY|O_CREAT, PMODE, &attr);
   if(mqfd == -1) {
      perror("Cleaning process mq_open failure");
      exit(0);
   }
   Req req;
   status = 0;
   struct timespec ts;
   clock_gettime(CLOCK_REALTIME, &ts);
   ts.tv_sec += 1;  // Set for 2 second
   while(status != -1){
      status = mq_timedreceive(mqfd, (char*) &req, MAX_MSG, 0, &ts);
      clock_gettime(CLOCK_REALTIME, &ts);
      ts.tv_sec += 1;  // Set for 2 seconds
      if(status != -1) count0++;
   }
   mq_close(mqfd);

   mqfd = mq_open(QUEUE_NAME_1, O_RDONLY|O_CREAT, PMODE, &attr);
   if(mqfd == -1) {
      perror("Cleaning process mq_open failure");
      exit(0);
   }
   status = 0;
   clock_gettime(CLOCK_REALTIME, &ts);
   ts.tv_sec += 1;  // Set for 1 second
   while(status != -1){
      status = mq_timedreceive(mqfd, (char*) &req, MAX_MSG, 0, &ts);
      clock_gettime(CLOCK_REALTIME, &ts);
      ts.tv_sec += 1;  // Set for 1 second
      if(status != -1) count1++;
   }
   mq_close(mqfd);

   printf("Cleaning process done; <%d, %d> requests cleaned from queue 0 and 1 \n", count0, count1);
   
   return 0;
}