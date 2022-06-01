#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "trace.h"
#include "bflex.h"

//We need one function to create the list and return the head of the list
event_t* get_list(char* filename, int end_val) {
    //Create the head of the list
    FILE *fp;
    //Linked list nodes to be used
    event_t *head, *cur, *next;
    //all are filler except type is R/W
    char complete, plus, aa[20], bb[20], type[20];
    //a,b,c are filler, off = offset, size = size
    int a,b,c,off,size;
    //Timestamp of the event
    double ts;
    int cnt = 1;
      
    fp = fopen(filename, "r");
    head = malloc(sizeof(event_t));
    cur = head;
    uint64_t max = 0;
    int cur_len = 0;
    //Read in the next line of the trace 
    while(fscanf(fp, "%s %d %d %lf %d %c %s %d %c %d %s",aa,&a,&b,&ts,&c,&complete,type,&off,&plus,&size,bb) > 0) {
        next = malloc(sizeof(event_t));
        cur->next = next;
        next->ts = ts;
        next->type = type[0];
        //also in sectors so we convert to pages
        next->offset = off/8;
        if (next->offset > max) max = next->offset;
        //blkparse reports in 512 bytes so we convert to pages
        next->size = size/8;
        next->num = cnt++;
        next->next = NULL;
        cur = cur->next;
        if (++cur_len == end_val) break;
    }
    //Debug stuff
    //fprintf(stdout, "%ld", max);
    //fflush(stdout);
    //close the file descriptor
    fclose(fp);
    //return the head of the list
    return head;
}


int get_ops(trace_t * t, int chl, event_t* copy, int n_ops) {
    //event_t * copy = head;
    int start = n_ops;
    event_t * q_next;
    //First check the queue
    pthread_mutex_lock(&t->q_locks[chl]);
    while(n_ops > 0 && t->q[chl]->next != NULL) {
        //Update
        q_next = t->q[chl]->next;
        free(t->q[chl]);
        t->q[chl] = q_next;

        //Create new event
        copy->next = malloc(sizeof(event_t));
        copy = copy->next;

        //Copy over the info
        copy->size = t->q[chl]->size;
        copy->ts = t->q[chl]->ts;
        copy->offset = t->q[chl]->offset;
        copy->type = t->q[chl]->type;
        copy->num = t->q[chl]->num;
        copy->next = NULL;
        n_ops--;
    }
    pthread_mutex_unlock(&t->q_locks[chl]);
    if (n_ops == 0) return start-n_ops;
    //Then we grab from the global list
    pthread_mutex_lock(&(t->list_mutex));
    while(n_ops > 0 && t->file_head != NULL) {
        if (t->file_head->type == 'W' || t->f_map[t->file_head->offset] == chl || t->f_map[t->file_head->offset] ==-1) {
            int off = t->file_head->offset;
            int sz = t->file_head->size;
            while(sz > 0) {
                t->f_map[off] = chl;
                sz--;
                off++;
            }
            //fprintf(stdout, "Giving %d to %d\n", t->file_head->num, chl);
            //fflush(stdout);
            copy->next = malloc(sizeof(event_t));
            copy = copy->next;
            //Copy over the info
            copy->ts = t->file_head->ts;
            copy->type = t->file_head->type;
            copy->offset = t->file_head->offset;
            copy->size = t->file_head->size;
            copy->num = t->file_head->num;
            copy->next = NULL;
            n_ops--;
        } else {
            //add to the appropriate q
            int cur = t->f_map[t->file_head->offset];
            pthread_mutex_lock(&t->q_locks[cur]);
            event_t* temp = t->q[cur];
            //advance to the end
            //int cc = 0;
            while(temp->next != NULL) {
                //cc++;
                temp = temp->next;
            }
            //fprintf(stdout, "%d\n", cc);
            //fflush(stdout);
            temp->next = malloc(sizeof(event_t));
            //fprintf(stdout, "Giving %d to %d\n", t->file_head->num, cur);
            //fflush(stdout);
            temp = temp->next;
            temp->ts = t->file_head->ts;
            temp->type = t->file_head->type;
            temp->offset = t->file_head->offset;
            temp->size = t->file_head->size;
            temp->num = t->file_head->num;
            temp->next = NULL;
            pthread_mutex_unlock(&t->q_locks[cur]);
        }
        q_next = t->file_head->next;
        free(t->file_head);
        t->file_head = q_next;
    }
    if (t->file_head == NULL) t->file_head = get_list(t->in_file, t->end_val);
    pthread_mutex_unlock(&(t->list_mutex));
    return start - n_ops;
}
 
//Prep the ds for on-demand op filling
trace_t * get_trace(char* filename, int end_val) {
    trace_t * ret_t = malloc(sizeof(trace_t));
    //Prep the global list
    ret_t->f_map = malloc((1 << 30) * sizeof(uint32_t));
    memset(ret_t->f_map, -1, (1 << 30) * sizeof(uint32_t));
    ret_t->file_head = get_list(filename, end_val);
    //Setup the temp queues
    ret_t->q = malloc(16 * sizeof(event_t *));
    ret_t->head = malloc(16 * sizeof(event_t *));
    int i;
    pthread_mutex_init(&(ret_t->list_mutex),NULL);
    ret_t->q_locks = malloc(16 * sizeof(pthread_mutex_t));
    for(i = 0; i < 16; i++) {
        ret_t->q[i] = malloc(sizeof(event_t));
        ret_t->q[i]->next = NULL;
        pthread_mutex_init(&(ret_t->q_locks[i]),NULL);
        ret_t->head[i] = malloc(sizeof(event_t));
        ret_t->head[i]->next = NULL;
    }
    ret_t->in_file = filename;
    ret_t->end_val = end_val;
    return ret_t;
}

//Free the aux ds for on-demand op filling
void free_trace(trace_t * t) {
    free_list(t->file_head);
    //Free aux ds
    int i;
    for(i = 0; i < 16; i++) {
        free_list(t->q[i]);
        free_list(t->head[i]);
    }
    free(t->q_locks);
    free(t->q);
    free(t->head);
    free(t->f_map);
    free(t);
}


//Aux functions
int list_len(event_t * head) {
    int len = 0;
    while(head!=NULL) {
        len++;
        head = head->next;
    }
    return len;
}

void debug_print(event_t* ev) {
    fprintf(stdout, "(%.6f) (%c) (%d) (%d)\n",ev->ts, ev->type,ev->offset, ev->size);
    fflush(stdout);
}

//traverse the list and free it
void free_list(event_t* head) {
    if (head == NULL) return;
    event_t* next;
    //Iterate over the list
    while(head->next != NULL) {
        next = head->next;
        free(head);
        head = next;
    }
    free(head);
    //Done
}

//TODO YOU CAN IGNORE THIS SINCE IT PREALLOCATES EVERYTHING, SWITCHED TO ON DEMAND
//Version for Multithreaded execution so we actually see a benefit to more channels
//event_t** get_lists(char* filename, int n_chl, int h_chl, int end_val) {
//    //Create the head of the list
//    FILE *fp;
//    //Linked list nodes to be used
//    event_t **head, **cur, *next;
//    //all are filler except type is R/W
//    char complete, plus, aa[20], bb[20], type[20];
//    //a,b,c are filler, off = offset, size = size
//    int a,b,c,off,size,i;
//    //Timestamp of the event
//    double ts;
//    int t_chl = n_chl + h_chl;
//    int cur_chl = n_chl;
//
//    uint32_t * mapping = malloc((1 << 30) * sizeof(uint32_t));
//    uint32_t * counts = malloc(t_chl * sizeof(uint32_t));
//    memset(mapping, 0, (1 << 30) * sizeof(uint32_t));
//    memset(counts, 0, t_chl * sizeof(uint32_t));
//
//    fp = fopen(filename, "r");
//    head = malloc(sizeof(event_t*) * t_chl);
//    cur = malloc(sizeof(event_t*) * t_chl);
//    for(i = 0; i < t_chl; i++) {
//        head[i] = malloc(sizeof(event_t));
//        cur[i] = head[i];
//        head[i]->next = NULL;
//    }
//    //Read in the next line of the trace 
//    int cnt =0;
//    int ch = 0;
//    while(fscanf(fp, "%s %d %d %lf %d %c %s %d %c %d %s",aa,&a,&b,&ts,&c,&complete,type,&off,&plus,&size,bb) > 0) {
//        next = malloc(sizeof(event_t));
//        ch = 0;
//        //Overwrite the previous balancing if this is a read that needs to be mapped.
//        next->ts = ts;
//        next->type = type[0];
//        //also in sectors so we convert to pages
//        next->offset = off/8;
//        //blkparse reports in 512 bytes so we convert to pages
//        next->size = size/8;
//        next->next = NULL;
//        //Spray the writes evenly, writes go anywhere
//        int lim = next->type == 'W' ? cur_chl : n_chl;
//        for(i = 0; i < lim; i++) {
//            if (counts[ch] > counts[i]) {
//                ch = i;
//            }
//        }
//        //If we have a read that has been written we want to place it to the correct channel
//        if (next->type == 'R') {
//            int off = next->offset;
//            int sz = next->size;
//            while(sz > 0) {
//                if (mapping[off]) {
//                    ch = mapping[off];
//                    break;
//                }
//                off++;
//                sz--;
//            }
//        }
//        cur[ch]->next = next;
//        cur[ch] = cur[ch]->next;
//        //Update the mappings for future reads later
//        if (next-> type == 'W') {
//            int off = next->offset;
//            int sz = next->size;
//            while(sz > 0) {
//                mapping[off] = ch;
//                sz--;
//                off++;
//            }
//        }
//        counts[ch] += next->size;
//        cnt++;
//        if (cnt == end_val / 4) {
//            //start harvesting
//            cur_chl += h_chl;
//            //Reset the op counts here to avoid sticking everything in the harvested storage for now
//            memset(counts, 0, t_chl * sizeof(uint32_t));
//        }
//        if (cnt == end_val) {
//            //done
//            break;
//        }
//    }
//    //close the file descriptor
//    fclose(fp);
//    free(cur);
//    free(mapping);
//    //return the head of the list
//    return head;
//}
