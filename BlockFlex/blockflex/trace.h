#ifndef _TRACE_H
#define _TRACE_H

#include "bflex.h"
#include <pthread.h>

typedef struct event{
    //Want a timestamp
    double ts;
    //Want event type
    char type;
    //Want event offset
    int offset;
    //Want event size
    int size;
    //Link to next event making a list
    struct event* next;
    //debug
    int num;
} event_t;

typedef struct trace{
    event_t* file_head;
    event_t** head;
    event_t** q;
    pthread_mutex_t * q_locks;
    pthread_mutex_t list_mutex;
    uint32_t * f_map;
    char* in_file;
    int end_val;
} trace_t;

//259,0    9       35     0.000110432     0  C  WS 65536
//
event_t* get_list(char* filename, int end_val);
//event_t** get_lists(char* filename, int t_chl, int h_chl, int end_val);
void debug_print(event_t* ev);
void free_list(event_t* head);
void free_trace(trace_t * t);
trace_t * get_trace(char* filename, int end_val);
int get_ops(trace_t * t, int chl, event_t* head, int n_ops);
int list_len(event_t * head);
#endif
