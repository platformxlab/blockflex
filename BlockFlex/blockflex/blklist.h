#ifndef _BLKLIST_H
#define _BLKLIST_H

#include "bflex.h"

typedef struct node{
    u32 ppa;
    int data;
    int rw;
    struct node* next;
} node_t;

//node_t *alloc_blk_list[CFG_NAND_CHANNEL_NUM] = {NULL};
//node_t *free_blk_list[CFG_NAND_CHANNEL_NUM] = {NULL};
//node_t *dead_blk_list[CFG_NAND_CHANNEL_NUM] = {NULL};

int add_alloc_list(int chl_id, node_t * node);
node_t* del_alloc_list(int chl_id);
node_t* find_alloc_list(int chl_id, u32 ppa);
int free_alloc_list(int chl_id);
void print_alloc_list(int chl_id);
int add_free_list(int chl_id, node_t* node);
node_t* del_free_list(int chl_id);
int free_free_list(int chl_id);
void print_free_list(int chl_id);
int add_dead_list(int chl_id, node_t * node);
node_t* del_dead_list(int chl_id);
int free_dead_list(int chl_id);
void print_dead_list(int chl_id);
int add_alloc_list_superblk(int chl_id, node_t * node);
node_t* del_alloc_list_superblk(int chl_id);
node_t* find_alloc_list_superblk(int chl_id, u32 ppa);
int free_alloc_list_superblk(int chl_id);
void print_alloc_list_super(int chl_id);
int add_free_list_superblk(int chl_id, node_t* node);
node_t* del_free_list_superblk(int chl_id);
int free_free_list_superblk(int chl_id);
void print_free_list_superblk(int chl_id);
int add_dead_list_superblk(int chl_id, node_t * node);
node_t* del_dead_list_superblk(int chl_id);
int free_dead_list_superblk(int chl_id);
void print_dead_list_superblk(int chl_id);

#endif
