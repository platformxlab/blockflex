#include <stdio.h>
#include <stdlib.h>

#include "blklist.h"
#include "bflex.h"

node_t *alloc_blk_list[CFG_NAND_CHANNEL_NUM] = {NULL};
node_t *free_blk_list[CFG_NAND_CHANNEL_NUM] = {NULL};
node_t *dead_blk_list[CFG_NAND_CHANNEL_NUM] = {NULL};


node_t *alloc_superblk_list[CFG_NAND_CHANNEL_NUM] = {NULL};
node_t *free_superblk_list[CFG_NAND_CHANNEL_NUM] = {NULL};
node_t *dead_superblk_list[CFG_NAND_CHANNEL_NUM] = {NULL};

//int add_alloc_list(int chl_id, u32 ppa, int ers_cnt) {
int add_alloc_list(int chl_id, node_t * node) {

	//node_t* node = malloc(sizeof(node_t));
	//node->ppa = ppa;
	//node->data = ers_cnt;
	node->next = NULL;
	node_t* current = NULL;

	if(alloc_blk_list[chl_id] == NULL) {
		alloc_blk_list[chl_id] = node;	
	}
	else{

	    if(alloc_blk_list[chl_id]->data >= node->data) {
		node->next = alloc_blk_list[chl_id];
		alloc_blk_list[chl_id] = node;
	    }
	    else{
		current = alloc_blk_list[chl_id];
		while(current->next != NULL && current->next->data < node->data){
		    current = current->next;
		}

		node->next = current->next;
		current->next = node;

	    }
	}

	return 0;
}

node_t* del_alloc_list(int chl_id) {
	node_t* current = NULL;

	if(alloc_blk_list[chl_id] == NULL) {
		return NULL;
	}
	else {
	    current = alloc_blk_list[chl_id];
	    alloc_blk_list[chl_id] = current->next;
	    current->next = NULL;
	    return current;
	}

}

node_t* find_alloc_list(int chl_id, u32 ppa) {
	node_t* current = NULL;
	node_t* cur_next = NULL;

	if(alloc_blk_list[chl_id] == NULL)
		return NULL;
	else {
		current = alloc_blk_list[chl_id];
		cur_next = current->next;

		if(current->ppa == ppa) {
			alloc_blk_list[chl_id] = current->next;
			current->next = NULL;
			return current;
		}
		else{
			while(cur_next != NULL){
				if(cur_next->ppa == ppa) {
					current->next = cur_next->next;
					cur_next->next = NULL;
					return cur_next;
				}	
				else{
					current = cur_next;
					cur_next = cur_next->next;
				}
			}
			return NULL;
		}
	}
}

int free_alloc_list(int chl_id) {
	node_t* current = NULL;
	node_t* temp = NULL;

	current = alloc_blk_list[chl_id];
	

	while(current != NULL) {
		temp = current;
		current = current->next;
		temp->next = NULL;
		free(temp);		
	}

	return 0;
}



void print_alloc_list(int chl_id) {
	node_t* current = NULL;

	current = alloc_blk_list[chl_id];

	while(current != NULL) {
		printf("[%d,%d]->", current->ppa, current->data);
		current = current->next;
	}
	printf("NULL\n");
	return;
}


//int add_free_list(int chl_id, u32 ppa, int ers_cnt) {
int add_free_list(int chl_id, node_t* node) {
	//node_t* node = malloc(sizeof(node_t));
	//node->ppa = ppa;
	//node->data = ers_cnt;
	node->next = NULL;
	node_t* current = NULL;

	if(free_blk_list[chl_id] == NULL) {
		free_blk_list[chl_id] = node;	
	}
	else{

	    if(free_blk_list[chl_id]->data <= node->data) {
		node->next = free_blk_list[chl_id];
		free_blk_list[chl_id] = node;
	    }
	    else{
		current = free_blk_list[chl_id];
		while(current->next != NULL && current->next->data > node->data){
		    current = current->next;
		}

		node->next = current->next;
		current->next = node;

	    }
	}

	return 0;
}


node_t* del_free_list(int chl_id) {
	node_t* current = NULL;

	if(free_blk_list[chl_id] == NULL) {
		return NULL;
	}
	else {
	    current = free_blk_list[chl_id];
	    free_blk_list[chl_id] = current->next;
	    current->next = NULL;
	    return current;
	}

}


int free_free_list(int chl_id) {
	node_t* current = NULL;
	node_t* temp = NULL;

	current = free_blk_list[chl_id];
	

	while(current != NULL) {
		temp = current;
		current = current->next;
		temp->next = NULL;
		free(temp);		
	}

	return 0;
}




void print_free_list(int chl_id) {
	node_t* current = NULL;

	current = free_blk_list[chl_id];

	while(current != NULL) {
		printf("[%d,%d]->", current->ppa, current->data);
		current = current->next;
	}
	printf("NULL\n");
	return;
}


//int add_dead_list(int chl_id, u32 ppa, int ers_cnt) {
int add_dead_list(int chl_id, node_t * node) {
	//node_t* node = malloc(sizeof(node_t));
	//node->ppa = ppa;
	//node->data = ers_cnt;
	node->next = NULL;
	node_t* current = NULL;

	if(dead_blk_list[chl_id] == NULL) {
		dead_blk_list[chl_id] = node;	
	}
	else{
	    current = dead_blk_list[chl_id];
	    while(current->next != NULL) {
		current = current->next;
	    }
	    
	    current->next = node;

	}

	return 0;
}


node_t* del_dead_list(int chl_id) {
	node_t* current = NULL;

	if(dead_blk_list[chl_id] == NULL) {
		return NULL;
	}
	else {
	    current = dead_blk_list[chl_id];
	    dead_blk_list[chl_id] = current->next;
	    current->next = NULL;
	    return current;
	}

}


int free_dead_list(int chl_id) {
	node_t* current = NULL;
	node_t* temp = NULL;

	current = dead_blk_list[chl_id];
	

	while(current != NULL) {
		temp = current;
		current = current->next;
		temp->next = NULL;
		free(temp);		
	}

	return 0;
}




void print_dead_list(int chl_id) {
	node_t* current = NULL;

	current = dead_blk_list[chl_id];

	while(current != NULL) {
		printf("[%d,%d]->", current->ppa, current->data);
		current = current->next;
	}
	printf("NULL\n");
	return;
}







//int add_alloc_list(int chl_id, u32 ppa, int ers_cnt) {
int add_alloc_list_superblk(int chl_id, node_t * node) {

	//node_t* node = malloc(sizeof(node_t));
	//node->ppa = ppa;
	//node->data = ers_cnt;
	node->next = NULL;
	node_t* current = NULL;

	if(alloc_superblk_list[chl_id] == NULL) {
		alloc_superblk_list[chl_id] = node;	
	}
	else{

	    if(alloc_superblk_list[chl_id]->data >= node->data) {
		node->next = alloc_superblk_list[chl_id];
		alloc_superblk_list[chl_id] = node;
	    }
	    else{
		current = alloc_superblk_list[chl_id];
		while(current->next != NULL && current->next->data < node->data){
		    current = current->next;
		}

		node->next = current->next;
		current->next = node;

	    }
	}

	return 0;
}

node_t* del_alloc_list_superblk(int chl_id) {
	node_t* current = NULL;

	if(alloc_superblk_list[chl_id] == NULL) {
		return NULL;
	}
	else {
	    current = alloc_superblk_list[chl_id];
	    alloc_superblk_list[chl_id] = current->next;
	    current->next = NULL;
	    return current;
	}

}

node_t* find_alloc_list_superblk(int chl_id, u32 ppa) {
	node_t* current = NULL;
	node_t* cur_next = NULL;

	if(alloc_superblk_list[chl_id] == NULL)
		return NULL;
	else {
		current = alloc_superblk_list[chl_id];
		cur_next = current->next;

		if(current->ppa == ppa) {
			alloc_superblk_list[chl_id] = current->next;
			current->next = NULL;
			return current;
		}
		else{
			while(cur_next != NULL){
				if(cur_next->ppa == ppa) {
					current->next = cur_next->next;
					cur_next->next = NULL;
					return cur_next;
				}	
				else{
					current = cur_next;
					cur_next = cur_next->next;
				}
			}
			return NULL;
		}
	}
}


int free_alloc_list_superblk(int chl_id) {
	node_t* current = NULL;
	node_t* temp = NULL;

	current = alloc_superblk_list[chl_id];
	

	while(current != NULL) {
		temp = current;
		current = current->next;
		temp->next = NULL;
		free(temp);		
	}

	return 0;
}



void print_alloc_list_super(int chl_id) {
	node_t* current = NULL;

	current = alloc_superblk_list[chl_id];

	while(current != NULL) {
		printf("[%d,%d]->", current->ppa, current->data);
		current = current->next;
	}
	printf("NULL\n");
	return;
}


//int add_free_list(int chl_id, u32 ppa, int ers_cnt) {
int add_free_list_superblk(int chl_id, node_t* node) {
	//node_t* node = malloc(sizeof(node_t));
	//node->ppa = ppa;
	//node->data = ers_cnt;
	node->next = NULL;
	node_t* current = NULL;

	if(free_superblk_list[chl_id] == NULL) {
		free_superblk_list[chl_id] = node;	
	}
	else{

	    if(free_superblk_list[chl_id]->data <= node->data) {
		node->next = free_superblk_list[chl_id];
		free_superblk_list[chl_id] = node;
	    }
	    else{
		current = free_superblk_list[chl_id];
		while(current->next != NULL && current->next->data > node->data){
		    current = current->next;
		}

		node->next = current->next;
		current->next = node;

	    }
	}

	return 0;
}


node_t* del_free_list_superblk(int chl_id) {
	node_t* current = NULL;

	if(free_superblk_list[chl_id] == NULL) {
		return NULL;
	}
	else {
	    current = free_superblk_list[chl_id];
	    free_superblk_list[chl_id] = current->next;
	    current->next = NULL;
	    return current;
	}

}


int free_free_list_superblk(int chl_id) {
	node_t* current = NULL;
	node_t* temp = NULL;

	current = free_superblk_list[chl_id];
	

	while(current != NULL) {
		temp = current;
		current = current->next;
		temp->next = NULL;
		free(temp);		
	}

	return 0;
}




void print_free_list_superblk(int chl_id) {
	node_t* current = NULL;

	current = free_superblk_list[chl_id];

	while(current != NULL) {
		printf("[%d,%d]->", current->ppa, current->data);
		current = current->next;
	}
	printf("NULL\n");
	return;
}


//int add_dead_list(int chl_id, u32 ppa, int ers_cnt) {
int add_dead_list_superblk(int chl_id, node_t * node) {
	//node_t* node = malloc(sizeof(node_t));
	//node->ppa = ppa;
	//node->data = ers_cnt;
	node->next = NULL;
	node_t* current = NULL;

	if(dead_superblk_list[chl_id] == NULL) {
		dead_superblk_list[chl_id] = node;	
	}
	else{
	    current = dead_superblk_list[chl_id];
	    while(current->next != NULL) {
		current = current->next;
	    }
	    
	    current->next = node;

	}

	return 0;
}


node_t* del_dead_list_superblk(int chl_id) {
	node_t* current = NULL;

	if(dead_superblk_list[chl_id] == NULL) {
		return NULL;
	}
	else {
	    current = dead_superblk_list[chl_id];
	    dead_superblk_list[chl_id] = current->next;
	    current->next = NULL;
	    return current;
	}

}


int free_dead_list_superblk(int chl_id) {
	node_t* current = NULL;
	node_t* temp = NULL;

	current = dead_superblk_list[chl_id];
	

	while(current != NULL) {
		temp = current;
		current = current->next;
		temp->next = NULL;
		free(temp);		
	}

	return 0;
}




void print_dead_list_superblk(int chl_id) {
	node_t* current = NULL;

	current = dead_superblk_list[chl_id];

	while(current != NULL) {
		printf("[%d,%d]->", current->ppa, current->data);
		current = current->next;
	}
	printf("NULL\n");
	return;
}


