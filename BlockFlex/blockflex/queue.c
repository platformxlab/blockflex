#include<stdio.h>
#include<stdlib.h>
#include "queue.h"


Queue* createQueue(int max_element) {

	Queue *queue;
	queue = (Queue *)malloc(sizeof(Queue));

	queue->elements = (int *)malloc(sizeof(int)*max_element);
	queue->size = 0;
	queue->capacity = max_element;
	queue->front = 0;
	queue->tail = -1;

	return queue;
}

int dequeue(Queue *queue) {
	int element;	
	if(queue->size == 0) {
		//printf("queue is empty\n");
		return -1;
	}
	else {
		//printf("front=%d\n", queue->front);
		element = queue->elements[queue->front];
		queue->size--;
		queue->front++;

		if(queue->front == queue->capacity) {
			queue->front = 0;
		}

		return element;
	}
}

int enqueue(Queue *queue, int element) {
	if(queue->size == queue->capacity) {
		printf("queue is full\n");
		return -1;
	}
	else {
		queue->size++;
		queue->tail = queue->tail + 1;

		if(queue->tail == queue->capacity)
			queue->tail = 0;

		queue->elements[queue->tail] = element;
		return 0;
	}

}


int front(Queue *queue) {
	if(queue->size == 0) {
		printf("queue is empty\n");
		exit(0);
	}

	return queue->elements[queue->front];
	
}
