#ifndef _QUEUE_H
#define _QUEUE_H

typedef struct Queue {
        int capacity;
        int size;
        int front;
        int tail;
        int *elements;
} Queue;

int dequeue(Queue *queue);
int enqueue(Queue *queue, int element);
Queue* createQueue(int max_element);
int front(Queue *queue);

#endif

