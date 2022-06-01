all: mqueue_cleaner iscsi

.PHONY: mqueue_cleaner
mqueue_cleaner: mqueue_cleaner.c ocssd_queue.h
	gcc mqueue_cleaner.c -o mqueue_cleaner -lrt

.PHONY: iscsi
iscsi:
	make -C iscsi

.PHONY: clean
clean:
	rm -f mqueue_cleaner mqueue_client
	make -C iscsi clean
