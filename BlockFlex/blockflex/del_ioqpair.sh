#!/bin/bash
#parameter: $1 qid ;
#

if [ $# -lt 1 ];then
    echo "please provide the qid to delete, such as"
    echo "./del_ioqpair 1"
    exit 1
fi

echo "deleting submission queue $1 first"
./cnexcmd -delsq -q $1

echo "deleting completion queue $1 next"
./cnexcmd -delcq -q $1