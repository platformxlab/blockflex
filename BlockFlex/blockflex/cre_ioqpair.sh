#!/bin/bash
#parameter: $1 qid ; $2 qsize; $3 priority
#

declare -i qid
declare -i qsize
declare -i prio

let qid=1
let qsize=1024
let prio=1

if [ $# -ge 1 ];then
    qid=$1
fi
echo "using qid=$qid"

if [ $# -ge 2 ];then 
    qsize=$2
fi
echo "using qsize=$qsize"    

if [ $# -ge 3 ];then 
    prio=$3
fi
echo "using priority=$prio"    

./cnexcmd -crtcq -q $qid -s $qsize

if [ $? -ne 0 ]; then
    echo "create completion queue $qid has failed, exiting..."
    exit 1
fi

./cnexcmd -crtsq -q $qid -s $qsize -p $prio

if [ $? -ne 0 ]; then
    echo "create submission queue $qid has failed, deleting completion queue $qid..."
    ./cnexcmd -delcq -q $qid
    exit 1
fi