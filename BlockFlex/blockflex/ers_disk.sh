#!/bin/bash
#parameter: $1 badmark.bin ; $2 qid; $3 nsid
#

declare -i qid
declare -i nsid

let qid=1
let nsid=1

if [ $# -ge 1 ];then
    echo "using badmark file: $1"
fi


if [ $# -ge 2 ];then
    qid=$1    
fi
echo "using qid=$qid"  

if [ $# -ge 3 ];then
  nsid=$3
fi
echo "using nsid=$nsid"

if [ $# -ge 1 ];then
    
    ./cnexcmd -ersall -k 0 -q $qid -i $nsid -f $1
 else
    ./cnexcmd -ersall -k 0 -q $qid -i $nsid
fi