#!/bin/bash
#parameter: $1 block; $2 badmark.bin ; $3 qid; $4 nsid
#
declare -i block
declare -i qid
declare -i nsid

let qid=1
let nsid=1

if [ $# -lt 1 ];then
    echo "ers_block.sh block# [badmark qid# nsid#]"
    exit 1
fi
block=$1

if [ $# -ge 2 ];then
    echo "using badmark file: $2"    
fi

if [ $# -ge 3 ];then
    qid=$3   
fi
echo "using qid=$qid"  

if [ $# -ge 4 ];then
  nsid=$4
fi
echo "using nsid=$nsid"

if [ $# -ge 2 ];then    
    ./cnexcmd -ersblock -k 0 -q $qid -i $nsid -b $block -f $2
 else
    ./cnexcmd -ersblock -k 0 -q $qid -i $nsid -b $block
fi