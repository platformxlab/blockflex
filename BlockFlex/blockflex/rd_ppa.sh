#!/bin/bash
#parameter: 
#

declare -i qid
declare -i nsid
declare -i block
declare -i page
declare -i lun
declare -i plane
declare -i ep
declare -i channel
declare -i ppa

# EP mode: 1, CH mode: 0
declare -i addr_mode

let qid=1
let nsid=1
let nlb=1
let addr_mode=0

# default block#1024
let block=1000
let page=0
let lun=0
let plane=0
let ep=0
let channel=0

data_out="datafile.out"
meta_out="metafile.out"

if [ $# -lt 3 ];then
    echo "please provide block# page# lun# pl# ep# ch# [nlb# addr_mode datafile metafile]"
fi

block=$1
page=$2
lun=$3

if [ $# -ge 4 ]; then
    plane=$4
fi

if [ $# -ge 5 ]; then
    ep=$5
fi

if [ $# -ge 6 ]; then
    channel=$6
fi

if [ $# -ge 7 ]; then
    nlb=$7
fi

if [ $# -ge 8 ]; then
    addr_mode=$8
fi

if [ $addr_mode -eq 0 ]; then 
   echo "using CH++ mode"
else
   echo "using EP++ mode"
fi

if [ $# -ge 9 ];then
    data_out=$9
fi

if [ $# -ge 10 ];then
    meta_out=$10
fi

#address bits (Toshiba) block:13 pg:8 lun:2 pl:2 ep:2 ch:4
let "ppa=(block<<18)"
let "ppa+=(page<<10)"
let "ppa+=(lun<<8)"
let "ppa+=(pl<<6)"
let "ppa+=(ep<<4)"
let ppa+=channel

echo "using ppa_addr: 0n$ppa"

./cnexcmd -rdppasync -k 0 -q $qid -i $nsid -f $data_out -m $meta_out -a $ppa -n $nlb -c $addr_mode 

