#!/bin/bash
#parameter: block# page# lun# ch# [nlb# datafile metafile]
#

declare -i qid
declare -i nsid
declare -i block
declare -i page
declare -i lun
declare -i channel
declare -i ppa

# EP mode: 1, CH mode: 0
declare -i addr_mode

let qid=1
let nsid=1
let nlb=16

# default block#1024
let block=1000
let page=0
let lun=0
let channel=0

data_in="datafile.in"
meta_in="metafile.in"

if [ $# -lt 4 ];then
    echo "please provide block# page# lun# ch# [nlb# datafile metafile]"
fi

block=$1
page=$2
lun=$3
channel=$4
nlb=$5

# if nlb > 16, it must be CH mode
if [ $nlb -gt 16 ]; then
    echo "nlb:$nlb greater than 16 (for Micron, nlb > 8), it must be CH mode and SINGLE_PLANE "
    addr_mode=0
fi

# if nlb ==8  or 16, it must be EP mode
if [ $nlb -eq 16 -o $nlb -eq 8 ]; then
    addr_mode=1
fi

if [ $addr_mode -eq 0 ]; then
     echo "in CH++ mode, we must start from CH:0"
     channel=0
fi

if [ $# -ge 6 ];then
    data_in=$6
fi

if [ $# -ge 7 ];then
    meta_in=$7
fi

#address bits (Toshiba) block:13 pg:8 lun:2 pl:2 ep:2 ch:4
let "ppa=(block<<18)"
let "ppa+=(page<<10)"
let "ppa+=(lun<<8)"
let ppa+=channel

echo "using ppa_addr: $ppa "

./cnexcmd -wrppasync -k 0 -q $qid -i $nsid -f $data_in -m $meta_in -a $ppa -n $nlb -c $addr_mode 
#./cnexcmd -wrppasync -k 0 -q $qid -i $nsid -a $ppa -n $nlb -c $addr_mode 

