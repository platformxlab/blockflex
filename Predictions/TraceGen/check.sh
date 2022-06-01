#!/bin/bash
DIR=$1
OUT=$2

while true; do
    sleep 1
    du -s $1 >> $2
done
