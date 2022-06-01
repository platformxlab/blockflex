#!/bin/bash
TYPE=ml_prep
OUT=ml_prep
TRACE_DIR=~/benchmarks/tracing
HOME=/home/breidys2/benchmarks/ml_prep
RUNS_DIR=$HOME/runs
DEV=nvme0n1
IMG_DIR="/mnt/nvme0n1/ml_prep/input"
IN_DIR="/home/breidys2/benchmarks/ml_prep/images/Data/DET/train/images"
DIR1=/mnt/nvme0n1/ml_prep/output/1
DIR2=/mnt/nvme0n1/ml_prep/output/2
DIR3=/mnt/nvme0n1/ml_prep/output/3
OUT_DIR=/mnt/nvme0n1/ml_prep/output

echo "Resetting sudo so it does not expire"
echo " Comment this out for batched runs"
sudo -k
sudo echo "Success"
for RUN in {1..5}; do
    echo "Running workload ${TYPE} and outputting to ${OUT}"
    sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"
    echo "Removing old output"
    [ -f $OUT.resize ] && rm $OUT.resize
    [ -d $IMG_DIR ] && rm -rf $IMG_DIR 
    [ ! -d $OUT_DIR ] && mkdir $OUT_DIR
    cp -r $IN_DIR $IMG_DIR
    [ -d $DIR1 ] && rm -rf $DIR1
    mkdir $DIR1
    [ -d $DIR2 ] && rm -rf $DIR2
    mkdir $DIR2
    [ -d $DIR3 ] && rm -rf $DIR3 
    mkdir $DIR3
    echo "Grabbing starting size"
    TEMP=$(du -s $OUT_DIR)
    TOKENS=( $TEMP )
    START=${TOKENS[0]}
    echo "Start size in KB ${START}" >> $OUT.resize
    echo "Starting tracing"
    sudo blktrace -d /dev/$DEV -a read -a write --output-dir $TRACE_DIR &
    echo "Start tracking sizes"
    [ -f "${OUT}_${RUN}.sizing" ] && rm "${OUT}_${RUN}.sizing"
    /home/breidys2/check.sh ${OUT_DIR} "${OUT}_${RUN}.sizing" & 
    SIZE_PID=$!
    sleep 2
    for i in {1..5}; do
        echo "Running workload: ${i}"
        (time python3 ml_prep_parallel.py) >> $OUT.ml
        if [ $i -lt 4 ]
        then 
            [ -d $DIR1 ] && rm -rf $DIR1
            mkdir $DIR1
            [ -d $DIR2 ] && rm -rf $DIR2
            mkdir $DIR2
            [ -d $DIR3 ] && rm -rf $DIR3 
            mkdir $DIR3
            sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"
        fi
    done
    echo "Getting new size"
    TEMP=$(du -s $OUT_DIR)
    TOKENS=( $TEMP )
    END=${TOKENS[0]}
    echo "End size in KB ${END}" >> $OUT.resize
    echo "Size difference in KB $( expr $END - $START )" >> $OUT.resize
    echo "Killing blktrace"
    sudo pkill blktrace 
    sudo pkill check
    #sudo kill -9 $SIZE_PID
    sleep 2
    echo "Parsing traced file"
    pushd $TRACE_DIR
    blkparse $DEV -a complete -f "%D %2c %8s %5T.%9t %5p %2a %3d %N\n" -o "${OUT}_${RUN}.trace"
    #If we want offsets in the trace
    #blkparse $DEV -a complete -o "${OUT}_offset.trace"
    popd
    echo "Copying output files to runs"
    cp $TRACE_DIR/"${OUT}_${RUN}.trace" $RUNS_DIR
    #If we want offsets in the trace
    #cp $TRACE_DIR/"${OUT}_offset.trace" $RUNS_DIR
    cp $OUT.ml $RUNS_DIR
    cp $OUT.resize $RUNS_DIR
    cp "${OUT}_${RUN}.sizing" $RUNS_DIR
    echo "Done"
done
