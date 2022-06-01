#!/bin/bash
TYPE=$1
OUT=$TYPE
GRAPH=$2
TRACE_DIR=/home/breidys2/benchmarks/tracing
CUR_DIR=/home/breidys2/benchmarks/graphchi-cpp
RUNS_DIR=$CUR_DIR/runs
DEV=nvme0n1
PR_DIR=/mnt/nvme0n1/pr

echo "Resetting sudo so it does not expire"
echo " Comment this out for batched runs"
sudo -k
sudo echo "Success"
echo "Running workload ${TYPE} and outputting to ${OUT}"
if [[ $TYPE == "pagerank" ]]
then
    for RUN in {1..5}; do
        echo "Removing old output"
        [ -d $PR_DIR ] && rm -rf $PR_DIR 
        mkdir $PR_DIR && cp $2 $PR_DIR 
        #[ -f $OUT.resize ] && rm $OUT.resize
        echo "Grabbing original Size"
        TEMP=$(du -s $PR_DIR)
        TOKENS=( $TEMP )
        START=${TOKENS[0]}
        echo "Start size in KB ${START}" >> $OUT.resize
        sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"
        echo "Starting tracing"
        sudo blktrace -d /dev/$DEV --output-dir $TRACE_DIR &
        echo "Start tracking size"
        [ -f "${OUT}_${RUN}.sizing" ] && rm "${OUT}_${RUN}.sizing"
        /home/breidys2/check.sh $PR_DIR "${OUT}_${RUN}.sizing" &
        SIZE_PID=$!
        #BLKTRACE_PID=$!
        sleep 2
        pushd $PR_DIR
        echo "Running workload"
        (time $CUR_DIR/bin/example_apps/$TYPE file $GRAPH niters 10) &> $OUT.pr
        sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"
        (time $CUR_DIR/bin/example_apps/$TYPE file $GRAPH niters 10) &> $OUT.pr
        sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"
        (time $CUR_DIR/bin/example_apps/$TYPE file $GRAPH niters 10) &> $OUT.pr
        sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"
        (time $CUR_DIR/bin/example_apps/$TYPE file $GRAPH niters 10) &> $OUT.pr
        popd
        echo "Getting new size"
        TEMP=$(du -s $PR_DIR)
        TOKENS=( $TEMP )
        END=${TOKENS[0]}
        echo "End size in KB ${END}" >> $OUT.resize
        echo "Size difference in KB $( expr $END - $START )" >> $OUT.resize
        echo "Killing blktrace"
        #sudo kill -9 $BLKTRACE_PID
        #sudo kill -9 $SIZE_PID
        sudo pkill check
        sudo pkill blktrace 
        sleep 2
        echo "Parsing traced file"
        pushd $TRACE_DIR
        blkparse $DEV -a complete -f "%D %2c %8s %5T.%9t %5p %2a %3d %N\n" -o "${OUT}_${RUN}.trace"
        #If we want to have offsets in the trace
        #blkparse $DEV -a complete -o "${OUT}_offset.trace"
        popd
        echo "Copying output files to runs"
        cp $TRACE_DIR/"${OUT}_${RUN}.trace" $RUNS_DIR
        #If we want to have offsets in the trace
        #cp $TRACE_DIR/"${OUT}_offset.trace" $RUNS_DIR
        cp $PR_DIR/$OUT.pr $RUNS_DIR
        cp $OUT.resize $RUNS_DIR
        mv "${OUT}_${RUN}.sizing" $RUNS_DIR
done
else
    echo "Please specify a valid workload" && exit
fi
echo "Done"
