#!/bin/bash
TYPE=$1
OUT=$TYPE
TRACE_DIR=~/benchmarks/tracing
RUNS_DIR=$HADOOP_HOME/runs
HDFS=/mnt/nvme0n1/hdfs/
DEV=nvme0n1
IN_SZ=750000000

echo "Resetting sudo so it does not expire"
echo " Comment this out for batched runs"
sudo -k
sudo echo "Success"
echo "Running workload ${TYPE} and outputting to ${OUT}"
if [[ $TYPE == "terasort" ]]
then
    for RUN in {1..5}; do
        echo "Removing old output"
        hadoop fs -rm -r /user/breidys2/teraoutput
        hadoop fs -rm -r /user/breidys2/terainput
        [ -f $OUT.resize ] && rm $OUT.resize
        [ -f "${OUT}_${RUN}.sizing" ] && rm "${OUT}_${RUN}.sizing"
        echo "Grabbing hdfs Size"
        TEMP=$(du -s $HDFS)
        TOKENS=( $TEMP )
        START=${TOKENS[0]}
        echo "Start size in KB ${START}" >> $OUT.resize
        sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"

        #echo "Generating new input"
        #hadoop jar $HADOOP_EXAMPLE_JAR teragen $IN_SZ /user/breidys2/terainput
        #sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"
        #sleep 10
        #sudo pkill blktrace 
        #pushd $TRACE_DIR
        #blkparse $DEV -a complete -f "%D %2c %8s %5T.%9t %5p %2a %3d %N\n" -o $OUT.trace
        #blkparse $DEV -a complete -o "${OUT}_prep".trace
        #popd
        #cp $TRACE_DIR/"${OUT}_prep.trace" $RUNS_DIR

        #echo "Starting tracing"
        #sudo blktrace -d /dev/$DEV --output-dir $TRACE_DIR &
        #BLKTRACE_PID=$!
        echo "Start tracking sizes"
        /home/breidys2/check.sh $HDFS "${OUT}_${RUN}.sizing" &
        sudo blktrace -d /dev/$DEV --output-dir $TRACE_DIR &
        SIZE_PID=$!
        sleep 2
        for i in {1..6}; do
            hadoop fs -rm -r /user/breidys2/teraoutput
            hadoop fs -rm -r /user/breidys2/terainput
            hadoop jar $HADOOP_EXAMPLE_JAR teragen $IN_SZ /user/breidys2/terainput
            sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"
            echo "Running workload"
            (time hadoop jar $HADOOP_EXAMPLE_JAR terasort -D mapred.reduce.tasks=2 /user/breidys2/terainput /user/breidys2/teraoutput) 
            #hadoop fs -rm -r /user/breidys2/teraoutput
            #sudo sh -c "echo '3' >> /proc/sys/vm/drop_caches"
            #echo "Running workload"
            #(time hadoop jar $HADOOP_EXAMPLE_JAR terasort -D mapred.reduce.tasks=2 /user/breidys2/terainput /user/breidys2/teraoutput)  
        done
        echo "Getting new hdfs size"
        TEMP=$(du -s $HDFS)
        TOKENS=( $TEMP )
        END=${TOKENS[0]}
        echo "End size in KB ${END}" >> $OUT.resize
        echo "Size difference in KB $( expr $END - $START )" >> $OUT.resize
        echo "Killing blktrace"
        sudo pkill blktrace 
        sudo pkill check 
        sleep 2
        echo "Parsing traced file"
        pushd $TRACE_DIR
        blkparse $DEV -a complete -f "%D %2c %8s %5T.%9t %5p %2a %3d %N\n" -o "${OUT}_${RUN}.trace"
        #Uncomment if you want offsets in the trace
        #blkparse $DEV -a complete -o "${OUT}_offset".trace
        popd
        echo "Copying output files to runs"
        cp $TRACE_DIR/"${OUT}_${RUN}.trace" $RUNS_DIR
        #Uncomment if you want offsets in the trace
        #cp $TRACE_DIR/"${OUT}_offset.trace" $RUNS_DIR
        cp $OUT.terasort $RUNS_DIR
        cp $OUT.resize $RUNS_DIR
        cp "${OUT}_${RUN}.sizing" $RUNS_DIR
    done
elif [[ $TYPE == "teragen" ]]
then
    echo "Removing old input"
    hadoop fs -rm -r /user/breidys2/terainput
    echo "Grabbing hdfs Size"
    TEMP=$(du -s $HDFS) TOKENS=( $TEMP ) START=${TOKENS[0]}
    echo "Start size in KB ${START}" >> $OUT.resize
    echo "Generating new input"
    hadoop jar $HADOOP_EXAMPLE_JAR teragen $IN_SZ /user/breidys2/terainput
    echo "Getting new hdfs size"
    TEMP=$(du -s $HDFS)
    TOKENS=( $TEMP )
    END=${TOKENS[0]}
    echo "End size in KB ${END}" >> $OUT.resize
    echo "Size difference in KB $( expr $END - $START )" >> $OUT.resize
else
    echo "Please specify a valid workload" && exit
fi
echo "Done"
