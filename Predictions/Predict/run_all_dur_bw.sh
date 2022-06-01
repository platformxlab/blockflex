#!/bin/bash

WLS=(terasort ml_prep pagerank)
CP_DIR=../Prep/inputs/dur_bw/
IN=./inputs/dur_bw/
#These values are not used, only for clarity, testing buffers
WINDOWS=("1" "1.05" "1.1" "1.2" "1.3" "1.4")
OUT=./outputs/dur_bw/
COPY=1
PRED=1
ACC=1

if [ $COPY -gt 0 ]
then
    #First we copy everything over
    echo "Copying files first"
    for (( j=0; j < ${#WINDOWS[@]}; j++ )); do
        for FILE in "${WLS[@]}"; do
            for i in {1..5}; do
                echo "Copying over ${FILE} ${i}"
                echo $IN
                cp "${CP_DIR}${FILE}_${i}_dur" $IN
            done
        done
    done
fi

if [ $PRED -gt 0 ]
then
    echo "Running predictions"
    #Now we use this
    for (( j=0; j < ${#WINDOWS[@]}; j++ )); do
        cnt=0
        for FILE in "${WLS[@]}"; do
            for i in {1..5}; do
                for k in {1..5}; do
                    cnt=$((cnt+1))
                    echo "Running ${FILE} ${i}"
                    python3 lstm_dur.py "${IN}/${FILE}_${i}_dur" $j > "${OUT}/${FILE}_${j}_${cnt}.out"
                done
            done
            wait
        done
    done
    for (( j=0; j < ${#WINDOWS[@]}; j++ )); do
        cnt=0
        mult=0
        for file in ${IN}/c_*.txt; do
            for i in {1..5}; do
                echo $file
                python3 lstm_dur_ali.py $file $j > "${OUT}/ali_${j}_${mult}.out" &
                mult=$((mult+1))
            done
            wait
            cnt=$((cnt+1))
            if [ $cnt == 5 ]
            then
                break
            fi
        done
        wait
    done
fi

if [ $ACC -gt 0 ]
then
    #Now we parse and output some results
    echo "Parsing accuracy output"
    python3 acc_parser.py outputs/dur_bw outputs/dur_bw_acc_out.txt 

    echo "Outputting Accuracy Results"
    cat outputs/dur_bw_acc_out.txt
fi
#Done
echo "Done"
