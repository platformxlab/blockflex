#!/bin/bash

WLS=(terasort ml_prep pagerank)
CP_DIR=../Prep/inputs/sz/
IN=./inputs/sz/
#These values are not used, only for clarity, testing buffers
WINDOWS=("1" "1.05" "1.1" "1.2" "1.3" "1.4")
OUT=./outputs/sz/
COPY=1
PRED=1
ACC=1

if [ $COPY -gt 0 ]
then
    #First we copy everything over
    echo "Copying files first"
    for ((j=0;j<${#WINDOWS[@]};j++)); do
        for FILE in "${WLS[@]}"; do
            for i in {1..5}; do
                echo "Copying over ${FILE} ${i}"
                echo $IN
                cp "${CP_DIR}${FILE}_${i}_sizing" $IN
            done
        done
    done
fi

if [ $PRED -gt 0 ]
then
    echo "Running predictions"
    for ((j=0;j<${#WINDOWS[@]};j++)); do
        cnt=0
        for FILE in "${WLS[@]}"; do
            for i in {1..5}; do
                #Running each 5 times
                for k in {1..5}; do
                    echo "Running ${FILE} ${i}"
                    cnt=$((cnt+1))
                    python3 lstm_sz.py "${IN}/${FILE}_${i}_sizing" $j > "${OUT}/${FILE}_${j}_${cnt}.out" &
                done
            done
            wait
        done
    done
    for ((j=0;j<${#WINDOWS[@]};j++)); do
        cnt=0
        mult=0
        for file in ${IN}/*.txt; do
            for i in {1..5}; do
                echo $file
                python3 lstm_sz_gl.py $file $j > "${OUT}/google_${j}_${mult}.out" &
                mult=$((mult+1))
            done
            cnt=$((cnt+1))
            wait
            if [ $cnt == 5 ] 
            then
                break
            fi
        done
    done
fi

if [ $ACC -gt 0 ]
then
    #Now we parse and output some results
    echo "Parsing accuracy output"
    python3 acc_parser.py outputs/sz outputs/sz_acc_out.txt 

    echo "Outputting Accuracy Results"
    cat outputs/sz_acc_out.txt
fi
#Done
echo "Done"
