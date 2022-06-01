#!/bin/bash

WLS=(pagerank terasort ml_prep)
WINDOWS=(180)

for ((j=0; j<${#WINDOWS[@]};j++)); do
    for FILE in "${WLS[@]}"; do
        echo "Running ${FILE}"
        for i in {1..5}; do
            python3 parse_duration.py "${FILE}_${i}.trace" "inputs/dur_bw/${FILE}_${i}_dur" "${WINDOWS[${j}]}" &
        done
        wait
    done
done
