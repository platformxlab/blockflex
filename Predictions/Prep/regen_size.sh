#!/bin/bash

WLS=(pagerank terasort ml_prep)
WINDOWS=(180)

for (( j=0; j<${#WINDOWS[@]}; j++)); do
    for FILE in "${WLS[@]}"; do
        echo "Running ${FILE}"
        for i in {1..5}; do
            python3 parse_sizing.py "${FILE}_${i}.sizing" "inputs/sz/${FILE}_${i}_sizing" "${WINDOWS[${j}]}" "inputs/dur_sz/${FILE}_${i}_dur" &
        done
        wait
    done
done
