#!/bin/bash

#WL=(pagerank ml_prep terasort)
WL=(ml_prep)

make

for FILE in "${WL[@]}"; do
    echo "Running ${FILE}_harvest"
    sudo ./harvest "${FILE}_offset.trace" 1 > "${FILE}_harvest.out"
done
echo "Done with the harvested sections, now running the no harvesting parts"
for FILE in "${WL[@]}"; do
    echo "Running ${FILE}_no_harvest"
    sudo ./harvest "${FILE}_offset.trace" 0 > "${FILE}_no_harvest.out"
done
#sudo ./harvest ml_prep_offset.trace 0 > ml_prep_no_harvest.out
#sudo ./harvest pagerank_offset.trace 0 > pagerank_no_harvest.out
#sudo ./harvest pagerank_offset.trace 1 > pagerank_harvest.out
#sudo ./harvest ml_prep_offset.trace 1 > ml_prep_harvest.out
