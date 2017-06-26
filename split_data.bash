#!/bin/bash
declare -a datasets=("christine", "sylvine", "albert", "philippine", "madeline", "evita", "jasmine")

for i in ${datasets[@]};
do
    echo "Splitting $i..."
    python split_data.py "$i"
done
