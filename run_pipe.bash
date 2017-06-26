#!/bin/bash
declare -a datasets=("christine", "sylvine", "albert", "philippine", "madeline", "evita", "jasmine")
declare -a algorithms=("logistic", "svm", "xgboost")

N=1000

for d in ${datasets[@]};
do
    for a in ${algorithms[@]};
    do
        echo "Running with dataset $d, algorithm $a, $N configurations."
        python src/run_pipe.py "$a" "$d" $N
    done
done