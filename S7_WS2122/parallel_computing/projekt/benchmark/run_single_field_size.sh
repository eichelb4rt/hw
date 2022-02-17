#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "$SCRIPT_DIR/.." || exit 1

# we're in the root dir now
RESULTS_BASE_DIR="./benchmark/results/"
mkdir -p "$RESULTS_BASE_DIR"

FIELD_SIZE_LIST=(1024)
PARAM_G_LIST=(48 64 96 128)

for field_size in "${FIELD_SIZE_LIST[@]}"; do
    for param_g in "${PARAM_G_LIST[@]}"; do
        CURRENT_PARAM_DIR="$RESULTS_BASE_DIR/size_$field_size/g_$param_g"
        mkdir -p "$CURRENT_PARAM_DIR"
        for run in {0..9}; do
            mpirun -np 8 ./build/waermeleitung -g "$param_g" "$field_size" 1000 4 2 --benchmark > "$CURRENT_PARAM_DIR/run_$run.txt"
        done
        echo "field_size=$field_size, g=$param_g done."
    done
done