#!/bin/bash
JOB_NAME="pc1-project-mw"
NODES=4
PROC=36
TIME=1:00:00
PARTITION="s_hadoop"

sbatch --job-name="$JOB_NAME" --partition="$PARTITION" --nodes="$NODES" --ntasks-per-node="$PROC" --time="$TIME" ./benchmark.sh
