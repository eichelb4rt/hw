#!/bin/bash

px=12
py=12
iterations=1000
# param_g = $1
# field_size = $2

mpirun ./build/waermeleitung -g "$1" "$2" "$iterations" "$px" "$py" --benchmark