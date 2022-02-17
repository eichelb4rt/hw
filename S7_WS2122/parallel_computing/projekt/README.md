# Waermeleitung

## Disclaimer

Die getesteten Werte sind vom lokalen Rechner. Beim ARA-Cluster kam es aus mir nicht ersichtlichen Gründen bei MPI_Finalize() (nicht vorher) zu Segmentation faults. Lokal funktioniert alles.

## Usage

```plaintext
usage: stencil_mpi [options] <field_size> <iterations> <px> <py>
options:
    --out, -o <argument>: output file
    --ghost, g <argument>: width of halo
    --print, -p <argument>: print every <argument> iterations
    --benchmark, -b: enable benchmark, disable printing
examples:
mpirun -np 4 ./build/waermeleitung 128 10 2 2
mpirun -np 8 ./build/waermeleitung -o "out/output" -g 3 -p 0 1024 1000 4 2
```

Set the print distance (`--print`) to 0 to only print the last state.
