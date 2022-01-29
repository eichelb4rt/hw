# Waermeleitung

```bash
usage: stencil_mpi [options] <field_size> <iterations> <px> <py>
options:
    --out, -o <argument>: output file
    --ghost, g <argument>: width of halo
    --print, -p <argument>: print every <argument> iterations
examples:
mpirun -np 4 ./waermeleitung 128 10 2 2
mpirun -np 8 ./waermeleitung -o "out/output" -g 3 -p 0 128 10 4 2
```

Set the print distance (`--print`) to 0 to disable printing states except for the last one.
