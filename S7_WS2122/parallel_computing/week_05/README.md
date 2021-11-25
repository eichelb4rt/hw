# MPI Matrix Multiplikation

```bash
make
mpirun -np 4 ./mpi_matrices_blocking 10 10 10
mpirun -np 4 ./mpi_matrices_serial 10 10 10
mpirun -np 4 ./mpi_matrices_non_blocking 10 10 10
```

Ergebnisse: Hab absolut nicht wirklich Lust das zu plotten. Serial ist komplett schnell. Non-Blocking ist schneller als Blocking.
