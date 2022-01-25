#ifndef WAERMELEITUNG_H_
#define WAERMELEITUNG_H_

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// params for the specific problem
#define PARAM_ALPHA 1
#define PARAM_H 1
#define PARAM_T 0.25
#define PADDING_TEMPERATURE 0

// array access
#define global_index(x,y) y * size + x
#define chunk_index(x,y) y * (chunk_dimensions[X_AXIS] + 2 * g) + x

// neighbours
#define N_NEIGHBOURS 4
#define EAST 0
#define WEST 1
#define NORTH 2
#define SOUTH 3

// our problem is 2-dimensional
#define N_DIMENSIONS 2
#define X_AXIS 0
#define Y_AXIS 1

// rank for printing and stuff
#define MAIN_RANK 0

// coords [x, y] -> rank
int get_rank(int* coords, int* n_processes) {
    if (coords[X_AXIS] < 0) return -1;
    if (coords[Y_AXIS] < 0) return -1;
    if (coords[X_AXIS] >= n_processes[X_AXIS]) return -1;
    if (coords[Y_AXIS] >= n_processes[Y_AXIS]) return -1;
    return n_processes[X_AXIS] * coords[Y_AXIS] + coords[X_AXIS];
}

// rank -> coords [x, y]
void get_coords(int rank, int* n_processes, int* coords) {
    coords[X_AXIS] = rank % n_processes[X_AXIS];
    coords[Y_AXIS] = rank / n_processes[Y_AXIS];
}

void swap(double* pointer_0, double* pointer_1) {
	double* temp = pointer_0;
	pointer_0 = pointer_1;
	pointer_1 = temp;
}

#endif // WAERMELEITUNG_H_