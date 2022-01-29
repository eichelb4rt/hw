#ifndef WAERMELEITUNG_H_
#define WAERMELEITUNG_H_

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

// params for the specific problem
#define PARAM_ALPHA 1
#define PARAM_H 1
#define PARAM_T 0.25
#define PADDING_TEMPERATURE 0

// special command-line-parameter values
#define DONT_PRINT 0

// array access
#define global_index(x,y) (y) * size + (x)
#define chunk_index(x,y) (y) * (chunk_dimensions[X_AXIS] + 2 * g) + (x)

// our problem is 2-dimensional
#define N_DIMENSIONS 2
#define X_AXIS 0
#define Y_AXIS 1

// rank for printing and stuff
#define MAIN_RANK 0
#define UNDEFINED_RANK -1

// neighbours
#define N_NEIGHBOURS 4
#define EAST 0
#define WEST 1
#define NORTH 2
#define SOUTH 3

// which directions are the directions actually?
int diff_directions[N_NEIGHBOURS][N_DIMENSIONS] = {
    // EAST (X,Y)
    { 1, 0 },
    // WEST (X,Y)
    { -1, 0 },
    // NORTH (X,Y)
    { 0, -1 },
    // SOUTH (X,Y)
    { 0, 1 },
};

// coords [x, y] -> rank
int get_rank(int* coords, int* n_processes) {
    if (coords[X_AXIS] < 0) return UNDEFINED_RANK;
    if (coords[Y_AXIS] < 0) return UNDEFINED_RANK;
    if (coords[X_AXIS] >= n_processes[X_AXIS]) return UNDEFINED_RANK;
    if (coords[Y_AXIS] >= n_processes[Y_AXIS]) return UNDEFINED_RANK;
    return coords[Y_AXIS] * n_processes[X_AXIS] + coords[X_AXIS];
}

// rank -> coords [x, y]
void get_coords(int rank, int* n_processes, int* coords) {
    coords[X_AXIS] = rank % n_processes[X_AXIS];
    coords[Y_AXIS] = rank / n_processes[X_AXIS];
}

void swap(double** pointer_0, double** pointer_1) {
    double* temp = *pointer_0;
    *pointer_0 = *pointer_1;
    *pointer_1 = temp;
}

#endif // WAERMELEITUNG_H_
