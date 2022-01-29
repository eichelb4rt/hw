#include "waermeleitung.h"

//Ausgabe des Feldes t als PPM (Portable Pix Map) in filename
//mit schönen Farben
void printResult(double* t, int size, char* filename, int iter) {
    char buf[32];
    sprintf(buf, "%s-%i.ppm", filename, iter);
    FILE* f = fopen(buf, "w");
    fprintf(f, "P3\n%i %i\n255\n", size, size);
    double tmax = 25.0;
    double tmin = -tmax;
    double r, g, b;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double val = t[j + i * size];
            r = 0;
            g = 0;
            b = 0;
            if (val <= tmin) {
                b = 1.0 * 255.0;
            } else if (val >= -25.0 && val < -5) {
                b = 255 * 1.0;
                g = 255 * ((val + 25) / 20);
            } else if (val >= -5 && val <= 0.0) {
                g = 255 * 1.0;
                b = 255 * (1.0 - (val + 5) / 5);
            } else if (val > 0.0 && val <= 5) {
                g = 255 * 1.0;
                r = 255 * ((val) / 5);
            } else if (val > 5 && val < 25.0) {
                r = 255 * 1.0;
                g = 255 * ((25 - val) / 20);
            } else {
                r = 255 * 1.0;
            }
            fprintf(f, "%i\n%i\n%i\n", (int) r, (int) g, (int) b);
        }
        //      fprintf(f,"\n");
    }
    fclose(f);
}

void collect(double* global_grid, int size, double* local_chunk, int* chunk_dimensions, int* n_processes, int g, MPI_Datatype chunk_inner_values_t, MPI_Datatype chunk_in_global_array_t) {
    // first collect all the data from the chunks
    double* recv_buf = (double*) malloc(size * size * sizeof(double));
    MPI_Gather(&local_chunk[chunk_index(g, g)], 1, chunk_inner_values_t, recv_buf, chunk_dimensions[X_AXIS] * chunk_dimensions[Y_AXIS], MPI_DOUBLE, MAIN_RANK, MPI_COMM_WORLD);
    // then arrange them correctly
    int recv_buf_index, global_grid_index;
    // number of all the processes
    int total_ranks = n_processes[X_AXIS] * n_processes[Y_AXIS];
    // coords of the current rank
    int coords[N_DIMENSIONS];
    // offset in the global grid of the chunk that the current rank is processing
    int offset[N_DIMENSIONS];
    for (int rank = 0; rank < total_ranks; ++rank) {
        for (int chunk_y = 0; chunk_y < chunk_dimensions[Y_AXIS]; ++chunk_y) {
            for (int chunk_x = 0; chunk_x < chunk_dimensions[X_AXIS]; ++chunk_x) {
                // recv buf if now basically a 3-dimensional array with dimensions (rank, y, x)
                recv_buf_index = chunk_dimensions[X_AXIS] * (chunk_dimensions[Y_AXIS] * rank + chunk_y) + chunk_x;
                // index in the global array is defined by the coordinates that the chunk is written to
                get_coords(rank, n_processes, coords);
                // offset of the chunk in the global array
                offset[X_AXIS] = coords[X_AXIS] * chunk_dimensions[X_AXIS];
                offset[Y_AXIS] = coords[Y_AXIS] * chunk_dimensions[Y_AXIS];
                global_grid_index = global_index(offset[X_AXIS] + chunk_x, offset[Y_AXIS] + chunk_y);
                // now actually write it
                global_grid[global_grid_index] = recv_buf[recv_buf_index];
            }
        }
    }
}