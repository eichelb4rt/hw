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

void collect(double* global_grid, double* local_chunk, int* n_processes, MPI_Datatype chunk_inner_values_t, MPI_Datatype chunk_in_global_array_t) {
    MPI_Gather(local_chunk, 1, chunk_inner_values_t, global_grid, n_processes[X_AXIS] * n_processes[Y_AXIS], chunk_in_global_array_t, MAIN_RANK, MPI_COMM_WORLD);
}