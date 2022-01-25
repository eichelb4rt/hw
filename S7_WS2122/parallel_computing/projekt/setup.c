#include "waermeleitung.h"

//Initialisiere Wärmefeld mit Startwerten:
//innen: 0.0
//Rand: 
//links/oben warm=25.0
//rechts/unten kalt=-25.0
void init(double* t, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            t[j + i * size] = 0.0;
            if (j == 0) t[i * size] = 25.0;
            if (j == size - 1) t[j + i * size] = -25.0;
            if (i == 0) t[j + i * size] = 25.0;
            if (i == size - 1) t[j + i * size] = -25.0;
        }
    }
}

void setup(int rank, int num, int argc, char** argv, int* field_length, int* iterations, int* n_ghost_blocks, int* n_processes, char* output_filename, int* finalize) {
    *finalize = 0;
    if (argc < 6) {
        if (rank == MAIN_RANK)
            printf("usage: stencil_mpi <field_size> <iterations> <ghost_blocks> <px> <py> <out_file>\n");
        (*finalize) = 1;
        return;
    }

    // read paramaters
    // TODO: better UI?
    (*field_length) = atoi(argv[1]);      /* nxn grid */
    (*iterations) = atoi(argv[2]);
    (*n_ghost_blocks) = atoi(argv[3]); /* number of iterations */
    n_processes[X_AXIS] = atoi(argv[4]);     /* 1st dim processes */
    n_processes[Y_AXIS] = atoi(argv[5]);     /* 2nd dim processes */
    output_filename = argv[6];

    if (num != n_processes[X_AXIS] * n_processes[Y_AXIS]) {
        MPI_Abort(MPI_COMM_WORLD, 1);  /* abort if px or py are wrong */
    }
    int rect_field_size = (*field_length) * (*field_length);
    if (rect_field_size % n_processes[X_AXIS] != 0) {
        MPI_Abort(MPI_COMM_WORLD, 2);  /* abort px needs to divide n */
    }
    if (rect_field_size % n_processes[Y_AXIS] != 0) {
        MPI_Abort(MPI_COMM_WORLD, 3);  /* abort py needs to divide n */
    }
}

void split_up_domain(int rank, int size, int* n_processes, int* dimensions, int* neighbours) {
    // split up domain between processes
    dimensions[X_AXIS] = size / n_processes[X_AXIS];
    dimensions[Y_AXIS] = size / n_processes[Y_AXIS];

    // where are the neighbours
    int coords_self[N_DIMENSIONS];
    int coords_neighbours[N_NEIGHBOURS][N_DIMENSIONS];

    get_coords(rank, n_processes, coords_self);
    coords_neighbours[EAST][X_AXIS] = coords_self[X_AXIS] - 1;
    coords_neighbours[EAST][Y_AXIS] = coords_self[Y_AXIS];
    coords_neighbours[WEST][X_AXIS] = coords_self[X_AXIS] + 1;
    coords_neighbours[WEST][Y_AXIS] = coords_self[Y_AXIS];
    coords_neighbours[NORTH][X_AXIS] = coords_self[X_AXIS];
    coords_neighbours[NORTH][Y_AXIS] = coords_self[Y_AXIS] - 1;
    coords_neighbours[SOUTH][X_AXIS] = coords_self[X_AXIS];
    coords_neighbours[SOUTH][Y_AXIS] = coords_self[Y_AXIS] + 1;

    neighbours[EAST] = get_rank(coords_neighbours[EAST], n_processes);
    neighbours[WEST] = get_rank(coords_neighbours[WEST], n_processes);
    neighbours[NORTH] = get_rank(coords_neighbours[NORTH], n_processes);
    neighbours[SOUTH] = get_rank(coords_neighbours[SOUTH], n_processes);
}

void fill_local_chunk(int rank, int* n_processes, double* l_chunk, int* chunk_dimensions, double* global_field, int size, int g) {
    // distribute the workload
    int coords[N_DIMENSIONS];
    get_coords(rank, n_processes, coords);
    int offset_x = coords[X_AXIS] * chunk_dimensions[X_AXIS];
    int offset_y = coords[Y_AXIS] * chunk_dimensions[Y_AXIS];
    int global_x;
    int global_y;
    // initialise the whole chunk
    for (int y = 0; y < chunk_dimensions[Y_AXIS] + 2 * g; ++y) {
        for (int x = 0; x < chunk_dimensions[X_AXIS] + 2 * g; ++x) {
            l_chunk[chunk_index(x, y)] = PADDING_TEMPERATURE;
        }
    }
    // initialise inner chunk with respective values in the global field
    for (int y = 0; y < chunk_dimensions[Y_AXIS]; ++y) {
        for (int x = 0; x < chunk_dimensions[X_AXIS]; ++x) {
            global_x = offset_x + x;
            global_y = offset_y + y;
            l_chunk[chunk_index(g + x, g + y)] = global_field[global_index(global_x, global_y)];
        }
    }
}