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

void setup(int rank, int num, int argc, char** argv, int* field_length, int* iterations, int* print_distance, int* n_ghost_blocks, int* n_processes, char** output_filename, int* finalize) {
    *finalize = 0;
    if (argc < 6) {
        if (rank == MAIN_RANK)
            printf("usage: stencil_mpi <field_size> <iterations> <print_distance> <ghost_blocks> <px> <py> <out_file>\n");
        (*finalize) = 1;
        return;
    }

    // read paramaters
    // TODO: better UI?
    int arg_counter = 1;
    (*field_length) = atoi(argv[arg_counter++]);      /* nxn grid */
    (*iterations) = atoi(argv[arg_counter++]);
    (*print_distance) = atoi(argv[arg_counter++]);
    (*n_ghost_blocks) = atoi(argv[arg_counter++]); /* number of iterations */
    n_processes[X_AXIS] = atoi(argv[arg_counter++]);     /* 1st dim processes */
    n_processes[Y_AXIS] = atoi(argv[arg_counter++]);     /* 2nd dim processes */
    *output_filename = argv[arg_counter++];

    if (num != n_processes[X_AXIS] * n_processes[Y_AXIS]) {
        if (rank == MAIN_RANK)
            printf("Process dimensions (%d x %d) do not match number of processes (%d).\n", n_processes[X_AXIS], n_processes[Y_AXIS], num);
        (*finalize) = 1;
        return;
    }
    int rect_field_size = (*field_length) * (*field_length);
    if (rect_field_size % n_processes[X_AXIS] != 0) {
        if (rank == MAIN_RANK)
            printf("Number of processes in x dimension (%d) does not work with field length (%d).\n", n_processes[X_AXIS], *field_length);
        (*finalize) = 1;
        return;
    }
    if (rect_field_size % n_processes[Y_AXIS] != 0) {
        if (rank == MAIN_RANK)
            printf("Number of processes in y dimension (%d) does not work with field length (%d).\n", n_processes[Y_AXIS], *field_length);
        (*finalize) = 1;
        return;
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
    for (int direction = 0; direction < N_NEIGHBOURS; ++direction) {
        coords_neighbours[direction][X_AXIS] = coords_self[X_AXIS] + diff_directions[direction][X_AXIS];
        coords_neighbours[direction][Y_AXIS] = coords_self[Y_AXIS] + diff_directions[direction][Y_AXIS];
        neighbours[direction] = get_rank(coords_neighbours[direction], n_processes);
    }
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
            l_chunk[chunk_index(x + g, y + g)] = global_field[global_index(offset_x + x, offset_y + y)];
        }
    }
}

void get_printed_iterations(int iter, int n_printed_iterations, int* printed_iterations, int* is_printed) {
    // init is_printed with 0
    for (int i = 0; i < iter; ++i) {
        is_printed[i] = 0;
    }
    // fill is_printed with 1s for every iteration that is printed
    int print_last_iteration = 0;
    for (int i = 0; i < n_printed_iterations; ++i) {
        int printed_iteration = printed_iterations[i];
        // ignore printed iterations that are not among the actual iterations
        if (printed_iteration < 0 || printed_iteration >= iter) {
            continue;
        }
        is_printed[printed_iteration] = 1;
    }
}