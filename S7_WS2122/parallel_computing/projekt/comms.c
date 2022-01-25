#include "waermeleitung.h"

void set_comm_indices(int* send_indices, int* recv_indices, int* chunk_dimensions, int g) {
    send_indices[EAST] = chunk_index(chunk_dimensions[X_AXIS], g);
    send_indices[WEST] = chunk_index(g, g);
    send_indices[NORTH] = chunk_index(0, g);
    send_indices[SOUTH] = chunk_index(0, chunk_dimensions[Y_AXIS]);

    recv_indices[EAST] = chunk_index(chunk_dimensions[X_AXIS] + g, g);
    recv_indices[WEST] = chunk_index(0, g);
    recv_indices[NORTH] = chunk_index(0, 0);
    recv_indices[SOUTH] = chunk_index(0, chunk_dimensions[Y_AXIS] + g);
}

void send_ghosts(int direction, int* neighbours, int* send_indices, double* grid, MPI_Datatype border_type, int* array_of_requests, int* current_request) {
    int TAG = *current_request;
    if (neighbours[direction] == UNDEFINED_RANK) return;
    // send from non-ghost-zone-end - g = chunk_width + g - g
    MPI_Isend(&grid[send_indices[direction]], 1, border_type, neighbours[direction], TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request]);
    // update current index because communication was succesful
    ++(*current_request);
}

void recv_ghosts(int direction, int* neighbours, int* recv_indices, double* grid, MPI_Datatype border_type, int* array_of_requests, int* current_request) {
    if (neighbours[direction] == UNDEFINED_RANK) return;
    // receive in ghost-zone
    MPI_Irecv(&grid[recv_indices[direction]], 1, border_type, neighbours[direction], MPI_ANY_TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request]);
    // update current index because communication was succesful
    ++(*current_request);
}