#include "waermeleitung.h"

void send_ghosts(int direction, int* neighbours, int* send_buffer_start, double* grid, MPI_Datatype border_type, int* array_of_requests, int* current_request) {
    if (neighbours[direction] == UNDEFINED_RANK) return;
    // send from non-ghost-zone-end - g = chunk_width + g - g
    int TAG = *current_request;
    MPI_Isend(&grid[send_buffer_start[direction]], 1, border_type, neighbours[direction], TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request]);
    // update current index because communication was succesful
    ++(*current_request);
}

void recv_ghosts(int direction, int* neighbours, int* recv_buffer_start, double* grid, MPI_Datatype border_type, int* array_of_requests, int* current_request) {
    if (neighbours[direction] == UNDEFINED_RANK) return;
    // receive in ghost-zone
    MPI_Irecv(&grid[recv_buffer_start[direction]], 1, border_type, neighbours[direction], MPI_ANY_TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request]);
    // update current index because communication was succesful
    ++(*current_request);
}