#include "waermeleitung.h"

void send_east(int east_neighbour_rank, double* grid, int* chunk_dimensions, int g, MPI_Datatype vertical_border_t, int* array_of_requests, int* current_request_index) {
    int TAG = *current_request_index;
    if (east_neighbour_rank == -1) return;
    // send from non-ghost-zone-end - g = chunk_width + g - g
    MPI_Isend(&grid[chunk_index(chunk_dimensions[X_AXIS], g)], 1, vertical_border_t, east_neighbour_rank, TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request_index]);
    // update current index because communication was succesful
    ++(*current_request_index);
}

void recv_east(int east_neighbour_rank, double* grid, int* chunk_dimensions, int g, MPI_Datatype vertical_border_t, int* array_of_requests, int* current_request_index) {
    if (east_neighbour_rank == -1) return;
    // receive in ghost-zone
    MPI_Irecv(&grid[chunk_index(chunk_dimensions[X_AXIS] + g, g)], 1, vertical_border_t, east_neighbour_rank, MPI_ANY_TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request_index]);
    // update current index because communication was succesful
    ++(*current_request_index);
}

void send_west(int west_neighbour_rank, double* grid, int* chunk_dimensions, int g, MPI_Datatype vertical_border_t, int* array_of_requests, int* current_request_index) {
    int TAG = *current_request_index;
    if (west_neighbour_rank == -1) return;
    // send from non-ghost-zone
    MPI_Isend(&grid[chunk_index(g, g)], 1, vertical_border_t, west_neighbour_rank, TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request_index]);
    // update current index because communication was succesful
    ++(*current_request_index);
}

void recv_west(int west_neighbour_rank, double* grid, int* chunk_dimensions, int g, MPI_Datatype vertical_border_t, int* array_of_requests, int* current_request_index) {
    if (west_neighbour_rank == -1) return;
    // receive in ghost-zone
    MPI_Irecv(&grid[chunk_index(0, g)], 1, vertical_border_t, west_neighbour_rank, MPI_ANY_TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request_index]);
    // update current index because communication was succesful
    ++(*current_request_index);
}

void send_north(int north_neighbour_rank, double* grid, int* chunk_dimensions, int g, MPI_Datatype horizontal_border_t, int* array_of_requests, int* current_request_index) {
    int TAG = *current_request_index;
    if (north_neighbour_rank == -1) return;
    // send from non-ghost-zone
    MPI_Isend(&grid[chunk_index(0, g)], 1, horizontal_border_t, north_neighbour_rank, TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request_index]);
    // update current index because communication was succesful
    ++(*current_request_index);
}

void recv_north(int north_neighbour_rank, double* grid, int* chunk_dimensions, int g, MPI_Datatype horizontal_border_t, int* array_of_requests, int* current_request_index) {
    if (north_neighbour_rank == -1) return;
    // receive in ghost-zone
    MPI_Irecv(&grid[chunk_index(0, 0)], 1, horizontal_border_t, north_neighbour_rank, MPI_ANY_TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request_index]);
    // update current index because communication was succesful
    ++(*current_request_index);
}

void send_south(int south_neighbour_rank, double* grid, int* chunk_dimensions, int g, MPI_Datatype horizontal_border_t, int* array_of_requests, int* current_request_index) {
    int TAG = *current_request_index;
    if (south_neighbour_rank == -1) return;
    // send from non-ghost-zone
    MPI_Isend(&grid[chunk_index(0, chunk_dimensions[Y_AXIS])], 1, horizontal_border_t, south_neighbour_rank, TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request_index]);
    // update current index because communication was succesful
    ++(*current_request_index);
}

void recv_south(int south_neighbour_rank, double* grid, int* chunk_dimensions, int g, MPI_Datatype horizontal_border_t, int* array_of_requests, int* current_request_index) {
    if (south_neighbour_rank == -1) return;
    // receive in ghost-zone
    MPI_Irecv(&grid[chunk_index(0, chunk_dimensions[Y_AXIS] + g)], 1, horizontal_border_t, south_neighbour_rank, MPI_ANY_TAG,
        MPI_COMM_WORLD, &array_of_requests[*current_request_index]);
    // update current index because communication was succesful
    ++(*current_request_index);
}