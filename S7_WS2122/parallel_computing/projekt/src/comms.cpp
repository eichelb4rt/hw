#include "waermeleitung.h"

void get_vector_properties(int direction, int* chunk_dimensions, int g, int& block_count, int& block_length, int& block_stride) {
    if (direction == EAST || direction == WEST) {
        block_count = chunk_dimensions[Y_AXIS];
        block_length = g;
        block_stride = chunk_dimensions[X_AXIS] + 2 * g;
        return;
    }
    if (direction == NORTH || direction == SOUTH) {
        block_count = g;
        block_length = chunk_dimensions[X_AXIS] + 2 * g;
        block_stride = chunk_dimensions[X_AXIS] + 2 * g;
        return;
    }
}

// start indices for ghost block buffers
void set_comm_indices(int* send_buffer_start, int* recv_buffer_start, int* chunk_dimensions, int g) {
    send_buffer_start[EAST] = chunk_index(chunk_dimensions[X_AXIS], g);
    send_buffer_start[WEST] = chunk_index(g, g);
    send_buffer_start[NORTH] = chunk_index(0, g);
    send_buffer_start[SOUTH] = chunk_index(0, chunk_dimensions[Y_AXIS]);

    recv_buffer_start[EAST] = chunk_index(chunk_dimensions[X_AXIS] + g, g);
    recv_buffer_start[WEST] = chunk_index(0, g);
    recv_buffer_start[NORTH] = chunk_index(0, 0);
    recv_buffer_start[SOUTH] = chunk_index(0, chunk_dimensions[Y_AXIS] + g);
}

void pad(int direction, int* recv_buffer_start, double* grid, int* chunk_dimensions, int g) {
    // basically build the properties of the vector types again
    int block_stride, block_count, block_length;
    get_vector_properties(direction, chunk_dimensions, g, block_count, block_length, block_stride);
    // find out where the ghost blocks start
    int chunk_start_x = recv_buffer_start[direction] % (chunk_dimensions[X_AXIS] + 2 * g);
    int chunk_start_y = recv_buffer_start[direction] / (chunk_dimensions[X_AXIS] + 2 * g);
    // current coordinates in the chunk
    int chunk_x, chunk_y;
    // pixel that we take our vale from (reference)
    int ref_x, ref_y;
    for (int block_y = 0; block_y < block_count; ++block_y) {
        for (int block_x = 0; block_x < block_length; ++block_x) {
            chunk_x = chunk_start_x + block_x;
            chunk_y = chunk_start_y + block_y;
            ref_x = chunk_x;
            ref_y = chunk_y;
            // trim to inner field -> ref is going to be the nearest pixel in the field
            if (ref_x > chunk_dimensions[X_AXIS] + g - 1) ref_x = chunk_dimensions[X_AXIS] + g - 1;
            if (ref_x < g) ref_x = g;
            if (ref_y > chunk_dimensions[Y_AXIS] + g - 1) ref_y = chunk_dimensions[Y_AXIS] + g - 1;
            if (ref_y < g) ref_y = g;
            // now pad the ghost block with the nearest value in the field
            grid[chunk_index(chunk_x, chunk_y)] = grid[chunk_index(ref_x, ref_y)];
        }
    }
}

void send_ghosts(int direction, int* neighbours, int* send_buffer_start, double* grid, MPI_Datatype border_type, MPI_Request* array_of_requests, int& current_request, int tag) {
    // do nothing if the neighbour does not exist
    if (neighbours[direction] == UNDEFINED_RANK) return;
    // send from non-ghost-zone-end
    MPI_Isend(&grid[send_buffer_start[direction]], 1, border_type, neighbours[direction], tag,
        MPI_COMM_WORLD, &array_of_requests[current_request]);
    // update current index because communication was succesful
    ++current_request;
}

void recv_ghosts(int direction, int* neighbours, int* recv_buffer_start, double* grid, MPI_Datatype border_type, MPI_Request* array_of_requests, int& current_request, int* chunk_dimensions, int g) {
    // pad if the neighbour does not exist
    if (neighbours[direction] == UNDEFINED_RANK) {
        pad(direction, recv_buffer_start, grid, chunk_dimensions, g);
        return;
    }
    // receive in ghost-zone
    MPI_Irecv(&grid[recv_buffer_start[direction]], 1, border_type, neighbours[direction], MPI_ANY_TAG,
        MPI_COMM_WORLD, &array_of_requests[current_request]);
    // update current index because communication was succesful
    ++current_request;
}