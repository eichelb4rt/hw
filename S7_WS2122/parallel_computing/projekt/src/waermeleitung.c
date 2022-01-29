#include "waermeleitung.h"
#include "comms.c"
#include "setup.c"
#include "print.c"

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	//Größe des Feldes
	int size = 128;
	//Anzahl Iterationen
	int iter = 100;
	//Geisterzonenbreite
	int g = 1;
	//Ausgabedatei
	char* filename = "output";
	// printed iterations
	int print_distance = 5;

	int num, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &num);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int n_processes[N_DIMENSIONS];
	int finalize;
	setup(rank, num, argc, argv, &size, &iter, &print_distance, &g, n_processes, &filename, &finalize);
	if (finalize == 1) {
		MPI_Finalize();
		exit(0);
	}

	//2 Speicherbereiche für das Wärmefeld
	double* u1, * u2;

	//Größe des Speicherbereiches
	int mem = size * size * sizeof(double);
	//Allokiere Speicher auf Host
	u1 = (double*) malloc(mem);
	u2 = (double*) malloc(mem);
	//Initialisiere Speicher
	init(u1, size);

	//TODO: Implementieren Sie ein paralleles Programm, 
	//      das die Temperaturen u_k mit einer beliebigen 
	//      Anzahl von p Prozessoren unter Verwendung von MPI berechnet.
	//      Der Austausch der Randbereiche soll alle g Schritte passieren.

	// split up domain
	int* chunk_dimensions = (int*) malloc(N_DIMENSIONS * sizeof(int));
	int* neighbours = (int*) malloc(N_NEIGHBOURS * sizeof(int));
	split_up_domain(rank, size, n_processes, chunk_dimensions, neighbours);

	// make the types for communication
	int block_stride, block_count, block_length;
	// exchange all known data across the vertical line, g wide. (stride is the whole horizontal dimension)
	MPI_Datatype vertical_border_t;
	get_vector_properties(EAST, chunk_dimensions, g, &block_count, &block_length, &block_stride);
	MPI_Type_vector(block_count, block_length, block_stride, MPI_DOUBLE, &vertical_border_t);
	MPI_Type_commit(&vertical_border_t);
	// exchange all data across the horizontal line, whole horizontal line wide, g blocks. (stride is the whole horizontal dimension)
	MPI_Datatype horizontal_border_t;
	get_vector_properties(NORTH, chunk_dimensions, g, &block_count, &block_length, &block_stride);
	MPI_Type_vector(block_count, block_length, block_stride, MPI_DOUBLE, &horizontal_border_t);
	MPI_Type_commit(&horizontal_border_t);
	// type for all the inner values (non-ghost-blocks)
	MPI_Datatype chunk_inner_values_t;
	MPI_Type_vector(chunk_dimensions[Y_AXIS], chunk_dimensions[X_AXIS], chunk_dimensions[X_AXIS] + 2 * g, MPI_DOUBLE, &chunk_inner_values_t);
	MPI_Type_commit(&chunk_inner_values_t);

	// local chunk with ghost blocks
	double* l_chunk = (double*) malloc((chunk_dimensions[X_AXIS] + 2 * g) * (chunk_dimensions[Y_AXIS] + 2 * g) * sizeof(double));
	// buffer for local chunk
	double* l_chunk_buf = (double*) malloc((chunk_dimensions[X_AXIS] + 2 * g) * (chunk_dimensions[Y_AXIS] + 2 * g) * sizeof(double));
	// fill the local chunk
	fill_local_chunk(rank, n_processes, l_chunk, chunk_dimensions, u1, size, g);

	// TODO (actually):
	// [x] update grid
	// [x] exchanges
	// rank 0 distributing and collecting?
	// [x]? make asynchronous

	// we divide all the requests into requests between east and west, and requests between north and south (because we want to send the corners correctly)
	int n_requests_ew = (neighbours[EAST] != UNDEFINED_RANK) + (neighbours[WEST] != UNDEFINED_RANK);
	int n_requests_ns = (neighbours[NORTH] != UNDEFINED_RANK) + (neighbours[SOUTH] != UNDEFINED_RANK);
	MPI_Request* array_of_requests_ew = (MPI_Request*) malloc(n_requests_ew * sizeof(MPI_Request));
	MPI_Request* array_of_requests_ns = (MPI_Request*) malloc(n_requests_ns * sizeof(MPI_Request));
	MPI_Status* array_of_status_ew = (MPI_Status*) malloc(n_requests_ew * sizeof(MPI_Status));
	MPI_Status* array_of_status_ns = (MPI_Status*) malloc(n_requests_ns * sizeof(MPI_Status));

	// define where communication is written from and to
	int send_buffer_start[N_NEIGHBOURS];
	int recv_buffer_start[N_NEIGHBOURS];
	set_comm_indices(send_buffer_start, recv_buffer_start, chunk_dimensions, g);
	// request index that's dynamically adapted by the comm calls
	int current_request = 0;

	// border that narrows with the number of iterations that we have not communicated
	int border;
	// factor for thermal calculation stuff
	const double FACTOR = PARAM_ALPHA * PARAM_T / (PARAM_H * PARAM_H);
	// main loop
	for (int i = 0; i < iter; ++i) {
		// maybe print?
		if (print_distance != DONT_PRINT && i % print_distance == 0) {
			collect(u1, size, l_chunk, chunk_dimensions, n_processes, g, chunk_inner_values_t);
			if (rank == MAIN_RANK) printResult(u1, size, filename, i);
		}
		// only communicate every g iterations
		if (i % g == 0) {
			// communication between east and west
			current_request = 0;
			recv_ghosts(EAST, neighbours, recv_buffer_start, l_chunk, vertical_border_t, array_of_requests_ew, &current_request, chunk_dimensions, g);
			recv_ghosts(WEST, neighbours, recv_buffer_start, l_chunk, vertical_border_t, array_of_requests_ew, &current_request, chunk_dimensions, g);
			send_ghosts(EAST, neighbours, send_buffer_start, l_chunk, vertical_border_t, array_of_requests_ew, &current_request, ((rank * num + i) * N_DIMENSIONS) + EAST);
			send_ghosts(WEST, neighbours, send_buffer_start, l_chunk, vertical_border_t, array_of_requests_ew, &current_request, ((rank * num + i) * N_DIMENSIONS) + WEST);
			MPI_Waitall(n_requests_ew, array_of_requests_ew, array_of_status_ew);
			// communication between north and south
			current_request = 0;
			recv_ghosts(NORTH, neighbours, recv_buffer_start, l_chunk, horizontal_border_t, array_of_requests_ns, &current_request, chunk_dimensions, g);
			recv_ghosts(SOUTH, neighbours, recv_buffer_start, l_chunk, horizontal_border_t, array_of_requests_ns, &current_request, chunk_dimensions, g);
			send_ghosts(NORTH, neighbours, send_buffer_start, l_chunk, horizontal_border_t, array_of_requests_ns, &current_request, ((rank * num + i) * N_DIMENSIONS) + NORTH);
			send_ghosts(SOUTH, neighbours, send_buffer_start, l_chunk, horizontal_border_t, array_of_requests_ns, &current_request, ((rank * num + i) * N_DIMENSIONS) + SOUTH);
			// TODO: does status ignore work like that?
			MPI_Waitall(n_requests_ns, array_of_requests_ns, array_of_status_ns);
		}

		// narrow the border
		border = i % g;
		// update the grid
		for (int y = 1 + border; y < chunk_dimensions[Y_AXIS] + 2 * g - (1 + border); ++y) {
			for (int x = 1 + border; x < chunk_dimensions[X_AXIS] + 2 * g - (1 + border); ++x) {
				l_chunk_buf[chunk_index(x, y)] = l_chunk[chunk_index(x, y)] + FACTOR * (l_chunk[chunk_index(x + 1, y)] + l_chunk[chunk_index(x, y + 1)] + l_chunk[chunk_index(x - 1, y)] + l_chunk[chunk_index(x, y - 1)] - 4 * l_chunk[chunk_index(x, y)]);
			}
		}
		swap(&l_chunk, &l_chunk_buf);
	}

	// collect last state
	collect(u1, size, l_chunk, chunk_dimensions, n_processes, g, chunk_inner_values_t);
	//Output last state into file
	if (rank == MAIN_RANK) printResult(u1, size, filename, iter);
	MPI_Finalize();
	return 0;
}
