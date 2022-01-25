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
	int n_printed_iterations = 5;
	int printed_iterations[5] = { 0, 3, 5, 7, 10 };

	int num, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &num);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int n_processes[N_DIMENSIONS];
	int finalize;
	setup(rank, num, argc, argv, &size, &iter, &g, n_processes, filename, &finalize);
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
	// exchange all known data across the vertical line, g wide. (stride is the whole horizontal dimension)
	MPI_Datatype vertical_border_t;
	MPI_Type_vector(chunk_dimensions[Y_AXIS], g, chunk_dimensions[X_AXIS] + 2 * g, MPI_DOUBLE, &vertical_border_t);
	MPI_Type_commit(&vertical_border_t);
	// exchange all data across the horizontal line, whole horizontal line wide, g blocks. (stride is the whole horizontal dimension)
	MPI_Datatype horizontal_border_t;
	MPI_Type_vector(g, chunk_dimensions[X_AXIS] + 2 * g, chunk_dimensions[X_AXIS] + 2 * g, MPI_DOUBLE, &horizontal_border_t);
	MPI_Type_commit(&horizontal_border_t);
	// type for all the inner values (non-ghost-blocks)
	MPI_Datatype chunk_inner_values_t;
	MPI_Type_vector(chunk_dimensions[Y_AXIS], chunk_dimensions[X_AXIS], chunk_dimensions[X_AXIS] + 2 * g, MPI_DOUBLE, &chunk_inner_values_t);
	MPI_Type_commit(&chunk_inner_values_t);
	// type for a chunk placed in the global array
	MPI_Datatype chunk_in_global_array_t;
	MPI_Type_vector(chunk_dimensions[Y_AXIS], chunk_dimensions[X_AXIS], size, MPI_DOUBLE, &chunk_in_global_array_t);
	MPI_Type_commit(&chunk_in_global_array_t);

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

	// printed iterations
	int* is_printed = (int*) calloc(iter, sizeof(int));
	int print_last_iteration = 0;
	for (int i = 0; i < n_printed_iterations; ++i) {
		int printed_iteration = printed_iterations[i];
		// ignore printed iterations that are not among the actual iterations
		if (printed_iteration < 0 || printed_iteration >= iter) {
			continue;
		}
		is_printed[printed_iteration] = 1;
	}

	// we divide all the requests into requests between east and west, and requests between north and south (because we want to send the corners correctly)
	int n_requests_ew = (neighbours[EAST] != UNDEFINED_RANK) + (neighbours[WEST] != UNDEFINED_RANK);
	int n_requests_ns = (neighbours[NORTH] != UNDEFINED_RANK) + (neighbours[SOUTH] != UNDEFINED_RANK);
	MPI_Request* array_of_requests_ew = (MPI_Request*) malloc(n_requests_ew * sizeof(MPI_Request));
	MPI_Request* array_of_requests_ns = (MPI_Request*) malloc(n_requests_ns * sizeof(MPI_Request));
	MPI_Status* array_of_status_ew = (MPI_Status*) malloc(n_requests_ew * sizeof(MPI_Status));
	MPI_Status* array_of_status_ns = (MPI_Status*) malloc(n_requests_ns * sizeof(MPI_Status));

	// define where communication is written from and to
	int send_indices[N_NEIGHBOURS];
	int recv_indices[N_DIMENSIONS];
	set_comm_indices(send_indices, recv_indices, chunk_dimensions, g);
	// request index that's dynamically adapted by the comm calls
	int current_request = 0;

	// DEBUG
	int coords[N_DIMENSIONS];
	get_coords(rank, n_processes, coords);
	printf("rank %d here, coords: %d,%d, EAST: %d, WEST: %d, NORTH: %d, SOUTH: %d\n", rank, coords[X_AXIS], coords[Y_AXIS], neighbours[EAST], neighbours[WEST], neighbours[NORTH], neighbours[SOUTH]);

	// border that narrows with the number of iterations that we have not communicated
	int border;
	// factor for thermal calculation stuff
	const int FACTOR = PARAM_ALPHA * PARAM_T / (PARAM_H * PARAM_H);
	// main loop
	for (int i = 0; i < iter; ++i) {
		// maybe print?
		if (is_printed[i]) {
			collect(u1, l_chunk, n_processes, chunk_inner_values_t, chunk_in_global_array_t);
			if (rank == MAIN_RANK) printResult(u1, size, filename, i);
		}
		// only communicate every g iterations
		if (i % g == 0) {
			// communication between east and west
			current_request = 0;
			recv_ghosts(neighbours[EAST], l_chunk, recv_indices[EAST], vertical_border_t, array_of_requests_ew, &current_request);
			recv_ghosts(neighbours[WEST], l_chunk, recv_indices[WEST], vertical_border_t, array_of_requests_ew, &current_request);
			send_ghosts(neighbours[EAST], l_chunk, send_indices[EAST], vertical_border_t, array_of_requests_ew, &current_request);
			send_ghosts(neighbours[WEST], l_chunk, send_indices[WEST], vertical_border_t, array_of_requests_ew, &current_request);
			MPI_Waitall(n_requests_ew, array_of_requests_ew, array_of_status_ew);
			printf("rank %d still here, iteration %d, request index: %d\n", rank, i, current_request);
			// communication between north and south
			current_request = 0;
			recv_ghosts(neighbours[NORTH], l_chunk, recv_indices[NORTH], horizontal_border_t, array_of_requests_ns, &current_request);
			recv_ghosts(neighbours[SOUTH], l_chunk, recv_indices[SOUTH], horizontal_border_t, array_of_requests_ns, &current_request);
			send_ghosts(neighbours[NORTH], l_chunk, send_indices[NORTH], horizontal_border_t, array_of_requests_ns, &current_request);
			send_ghosts(neighbours[SOUTH], l_chunk, send_indices[SOUTH], horizontal_border_t, array_of_requests_ns, &current_request);
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
		swap(l_chunk, l_chunk_buf);
	}


	// collect last state
	collect(u1, l_chunk, n_processes, chunk_inner_values_t, chunk_in_global_array_t);
	//Output last state into file
	if (rank == MAIN_RANK) printResult(u1, size, filename, iter);
	MPI_Finalize();
	return 0;
}
