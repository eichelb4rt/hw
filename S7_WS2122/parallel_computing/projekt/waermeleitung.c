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

	//TODO: Übergabeparameter für [size iter g filename] einlesen
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
	int* l_chunk = (double*) malloc((chunk_dimensions[X_AXIS] + 2 * g) * (chunk_dimensions[Y_AXIS] + 2 * g) * sizeof(double));
	// buffer for local chunk
	int* l_chunk_buf = (double*) malloc((chunk_dimensions[X_AXIS] + 2 * g) * (chunk_dimensions[Y_AXIS] + 2 * g) * sizeof(double));
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
		if (printed_iteration < 0 || printed_iteration > iter) {
			// bad bla bla
		} else if (printed_iteration == iter) {
			print_last_iteration = 1;
		} else {
			is_printed[printed_iteration] = 1;
		}
	}

	// we divide all the requests into requests between east and west, and requests between north and south (because we want to send the corners correctly)
	int n_requests_ew = 4;
	int n_requests_ns = 4;
	MPI_Request* array_of_requests_ew = (MPI_Request*) malloc(n_requests_ew * sizeof(MPI_Request));
	MPI_Request* array_of_requests_ns = (MPI_Request*) malloc(n_requests_ns * sizeof(MPI_Request));
	// init array of requests (so that we don't wait for neighbours that don't exist)
	for (int i = 0; i < n_requests_ns; ++i) {
		array_of_requests_ew[i] = MPI_REQUEST_NULL;
		array_of_requests_ns[i] = MPI_REQUEST_NULL;
	}

	// border that narrows with the number of iterations that we have not communicated
	int border;
	// factor for thermal calculation stuff
	const int FACTOR = PARAM_ALPHA * PARAM_T / (PARAM_H * PARAM_H);
	// main loop
	for (int i = 0; i < iter; ++i) {
		// maybe print?
		// only communicate every g iterations
		if (i % g == 0) {
			// communication between east and west
			recv_east(neighbours[EAST], l_chunk, chunk_dimensions, g, vertical_border_t, &array_of_requests_ew[0]);
			recv_west(neighbours[WEST], l_chunk, chunk_dimensions, g, vertical_border_t, &array_of_requests_ew[1]);
			send_east(neighbours[EAST], l_chunk, chunk_dimensions, g, vertical_border_t, &array_of_requests_ew[n_requests_ew + 0]);
			send_west(neighbours[WEST], l_chunk, chunk_dimensions, g, vertical_border_t, &array_of_requests_ew[n_requests_ew + 1]);
			MPI_Waitall(n_requests_ew, array_of_requests_ew, MPI_STATUS_IGNORE);
			// communication between north and south
			recv_north(neighbours[NORTH], l_chunk, chunk_dimensions, g, horizontal_border_t, &array_of_requests_ns[0]);
			recv_south(neighbours[SOUTH], l_chunk, chunk_dimensions, g, horizontal_border_t, &array_of_requests_ns[1]);
			send_north(neighbours[NORTH], l_chunk, chunk_dimensions, g, horizontal_border_t, &array_of_requests_ns[n_requests_ns + 0]);
			send_south(neighbours[SOUTH], l_chunk, chunk_dimensions, g, horizontal_border_t, &array_of_requests_ns[n_requests_ns + 1]);
			// TODO: does status ignore work like that?
			MPI_Waitall(n_requests_ns, array_of_requests_ns, MPI_STATUS_IGNORE);
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


	//Output last state into file
	printResult(u1, size, filename, iter);
	MPI_Finalize();
	return 0;
}
