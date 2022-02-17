#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define OUT_OF_BOUNDS(index, start, end) index < start || index >= end
#define STANCE_DISTANCE 1

// Initialisiere Wärmefeld mit Startwerten:
// innen: 0.0
// Rand: 
// links warm=25.0
// rechts kalt=-25.0
void init(double* t, int size) {
	for (int i = 0; i < size; i++) {
		if (i == 0) t[i] = 25;
		else if (i == size - 1) t[i] = -25;
		else t[i] = 0;
	}
}

void run_parallel_region(int num, int rank) {

}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	// Größe des Feldes
	int size = 128;
	// Anzahl Iterationen
	int iter = 100;
	// Geisterzonenbreite
	int g = 1;

	// funny values
	double alpha = 1;
	double t_delta = 1;
	double h = 2;
	const double factor = alpha * t_delta / (h * h);

	// 2 Speicherbereiche für das Wärmefeld
	double* initial_temp, * final_temp;

	// Größe des Speicherbereiches
	int mem = size * sizeof(double);
	// Allokiere Speicher auf Host
	initial_temp = (double*) malloc(mem);
	final_temp = (double*) malloc(mem);
	// Initialisiere Speicher
	init(initial_temp, size);

	// program
	int num, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &num);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// init parallel region
	// determine where and how big the blocks are
	int g_block_size = ceil((double) size / num);
	int l_block_start = g_block_size * rank;
	int l_block_end = max(g_block_size * (rank + 1), size);
	int l_block_size = l_block_end - l_block_start;

	// make local array for calculations
	int l_temps_size = l_block_size + 2 * g;
	double* l_temps = (double*) malloc(l_temps_size * sizeof(double));

	// fill with initial values
	for (int i = 0; i < l_temps_size; ++i) {
		// get the value from the initial array
		int g_index = l_block_start + i - g;
		// padding -> closest neighbour
		if (g_index < 0) g_index = 0;
		else if (g_index >= size) g_index = size - 1;

		l_temps[i] = initial_temp[g_index];
	}

	// calculate iterations
	for (int t = 0; t < iter; t += g) {
		// request ghost blocks (communication is only needed after first iteration)
		if (t > 0) {
			for (int i = 0; i < l_temps_size; ++i) {
				int containing_rank = i / g_block_size;
				// if the element is not already in this rank, 
				if (containing_rank != rank) {
					// get the stuff from that rank
				}
			}
		}
		// calculate inner values
		for (int i = g + STANCE_DISTANCE; i < l_temps_size - g - STANCE_DISTANCE; ++i) {
			l_temps[i] = factor * (l_temps[i - 1] + l_temps[i + 1] - 2 * l_temps[i]);
		}
		// wait for ghost points (communication is only needed after first iteration)
		if (t > 0) {

		}
		// calculate outer values
		for (int i = STANCE_DISTANCE; i < l_temps_size - STANCE_DISTANCE; ++i) {
			// don't calculate inner values again
			if (!OUT_OF_BOUNDS(i, g + STANCE_DISTANCE, l_temps_size - (g + STANCE_DISTANCE))) continue;
			l_temps[i] = factor * (l_temps[i - 1] + l_temps[i + 1] - 2 * l_temps[i]);
		}
	}

	// write back final values
	if (rank == 0) {
		// receive values
	} else {
		// send values
	}

	MPI_Finalize();
	return 0;
}
