#include <mpi.h>
#include <random>
#include <chrono>
#include <iostream>
#include <sstream>

using namespace std;

#define DEBUG_MODE false

double** init_matrix(const int rows, const int columns) {
    // allocates a matrix M as a 2-dim array so that M[i][j] is the entry in row i, column j
    double** matrix = new double* [rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new double[columns];
    }
    return matrix;
}

void fill_matrix(const int rows, const int columns, double** matrix) {
    // fills a matrix with random numbers
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_real_distribution<double> dist(0, 1);
    // init matrix and fill it with random numbers
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            matrix[i][j] = dist(generator);
        }
    }
}

double calc_block(const int length, const double a_row[], const double b_column[]) {
    double sum = 0;
    for (int i = 0; i < length; ++i) {
        sum += a_row[i] * b_column[i];
    }
    return sum;
}

int main(int argc, char* argv[]) {
    // init mpi
    int ret = MPI_Init(&argc, &argv);
    if (ret != MPI_SUCCESS) {
        MPI_Abort(MPI_COMM_WORLD, ret);
    }
    int num, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // abort if number of processes isn't even
    if (num < 2) {
        cerr << "Number of processes should at least be 2." << endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // make sure message size is passed as a parameter
    if (argc != 4) {
        cerr << "Sizes of matrices (m, n, r) needed." << endl << "A: m x r" << endl << "B: r x n" << "C: m x n" << endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // matrices:
    // A: m x r
    // B: r x n
    // C: m x n
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int r = atoi(argv[3]);

    // divide the calculations into waves, where every processor gets a new task each wave
    int problem_size = m * n;
    int working_processes = num - 1;
    int n_waves = problem_size / working_processes;
    // if task size is not divisible by number of working processes, add a "rest wave"
    if (problem_size % working_processes != 0) {
        ++n_waves;
    }

    if (rank == 0) {
        // master, distributes work
        // init matrices
        // matrix B is saved transposed to quickly get columns instead of rows
        double** matrix_a = init_matrix(m, r);
        double** matrix_b_T = init_matrix(n, r);
        double** matrix_c = init_matrix(m, n);
        // fill A and B with random numbers
        fill_matrix(m, r, matrix_a);
        fill_matrix(n, r, matrix_b_T);

        // measure comm time start
        auto clock = std::chrono::high_resolution_clock();
        auto time_start = clock.now();

        // now task the processes with the calculation of entries in the matrix C
        for (int wave = 0; wave < n_waves; ++wave) {
            // careful with the rest wave!
            int completed_tasks = wave * working_processes;
            int participating_processes = min(working_processes, problem_size - completed_tasks);
            // send the task
            for (int tasked_rank = 1; tasked_rank <= participating_processes; ++tasked_rank) {
                // which element the tasked ranked should calculate, starting at 0
                int task_element = completed_tasks + (tasked_rank - 1);
                int i = task_element / n;
                int j = task_element % n;
                // C[i][j] is calculated with row i of matrix A and column j of matrix B (row j of B^T)
                MPI_Send(matrix_a[i], r, MPI_DOUBLE, tasked_rank, 0, MPI_COMM_WORLD);
                MPI_Send(matrix_b_T[j], r, MPI_DOUBLE, tasked_rank, 1, MPI_COMM_WORLD);
            }
            // receive the result
            for (int tasked_rank = 1; tasked_rank <= participating_processes; ++tasked_rank) {
                // which element the tasked ranked should calculate, starting at 0
                int task_element = completed_tasks + (tasked_rank - 1);
                int i = task_element / n;
                int j = task_element % n;
                // C[i][j] is calculated with row i of matrix A and column j of matrix B (row j of B^T)
                MPI_Recv(&matrix_c[i][j], 1, MPI_DOUBLE, tasked_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        // measure comm time end
        auto time_end = clock.now();
        double time = chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count() / 1.0E6;

        cout << "Calculation took " << time << " ms." << endl;
    } else {
        // slaves, work
        double* a_row = new double[r];
        double* b_column = new double[r];
        for (int wave = 0; wave < n_waves; ++wave) {
            // careful with the rest wave!
            int completed_tasks = wave * working_processes;
            int participating_processes = min(working_processes, problem_size - completed_tasks);
            if (rank > participating_processes) continue;
            // get row i from matrix A and column j from matrix B, calculate C[i][j], send back the result
            MPI_Recv(a_row, r, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(b_column, r, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double result = calc_block(r, a_row, b_column);
            MPI_Send(&result, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        }
    }

    // all went fine
    MPI_Finalize();
    return EXIT_SUCCESS;
}