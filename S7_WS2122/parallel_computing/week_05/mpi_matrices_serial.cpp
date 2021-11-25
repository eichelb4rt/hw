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
    // make sure message size is passed as a parameter
    if (argc != 4) {
        cerr << "Sizes of matrices (m, n, r) needed." << endl << "A: m x r" << endl << "B: r x n" << "C: m x n" << endl;
        return EXIT_FAILURE;
    }

    // matrices:
    // A: m x r
    // B: r x n
    // C: m x n
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int r = atoi(argv[3]);

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

    // calc matrix
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix_c[i][j] = calc_block(r, matrix_a[i], matrix_b_T[j]);
        }
    }

    // measure comm time end
    auto time_end = clock.now();
    double time = chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count() / 1.0E6;

    cout << "Calculation took " << time << " ms." << endl;
    return EXIT_SUCCESS;
}