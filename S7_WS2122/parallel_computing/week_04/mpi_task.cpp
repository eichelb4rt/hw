#include <mpi.h>
#include <random>
#include <chrono>
#include <iostream>
#include <sstream>

using namespace std;

#define DEBUG_MODE false

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
    if (num % 2 != 0) {
        cerr << "Number of processes has to be even." << endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // make sure message size is passed as a parameter
    if (argc != 2) {
        cerr << "Message size needed." << endl;
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // prepare arrays
    int msg_size = atoi(argv[1]);
    double* rands_send = (double*) malloc(msg_size * sizeof(double));
    double* rands_recv = (double*) malloc(msg_size * sizeof(double));

    // fill sent array with random numbers
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count() * (1 + rank);
    default_random_engine generator(seed);
    uniform_real_distribution<double> dist(0, 1);
    for (size_t i = 0; i < msg_size; ++i) {
        rands_send[i] = dist(generator);
    }

    // measure comm time start
    auto clock = std::chrono::high_resolution_clock();
    auto time_start = clock.now();

    // communicate
    MPI_Status stat;
    int partner;
    if (rank % 2 == 0) {
        partner = rank + 1;
        MPI_Send(rands_send, msg_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);
        MPI_Recv(rands_recv, msg_size, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, &stat);
    } else {
        partner = rank - 1;
        MPI_Recv(rands_recv, msg_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, &stat);
        MPI_Send(rands_send, msg_size, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD);
    }

    // measure comm time end
    auto time_end = clock.now();
    double time = chrono::duration_cast<std::chrono::nanoseconds>(time_end - time_start).count() / 1.0E6;

    // print communicated numbers
    stringstream ss;
    ss << "Rank " << rank << " took " << time << " ms." << endl;
    if (DEBUG_MODE) {
        ss << "Rank " << rank << " sent these numbers:" << endl;
        for (size_t i = 0; i < msg_size; ++i) {
            ss << rands_send[i] << endl;
        }
        ss << "Rank " << rank << " received these numbers from " << partner << ":" << endl;
        for (size_t i = 0; i < msg_size; ++i) {
            ss << rands_recv[i] << endl;
        }
        ss << endl;
    }
    cout << ss.str();

    // all went fine
    MPI_Finalize();
    return EXIT_SUCCESS;
}