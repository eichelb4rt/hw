#include "waermeleitung.h"

#define N_SECTIONS_MEASURED 3
#define TIME_CALC 0
#define TIME_COMMS 1
#define TIME_TOTAL 2

double measurements_sum[N_SECTIONS_MEASURED];
int measurements_count[N_SECTIONS_MEASURED];
double avg_benchmark_results[N_SECTIONS_MEASURED];
double total_benchmark_results[N_SECTIONS_MEASURED]; 
vector<chrono::_V2::system_clock::time_point> start_time(N_SECTIONS_MEASURED);

#define SETUP_BENCHMARKS(enable) bool benchmarking_enabled = enable;
#define BENCHMARKS_ENABLED benchmarking_enabled

#define START_MEASURE(measurement) if (benchmarking_enabled) {start_time[measurement] = chrono::high_resolution_clock::now();}
#define END_MEASURE(measurement) if (benchmarking_enabled) {++measurements_count[measurement]; measurements_sum[measurement] += (chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start_time[measurement]).count() / 1E9);}

#define PROCESS_BENCHMARKS if (benchmarking_enabled) {average_all(avg_benchmark_results, num, measurements_sum, measurements_count); total_all(total_benchmark_results, num, measurements_sum, measurements_count);}
// #define PRINT_BENCHMARKS if(benchmarking_enabled && rank == MAIN_RANK) {stream << print_benchmarks(avg_benchmark_results, total_benchmark_results, measurements_sum[TIME_TOTAL]);}
#define PRINT_BENCHMARKS if(benchmarking_enabled && rank == MAIN_RANK) {print_benchmarks(avg_benchmark_results, total_benchmark_results, measurements_sum[TIME_TOTAL]);}

void print_benchmarks(double* avg_benchmark, double* total_benchmark, double total_time) {
    // stringstream ss;
    // ss << "times in seconds" << endl << "avg calc time:\t" << avg_benchmark[TIME_CALC] << endl << "tot calc time:\t" << total_benchmark[TIME_CALC] << endl << "avg comm time:\t" << avg_benchmark[TIME_COMMS] << endl << "tot comm time:\t" << total_benchmark[TIME_COMMS] << endl << "total time:\t" << total_time << endl;
    // return ss.str();
    printf("times in seconds:\navg calc time:\t%lf\ntot calc time:\t%lf\navg comm time:\t%lf\ntot comm time:\t%lf\ntotal time:\t%lf\n", avg_benchmark[TIME_CALC], total_benchmark[TIME_CALC], avg_benchmark[TIME_COMMS], total_benchmark[TIME_COMMS], total_time);
}

// average of all the processes
void average_process_measurements(double* recv, int num, double* measurement_array, int measurements_count[]) {
    // sum
    MPI_Reduce(measurement_array, recv, N_SECTIONS_MEASURED, MPI_DOUBLE, MPI_SUM, MAIN_RANK, MPI_COMM_WORLD);
    // average
    for (int measurement = 0; measurement < N_SECTIONS_MEASURED; ++measurement) {
        recv[measurement] /= num;
    }
}

// average of total times between processes
void total_all(double* recv, int num, double measurements_sum[], int measurements_count[]) {
    average_process_measurements(recv, num, measurements_sum, measurements_count);
}

// average time betweeen all the measurements
void average_all(double* recv, int num, double measurements_sum[], int measurements_count[]) {
    double measurements_avg[N_SECTIONS_MEASURED];
    for (int measurement = 0; measurement < N_SECTIONS_MEASURED; ++measurement) {
        measurements_avg[measurement] = measurements_sum[measurement] / measurements_count[measurement];
    }
    average_process_measurements(recv, num, measurements_avg, measurements_count);
}
