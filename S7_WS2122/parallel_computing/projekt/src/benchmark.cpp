#include "waermeleitung.h"

#define N_SECTIONS_MEASURED 3
#define TIME_CALC 0
#define TIME_COMMS 1
#define TIME_TOTAL 2

#define SETUP_BENCHMARKS(enable) vector<vector<double>> measured_times(N_SECTIONS_MEASURED); vector<chrono::_V2::system_clock::time_point> start_time(N_SECTIONS_MEASURED); bool benchmarking_enabled = enable;
#define BENCHMARKS_ENABLED benchmarking_enabled

#define START_MEASURE(measure) if (benchmarking_enabled) {start_time[measure] = chrono::high_resolution_clock::now();}
#define END_MEASURE(measure) if (benchmarking_enabled) {measured_times[measure].push_back(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start_time[measure]).count() / 1E9);}

#define PROCESS_BENCHMARKS vector<double> avg_benchmark, total_benchmark; if (benchmarking_enabled) {avg_benchmark = average_all(num, measured_times); total_benchmark = total_all(num, measured_times);}
#define PRINT_BENCHMARKS(stream) if(benchmarking_enabled && rank == MAIN_RANK) {stream << print_benchmarks(avg_benchmark, total_benchmark, measured_times[TIME_TOTAL][0]);}

string print_benchmarks(vector<double> avg_benchmark, vector<double> total_benchmark, double total_time) {
    stringstream ss;
    ss << "times in seconds" << endl << "avg calc time:\t" << avg_benchmark[TIME_CALC] << endl << "tot calc time:\t" << total_benchmark[TIME_CALC] << endl << "avg comm time:\t" << avg_benchmark[TIME_COMMS] << endl << "tot comm time:\t" << total_benchmark[TIME_COMMS] << endl << "total time:\t" << total_time << endl;
    return ss.str();
}

// average of a single process
vector<double> total_time(vector<vector<double>>& measured_times) {
    vector<double> sum(N_SECTIONS_MEASURED, 0);
    for (int measurement = 0; measurement < N_SECTIONS_MEASURED; ++measurement) {
        if (measured_times[measurement].empty()) continue;
        // sum over the vector
        for (double& time : measured_times[measurement]) {
            sum[measurement] += time;
        }
    }
    return sum;
}

// average of a single process
vector<double> average_measurements(vector<vector<double>>& measured_times) {
    vector<double> avg_time(N_SECTIONS_MEASURED, 0);
    for (int measurement = 0; measurement < N_SECTIONS_MEASURED; ++measurement) {
        if (measured_times[measurement].empty()) continue;
        // avg over the vector
        int n_measurements = measured_times[measurement].size();
        for (double& time : measured_times[measurement]) {
            avg_time[measurement] += time;
        }
        avg_time[measurement] /= n_measurements;
    }
    return avg_time;
}

// average of all the processes
vector<double> average_process_measurements(int num, vector<double>& times_single_process) {
    vector<double> avg_time_collected(N_SECTIONS_MEASURED, 0);
    MPI_Reduce(&times_single_process[0], &avg_time_collected[0], N_SECTIONS_MEASURED, MPI_DOUBLE, MPI_SUM, MAIN_RANK, MPI_COMM_WORLD);
    for (double& measurement : avg_time_collected) {
        measurement /= num;
    }
    return avg_time_collected;
}

// average of total times between processes
vector<double> total_all(int num, vector<vector<double>>& measured_times) {
    auto local_total = total_time(measured_times);
    return average_process_measurements(num, local_total);
}

// average time betweeen all the measurements
vector<double> average_all(int num, vector<vector<double>>& measured_times) {
    auto local_avg = average_measurements(measured_times);
    return average_process_measurements(num, local_avg);
}
