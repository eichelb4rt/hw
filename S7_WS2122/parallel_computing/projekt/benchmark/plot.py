import glob
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt


FIELD_SIZES = [128, 256, 512, 1024, 2048]


class Result:
    def __init__(self, avg_calc: np.double, avg_comm: np.double, total_calc: np.double, total_comm: np.double, total: np.double):
        self.avg_calc = avg_calc
        self.avg_comm = avg_comm
        self.total_calc = total_calc
        self.total_comm = total_comm
        self.total = total

    def __add__(self, other):
        avg_calc = self.avg_calc + other.avg_calc
        avg_comm = self.avg_comm + other.avg_comm
        total_calc = self.total_calc + other.total_calc
        total_comm = self.total_comm + other.total_comm
        total = self.total + other.total
        return Result(avg_calc, avg_comm, total_calc, total_comm, total)

    def divide_by_constant(self, constant):
        self.avg_calc /= constant
        self.avg_comm /= constant
        self.total_calc /= constant
        self.total_comm /= constant
        self.total /= constant

    def __str__(self):
        return f"times in seconds\navg calc time:\t{self.avg_calc}\ntot calc time:\t{self.total_calc}\navg comm time:\t{self.avg_comm}\ntot comm time:\t{self.total_comm}\ntotal time:\t{self.total}"


def read_result(file_path: str) -> Result:
    with open(file_path, "r") as file:
        # times in seconds
        file.readline()
        measurements = np.array([file.readline().split(
            ":")[1].strip() for _ in range(5)]).astype(np.double)
        return Result(measurements[0], measurements[2], measurements[1], measurements[3], measurements[4])


def read_results(file_glob: str) -> List[Result]:
    return [read_result(file_path) for file_path in glob.glob(file_glob)]


def avg_result(results: List[Result]) -> Result:
    end_result: Result = Result(0, 0, 0, 0, 0)
    for result in results:
        end_result += result
    end_result.divide_by_constant(len(results))
    return end_result


def get_param_g(dir_name: str) -> int:
    return int(Path(dir_name).stem.split("_")[-1])


def plot_field_size_total(field_size: int):
    directories_glob = f"./benchmark/results/size_{field_size}/*"
    plot_save_file = f"./benchmark/plots/plot_total_{field_size}.png"
    dir_paths = [(get_param_g(file), file)
                 for file in glob.glob(directories_glob)]
    dir_paths.sort(key=lambda tup: tup[0])
    avg_results = [avg_result(read_results(
        f"{dir_path}/*")) for _, dir_path in dir_paths]

    x_axis = [param_g for param_g, _ in dir_paths]
    line_total_calc = [result.total_calc for result in avg_results]
    line_total_comm = [result.total_comm for result in avg_results]
    line_total = [result.total for result in avg_results]

    plt.plot(x_axis, line_total_calc, label="total T comp")
    plt.plot(x_axis, line_total_comm, label="total T comm")
    plt.plot(x_axis, line_total, label="total T all")

    plt.xlabel('parameter g')
    plt.ylabel('time in seconds')
    plt.title(f"Waermeleitung Benchmark (size {field_size})")
    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.savefig(plot_save_file)
    plt.close()

def plot_field_size_avg(field_size: int):
    directories_glob = f"./benchmark/results/size_{field_size}/*"
    plot_save_file = f"./benchmark/plots/plot_avg_{field_size}.png"
    dir_paths = [(get_param_g(file), file)
                 for file in glob.glob(directories_glob)]
    dir_paths.sort(key=lambda tup: tup[0])
    avg_results = [avg_result(read_results(
        f"{dir_path}/*")) for _, dir_path in dir_paths]

    x_axis = [param_g for param_g, _ in dir_paths]
    line_avg_calc = [result.avg_calc for result in avg_results]
    line_avg_comm = [result.avg_comm for result in avg_results]
    # a single calculation was measured. let's scale it (we actually calcualte g times for every communication)
    for i in range(len(x_axis)):
        line_avg_calc[i] *= int(x_axis[i])

    plt.plot(x_axis, line_avg_calc, label="average T comp")
    plt.plot(x_axis, line_avg_comm, label="average T comm")

    plt.xlabel('parameter g')
    plt.ylabel('time in seconds')
    plt.title(f"Waermeleitung Benchmark (size {field_size})")
    # show a legend on the plot
    plt.legend()
    # function to show the plot
    plt.savefig(plot_save_file)
    plt.close()


def main():
    for field_size in FIELD_SIZES:
        plot_field_size_total(field_size)
        plot_field_size_avg(field_size)


if __name__ == "__main__":
    main()
