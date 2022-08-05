from timeit import default_timer as timer

import numpy as np

start_times: dict[str, float] = {}
measured: dict[str, list[float]] = {}


def start(clock_name: str):
    start_times[clock_name] = timer()


def stop(clock_name: str, print_time=False):
    if clock_name not in start_times:
        raise LookupError(f"The clock \"{clock_name}\" hasn't started yet!")
    # measure time
    end_time = timer()
    time_elapsed = end_time - start_times[clock_name]
    del start_times[clock_name]
    # save it
    if clock_name not in measured:
        measured[clock_name] = []
    measured[clock_name].append(time_elapsed)
    # print it if wanted
    if print_time:
        print(f"timed {clock_name}: {round(time_elapsed, 2)}s")


def avg(clock_name: str):
    if clock_name not in measured:
        raise LookupError(f"The clock \"{clock_name}\" hasn't been measured!")
    mean_time = np.mean(measured[clock_name])
    print(f"{clock_name}: avg time {round(mean_time, 2)}s")
