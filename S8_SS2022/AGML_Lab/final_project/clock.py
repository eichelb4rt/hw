from timeit import default_timer as timer

start_times: dict[str, float] = {}


def start(clock_name: str):
    start_times[clock_name] = timer()


def stop(clock_name: str):
    if clock_name not in start_times:
        raise LookupError(f"The clock \"{clock_name}\" hasn't started yet!")
    end_time = timer()
    time_elapsed = end_time - start_times[clock_name]
    print(f"Timed {clock_name}: {round(time_elapsed, 2)}s")
    del start_times[clock_name]
