import time


def measure_execution(function, *args, **kwargs):
    """Measure the execution time of a function."""

    start = time.perf_counter()

    result = function(*args, **kwargs)

    elapsed = time.perf_counter() - start

    return result, elapsed


def calculate_speedup(sequential_time, parallel_time):
    """Calculate parallel speedup."""

    if parallel_time <= 0:
        raise ValueError("Parallel execution time must be positive.")

    return sequential_time / parallel_time


def calculate_efficiency(speedup, workers):
    """Calculate parallel efficiency."""

    if workers <= 0:
        raise ValueError("Number of workers must be positive.")

    return speedup / workers