import os
import time

from src.pho.execution.worker_pool import ParallelWorkerPool


def square_task(x):
    time.sleep(1)
    return {
        "input": x,
        "output": x * x,
        "process_id": os.getpid()
    }


if __name__ == "__main__":
    tasks = [1, 2, 3, 4]

    start = time.perf_counter()

    pool = ParallelWorkerPool(max_workers=4)
    results = pool.execute(tasks, square_task)

    elapsed = time.perf_counter() - start

    print("Results:", results)
    print("Elapsed time:", round(elapsed, 2), "seconds")
    print("Unique worker processes:",
          len(set(result["process_id"] for result in results)))