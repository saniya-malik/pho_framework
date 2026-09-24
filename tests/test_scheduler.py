import os
import time

from src.pho.execution.scheduler import ParallelScheduler


def evaluate_task(task):
    time.sleep(1)

    return {
        "task_id": task.task_id,
        "parameters": task.parameters,
        "process_id": os.getpid()
    }


if __name__ == "__main__":
    parameter_combinations = [
        {"n_estimators": 10},
        {"n_estimators": 20},
        {"n_estimators": 30},
        {"n_estimators": 40},
    ]

    scheduler = ParallelScheduler(max_workers=4)

    start = time.perf_counter()

    results = scheduler.execute(
        parameter_combinations,
        evaluate_task
    )

    elapsed = time.perf_counter() - start

    print("Results:", results)
    print("Elapsed time:", round(elapsed, 2), "seconds")
    print(
        "Unique worker processes:",
        len(set(result["process_id"] for result in results))
    )