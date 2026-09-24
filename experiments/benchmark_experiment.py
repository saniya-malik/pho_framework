import os
import sys
import time

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.pho.benchmarking.benchmark import (
    calculate_efficiency,
    calculate_speedup,
)
from src.pho.execution.scheduler import ParallelScheduler


# Create a larger dataset so the ML workload is substantial.
X, y = make_classification(
    n_samples=20000,
    n_features=40,
    n_informative=25,
    n_redundant=10,
    n_classes=2,
    random_state=42
)


PARAMETER_COMBINATIONS = [
    {"n_estimators": 50, "max_depth": 8},
    {"n_estimators": 50, "max_depth": 12},
    {"n_estimators": 75, "max_depth": 8},
    {"n_estimators": 75, "max_depth": 12},
    {"n_estimators": 100, "max_depth": 8},
    {"n_estimators": 100, "max_depth": 12},
    {"n_estimators": 125, "max_depth": 8},
    {"n_estimators": 125, "max_depth": 12},
]


def evaluate_configuration(parameters):
    """Evaluate one hyperparameter configuration."""

    model = RandomForestClassifier(
        random_state=42,
        n_jobs=1,
        **parameters
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=3,
        n_jobs=1
    )

    return float(np.mean(scores))


def run_sequential():
    """Evaluate all configurations sequentially."""

    results = []

    start = time.perf_counter()

    for parameters in PARAMETER_COMBINATIONS:
        score = evaluate_configuration(parameters)

        results.append({
            "parameters": parameters,
            "score": score
        })

    elapsed = time.perf_counter() - start

    return results, elapsed


def parallel_worker(task):
    """Worker function used by the parallel scheduler."""

    return {
        "parameters": task.parameters,
        "score": evaluate_configuration(task.parameters)
    }


def run_parallel(workers):
    """Evaluate configurations using parallel workers."""

    scheduler = ParallelScheduler(
        max_workers=workers
    )

    start = time.perf_counter()

    results = scheduler.execute(
        PARAMETER_COMBINATIONS,
        parallel_worker
    )

    elapsed = time.perf_counter() - start

    return results, elapsed


if __name__ == "__main__":

    print("\nPHO Performance Benchmark")
    print("=" * 50)

    print(
        f"Dataset: {X.shape[0]} samples × "
        f"{X.shape[1]} features"
    )

    print(
        f"Configurations: "
        f"{len(PARAMETER_COMBINATIONS)}"
    )

    sequential_results, sequential_time = run_sequential()

    print(
        f"\nSequential execution time: "
        f"{sequential_time:.4f} seconds"
    )

    benchmark_results = []

    for workers in [1, 2, 4]:

        parallel_results, parallel_time = run_parallel(
            workers
        )

        speedup = calculate_speedup(
            sequential_time,
            parallel_time
        )

        efficiency = calculate_efficiency(
            speedup,
            workers
        )

        benchmark_results.append({
            "workers": workers,
            "time": parallel_time,
            "speedup": speedup,
            "efficiency": efficiency
        })

        print(f"\nWorkers: {workers}")
        print(
            f"Parallel execution time: "
            f"{parallel_time:.4f} seconds"
        )
        print(f"Speedup: {speedup:.4f}")
        print(f"Efficiency: {efficiency:.4f}")

    best_result = max(
        sequential_results,
        key=lambda result: result["score"]
    )

    print("\nBest configuration:")
    print(best_result)