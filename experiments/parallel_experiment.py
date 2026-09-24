import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.pho.execution.scheduler import ParallelScheduler


def evaluate_configuration(task):
    """Evaluate one hyperparameter configuration."""

    X, y = load_iris(return_X_y=True)

    model = RandomForestClassifier(
        random_state=42,
        **task.parameters
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=3,
        n_jobs=1
    )

    return {
        "task_id": task.task_id,
        "parameters": task.parameters,
        "score": float(scores.mean())
    }


if __name__ == "__main__":

    parameter_combinations = [
        {"n_estimators": 10, "max_depth": 2},
        {"n_estimators": 10, "max_depth": 4},
        {"n_estimators": 20, "max_depth": 2},
        {"n_estimators": 20, "max_depth": 4},
        {"n_estimators": 30, "max_depth": 2},
        {"n_estimators": 30, "max_depth": 4},
        {"n_estimators": 40, "max_depth": 2},
        {"n_estimators": 40, "max_depth": 4},
    ]

    scheduler = ParallelScheduler(max_workers=4)

    results = scheduler.execute(
        parameter_combinations,
        evaluate_configuration
    )

    results.sort(key=lambda result: result["score"], reverse=True)

    print("\nParallel Hyperparameter Search Results")
    print("=" * 45)

    for result in results:
        print(
            f"Task {result['task_id']}: "
            f"{result['parameters']} → "
            f"{result['score']:.4f}"
        )

    print("\nBest configuration:")
    print(results[0])