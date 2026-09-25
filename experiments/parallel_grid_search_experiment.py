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

from src.pho.search.parallel_grid_search import (
    ParallelGridSearch
)


if __name__ == "__main__":

    X, y = load_iris(
        return_X_y=True
    )

    parameter_grid = {
        "n_estimators": [10, 20, 30],
        "max_depth": [2, 4],
        "min_samples_split": [2, 4]
    }

    search = ParallelGridSearch(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_grid=parameter_grid,
        cv=3,
        max_workers=2
    )

    search.fit(X, y)

    print("\nPHO Parallel Grid Search Results")
    print("=" * 50)

    for result in search.get_results():

        print(
            f"Task {result['task_id']}: "
            f"{result['params']} → "
            f"{result['score']:.4f}"
        )

    print("\nBest configuration:")
    print(search.best_params_)

    print(
        f"Best CV score: "
        f"{search.best_score_:.4f}"
    )

    print(
        f"Execution time: "
        f"{search.elapsed_time_:.4f} seconds"
    )

    print(
        f"Configurations evaluated: "
        f"{len(search.results_)}"
    )