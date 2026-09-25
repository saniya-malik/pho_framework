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

from src.pho.search.random_search import (
    ParallelRandomSearch
)


if __name__ == "__main__":

    X, y = load_iris(
        return_X_y=True
    )

    parameter_distributions = {
        "n_estimators": [10, 20, 30, 40, 50],
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [2, 4, 6]
    }

    search = ParallelRandomSearch(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_distributions=parameter_distributions,
        n_iter=5,
        cv=3,
        random_state=42,
        max_workers=2
    )

    search.fit(X, y)

    print("\nPHO Random Search Results")
    print("=" * 45)

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