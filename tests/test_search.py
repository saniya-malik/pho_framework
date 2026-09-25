import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from src.pho.search.grid_search import (
    SequentialGridSearch
)

from src.pho.search.parallel_grid_search import (
    ParallelGridSearch
)

from src.pho.search.random_search import (
    ParallelRandomSearch
)


def get_dataset():
    """Load the Iris dataset for testing."""

    return load_iris(
        return_X_y=True
    )


def test_sequential_grid_search():

    X, y = get_dataset()

    search = SequentialGridSearch(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_grid={
            "n_estimators": [10, 20],
            "max_depth": [2, 4]
        },
        cv=3
    )

    search.fit(X, y)

    assert len(search.results_) == 4
    assert search.best_params_ is not None
    assert search.best_score_ > 0


def test_parallel_grid_search():

    X, y = get_dataset()

    search = ParallelGridSearch(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_grid={
            "n_estimators": [10, 20],
            "max_depth": [2, 4]
        },
        cv=3,
        max_workers=2
    )

    search.fit(X, y)

    assert len(search.results_) == 4
    assert search.best_params_ is not None
    assert search.best_score_ > 0


def test_parallel_random_search():

    X, y = get_dataset()

    search = ParallelRandomSearch(
        estimator=RandomForestClassifier(
            random_state=42
        ),
        param_distributions={
            "n_estimators": [10, 20, 30],
            "max_depth": [2, 4],
            "min_samples_split": [2, 4]
        },
        n_iter=4,
        cv=3,
        random_state=42,
        max_workers=2
    )

    search.fit(X, y)

    assert len(search.results_) == 4
    assert search.best_params_ is not None
    assert search.best_score_ > 0