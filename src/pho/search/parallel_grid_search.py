import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from time import perf_counter

from src.pho.execution.scheduler import ParallelScheduler


def evaluate_grid_configuration(task_data):
    """Evaluate one hyperparameter configuration."""

    (
        task_id,
        parameters,
        estimator,
        X,
        y,
        cv,
        scoring
    ) = task_data

    model = clone(estimator)

    model.set_params(**parameters)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    return {
        "task_id": task_id,
        "params": parameters,
        "score": float(np.mean(scores))
    }


class ParallelGridSearch:
    """
    Parallel hyperparameter grid search.

    Evaluates all hyperparameter combinations
    using multiple worker processes.
    """

    def __init__(
        self,
        estimator,
        param_grid,
        cv=3,
        scoring=None,
        max_workers=4
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.max_workers = max_workers

        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []
        self.elapsed_time_ = None

    def _generate_combinations(self):
        """Generate all hyperparameter combinations."""

        from itertools import product

        keys = list(self.param_grid.keys())

        values = [
            self.param_grid[key]
            for key in keys
        ]

        combinations = []

        for combination in product(*values):

            combinations.append(
                dict(zip(keys, combination))
            )

        return combinations

    def fit(self, X, y):
        """Run the parallel grid search."""

        start_time = perf_counter()

        combinations = (
            self._generate_combinations()
        )

        scheduler = ParallelScheduler(
            max_workers=self.max_workers
        )

        tasks = [
            (
                index,
                parameters,
                self.estimator,
                X,
                y,
                self.cv,
                self.scoring
            )
            for index, parameters
            in enumerate(combinations)
        ]

        results = scheduler.worker_pool.execute(
            tasks,
            evaluate_grid_configuration
        )

        self.results_ = results

        best_result = max(
            results,
            key=lambda result: result["score"]
        )

        self.best_params_ = (
            best_result["params"]
        )

        self.best_score_ = (
            best_result["score"]
        )

        self.elapsed_time_ = (
            perf_counter() - start_time
        )

        return self

    def get_results(self):
        """Return all evaluated configurations."""

        return self.results_

    def summary(self):
        """Return a search summary."""

        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "elapsed_time": self.elapsed_time_,
            "total_configurations": len(
                self.results_
            ),
            "max_workers": self.max_workers
        }