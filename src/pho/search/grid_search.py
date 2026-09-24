from itertools import product
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score


class SequentialGridSearch:
    """
    Sequential hyperparameter search implementation.

    Evaluates every combination in the search space one by one
    and returns the configuration with the best cross-validation score.
    """

    def __init__(self, estimator, param_grid, cv=3, scoring=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

        self.best_params_ = None
        self.best_score_ = None
        self.results_ = []
        self.elapsed_time_ = None

    def _generate_combinations(self):
        """Generate all hyperparameter combinations."""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[key] for key in keys]

        for combination in product(*values):
            yield dict(zip(keys, combination))

    def fit(self, X, y):
        """Run the complete sequential hyperparameter search."""

        start_time = perf_counter()

        best_score = -np.inf
        best_params = None

        for params in self._generate_combinations():

            model = clone(self.estimator)
            model.set_params(**params)

            scores = cross_val_score(
                model,
                X,
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=1
            )

            mean_score = float(np.mean(scores))

            result = {
                "params": params,
                "score": mean_score
            }

            self.results_.append(result)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.elapsed_time_ = perf_counter() - start_time

        self.best_score_ = best_score
        self.best_params_ = best_params

        return self

    def get_results(self):
        """Return all evaluated configurations."""
        return self.results_

    def summary(self):
        """Return a simple search summary."""
        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "elapsed_time": self.elapsed_time_,
            "total_configurations": len(self.results_)
        }