from dataclasses import dataclass
from typing import Any, Callable, List

from src.pho.execution.worker_pool import ParallelWorkerPool


@dataclass
class SearchTask:
    """Represents one hyperparameter evaluation task."""

    task_id: int
    parameters: dict


class ParallelScheduler:
    """Creates and schedules hyperparameter evaluation tasks."""

    def __init__(self, max_workers=None):
        self.worker_pool = ParallelWorkerPool(
            max_workers=max_workers
        )

    def create_tasks(self, parameter_combinations):
        """Convert parameter combinations into numbered tasks."""

        return [
            SearchTask(
                task_id=index,
                parameters=params
            )
            for index, params in enumerate(
                parameter_combinations
            )
        ]

    def execute(
        self,
        parameter_combinations,
        worker_function: Callable[[Any], Any]
    ) -> List[Any]:
        """Schedule tasks and execute them in parallel."""

        tasks = self.create_tasks(parameter_combinations)

        return self.worker_pool.execute(
            tasks,
            worker_function
        )