from concurrent.futures import ProcessPoolExecutor, as_completed


class ParallelWorkerPool:
    """Process-based worker pool for parallel task execution."""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def execute(self, tasks, worker_function):
        """
        Execute tasks in parallel.

        Parameters
        ----------
        tasks : iterable
            Collection of task inputs.
        worker_function : callable
            Function executed by each worker.

        Returns
        -------
        list
            Results returned by the workers.
        """
        results = []

        with ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:

            futures = [
                executor.submit(worker_function, task)
                for task in tasks
            ]

            for future in as_completed(futures):
                results.append(future.result())

        return results