from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed
)


class ParallelWorkerPool:
    """Process-based worker pool with retry support."""

    def __init__(
        self,
        max_workers=None,
        max_retries=2
    ):
        self.max_workers = max_workers
        self.max_retries = max_retries

    def execute(
        self,
        tasks,
        worker_function
    ):
        """
        Execute tasks in parallel with retry support.

        Failed tasks are retried up to max_retries times.
        """

        task_list = list(tasks)

        results = []
        failed_tasks = []

        with ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:

            futures = {
                executor.submit(
                    worker_function,
                    task
                ): task
                for task in task_list
            }

            for future in as_completed(futures):

                task = futures[future]

                try:

                    result = future.result()

                    results.append(result)

                except Exception as error:

                    failed_tasks.append(
                        (task, error)
                    )

        # Retry failed tasks.
        for task, error in failed_tasks:

            success = False

            for attempt in range(
                1,
                self.max_retries + 1
            ):

                try:

                    with ProcessPoolExecutor(
                        max_workers=1
                    ) as retry_executor:

                        future = retry_executor.submit(
                            worker_function,
                            task
                        )

                        result = future.result()

                    results.append(result)

                    success = True

                    break

                except Exception:

                    continue

            if not success:

                raise RuntimeError(
                    f"Task failed after "
                    f"{self.max_retries} retries: "
                    f"{task}"
                )

        return results