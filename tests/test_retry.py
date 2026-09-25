import pytest

from src.pho.execution.worker_pool import ParallelWorkerPool


def failing_worker(task):
    """Worker that always fails."""
    raise ValueError(
        f"Intentional failure: {task}"
    )


def test_retry_failure():

    pool = ParallelWorkerPool(
        max_workers=2,
        max_retries=2
    )

    with pytest.raises(RuntimeError):

        pool.execute(
            [1],
            failing_worker
        )