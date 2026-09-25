from src.pho.execution.worker_pool import ParallelWorkerPool


def square(number):
    """Return the square of a number."""
    return number * number


def test_worker_pool():

    pool = ParallelWorkerPool(
        max_workers=2
    )

    results = pool.execute(
        [1, 2, 3, 4],
        square
    )

    assert sorted(results) == [1, 4, 9, 16]