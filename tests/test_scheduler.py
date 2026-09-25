from src.pho.execution.scheduler import ParallelScheduler


def double_task(task):
    """Double the task value."""
    return task.parameters["value"] * 2


def test_scheduler():

    scheduler = ParallelScheduler(
        max_workers=2
    )

    parameters = [
        {"value": 1},
        {"value": 2},
        {"value": 3},
        {"value": 4}
    ]

    results = scheduler.execute(
        parameters,
        double_task
    )

    values = sorted(results)

    assert values == [2, 4, 6, 8]