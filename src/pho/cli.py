import argparse
import os
import subprocess
import sys


def run_experiment(filename):
    """Run an experiment from the experiments directory."""

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    experiment_path = os.path.join(
        project_root,
        "experiments",
        filename
    )

    subprocess.run(
        [sys.executable, experiment_path],
        check=True
    )


def main():
    """PHO command-line interface."""

    parser = argparse.ArgumentParser(
        prog="pho",
        description=(
            "PHO - Parallel Hyperparameter "
            "Optimization Framework"
        )
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    subparsers.add_parser(
        "grid-search",
        help="Run parallel grid search."
    )

    subparsers.add_parser(
        "random-search",
        help="Run parallel random search."
    )

    subparsers.add_parser(
        "benchmark",
        help="Run scalability benchmark."
    )

    args = parser.parse_args()

    if args.command == "grid-search":

        run_experiment(
            "parallel_grid_search_experiment.py"
        )

    elif args.command == "random-search":

        run_experiment(
            "random_search_experiment.py"
        )

    elif args.command == "benchmark":

        run_experiment(
            "benchmark_experiment.py"
        )

    else:

        parser.print_help()


if __name__ == "__main__":
    main()