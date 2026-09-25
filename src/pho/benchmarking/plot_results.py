import csv
import os

import matplotlib.pyplot as plt


def load_results(filepath):
    """Load scalability results from CSV."""

    workers = []
    execution_times = []
    speedups = []
    efficiencies = []

    with open(filepath, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            workers.append(int(row["workers"]))
            execution_times.append(float(row["execution_time"]))
            speedups.append(float(row["speedup"]))
            efficiencies.append(float(row["efficiency"]))

    return (
        workers,
        execution_times,
        speedups,
        efficiencies
    )


def create_plots(filepath="results/scalability.csv"):
    """Create scalability visualization plots."""

    (
        workers,
        execution_times,
        speedups,
        efficiencies
    ) = load_results(filepath)

    os.makedirs("results", exist_ok=True)

    # Execution time plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        workers,
        execution_times,
        marker="o"
    )

    plt.xlabel("Number of Workers")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Parallel Execution Time vs Workers")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "results/execution_time.png",
        dpi=300
    )

    plt.close()

    # Speedup plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        workers,
        speedups,
        marker="o"
    )

    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup")
    plt.title("Parallel Speedup vs Workers")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "results/speedup.png",
        dpi=300
    )

    plt.close()

    # Efficiency plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        workers,
        efficiencies,
        marker="o"
    )

    plt.xlabel("Number of Workers")
    plt.ylabel("Efficiency")
    plt.title("Parallel Efficiency vs Workers")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "results/efficiency.png",
        dpi=300
    )

    plt.close()

    print("Plots generated successfully.")
    print("results/execution_time.png")
    print("results/speedup.png")
    print("results/efficiency.png")


if __name__ == "__main__":
    create_plots()