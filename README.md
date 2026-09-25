# PHO — Parallel Hyperparameter Optimization Framework

PHO (Parallel Hyperparameter Optimization) is a Python framework designed to
speed up machine learning hyperparameter search using parallel task execution.

Instead of evaluating every machine learning configuration one after another,
PHO distributes independent hyperparameter configurations across multiple
worker processes.

The framework supports:

- Sequential Grid Search
- Parallel Grid Search
- Parallel Random Search
- Task-based parallel execution
- Multiprocessing worker pool
- Parallel task scheduling
- Retry and failure handling
- Benchmarking
- Speedup measurement
- Parallel efficiency measurement
- Scalability analysis
- Result visualization
- Command-line execution
- Automated testing

---

## 1. Problem Statement

Machine learning models often require testing many combinations of
hyperparameters such as:

- Number of trees
- Maximum tree depth
- Minimum samples per split
- Learning rate
- Batch size

Traditional hyperparameter search evaluates these configurations
sequentially.

For example:

```text
Configuration 1 → Configuration 2 → Configuration 3 → Configuration 4

This can become slow when the search space contains many configurations.

However, individual configurations are usually independent of each other.
Therefore, they can be executed simultaneously on multiple CPU workers.

PHO uses this task-level parallelism to distribute hyperparameter evaluation
across multiple processes.

Instead of:

       Configuration 1
              ↓
       Configuration 2
              ↓
       Configuration 3
              ↓
       Configuration 4

PHO can execute:

Worker 1 → Configuration 1
Worker 2 → Configuration 2
Worker 3 → Configuration 3
Worker 4 → Configuration 4
2. Main Objective

The main objective of PHO is to develop a scalable parallel framework for
machine learning hyperparameter optimization and experimentally analyze the
benefits of parallel execution.

The framework measures:

Execution time
Speedup
Parallel efficiency
Scalability

The project also compares sequential and parallel execution.

3. Architecture

The framework is organized into several components:

                    PHO Framework
                         |
          +--------------+--------------+
          |                             |
     Search Layer                  CLI Interface
          |                             |
   +------+------+                +-----+------+
   |             |                |            |
Grid Search  Random Search    grid-search  benchmark
   |             |
   +------+------+
          |
    Task Scheduler
          |
     Worker Pool
          |
   +------+------+------+------+ 
   |      |      |      |
Worker  Worker  Worker  Worker
   |      |      |      |
 Task    Task    Task    Task
   |      |      |      |
   +------+------+
          |
   ML Model Evaluation
          |
   Results + Metrics
          |
   Benchmarking
          |
   Speedup / Efficiency
          |
      Visualization
4. Project Structure
pho_framework/
│
├── src/
│   └── pho/
│       ├── cli.py
│       │
│       ├── search/
│       │   ├── grid_search.py
│       │   ├── parallel_grid_search.py
│       │   └── random_search.py
│       │
│       ├── execution/
│       │   ├── worker_pool.py
│       │   └── scheduler.py
│       │
│       ├── benchmarking/
│       │   ├── benchmark.py
│       │   ├── scalability.py
│       │   └── plot_results.py
│       │
│       └── utils/
│           ├── logger.py
│           └── config.py
│
├── experiments/
│   ├── sequential_experiment.py
│   ├── parallel_experiment.py
│   ├── benchmark_experiment.py
│   ├── random_search_experiment.py
│   └── parallel_grid_search_experiment.py
│
├── tests/
│   ├── test_search.py
│   ├── test_worker_pool.py
│   ├── test_scheduler.py
│   └── test_retry.py
│
├── notebooks/
│   └── PHO_Demo.ipynb
│
├── results/
│   ├── scalability.csv
│   ├── execution_time.png
│   ├── speedup.png
│   └── efficiency.png
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
5. Technologies Used
Python
NumPy
Pandas
Scikit-learn
Matplotlib
Multiprocessing
ProcessPoolExecutor
Pytest
Jupyter Notebook
6. Installation
Step 1 — Clone the repository
git clone https://github.com/saniya-malik/pho_framework.git

Go into the project:

cd pho_framework
Step 2 — Install dependencies
python -m pip install -r requirements.txt
Step 3 — Install PHO

Install the project in editable mode:

python -m pip install -e .

After installation, the pho command becomes available.

Check it using:

pho --help

You should see:

usage: pho [-h] {grid-search,random-search,benchmark} ...

PHO - Parallel Hyperparameter Optimization Framework
7. Running PHO

PHO provides three main commands.

A. Parallel Grid Search

Run:

pho grid-search

This executes the parallel grid search experiment.

It evaluates multiple hyperparameter configurations using multiple
worker processes.

B. Parallel Random Search

Run:

pho random-search

This executes the parallel random search experiment.

Random search samples configurations from the defined search space and
evaluates them in parallel.

C. Benchmark

Run:

pho benchmark

This compares sequential and parallel execution.

The benchmark measures:

Execution Time
Speedup
Efficiency
Scalability

The benchmark also saves the results to:

results/scalability.csv
8. Understanding the Benchmark

The benchmark compares different numbers of workers.

Example:

Workers: 1
Parallel execution time: 126.72 seconds
Speedup: 0.8590
Efficiency: 0.8590

Workers: 2
Parallel execution time: 91.56 seconds
Speedup: 1.1890
Efficiency: 0.5945

Workers: 4
Parallel execution time: 64.42 seconds
Speedup: 1.6897
Efficiency: 0.4224

The exact execution time can change depending on the computer,
CPU load, background processes and operating system.

9. Performance Metrics
Execution Time

The amount of time required to complete the hyperparameter search.

Execution Time = Time taken by the search

Lower execution time means the computation completed faster.

Speedup

Speedup compares sequential execution with parallel execution.

Speedup = Sequential Time / Parallel Time

For example:

Sequential = 108.86 seconds
Parallel   = 64.42 seconds

Speedup = 108.86 / 64.42
        ≈ 1.69

A speedup of 1.69 means the measured parallel run completed about
1.69 times as fast as the sequential baseline.

Parallel Efficiency

Efficiency measures how effectively the available workers are being used.

Efficiency = Speedup / Number of Workers

For example:

Speedup = 1.69
Workers = 4

Efficiency = 1.69 / 4
           ≈ 0.42
           ≈ 42%
10. Scalability

PHO tests the framework using different numbers of workers.

For example:

1 Worker
   ↓
2 Workers
   ↓
4 Workers

The purpose is to observe how execution time, speedup and efficiency change
as the amount of parallelism increases.

The measured results are stored in:

results/scalability.csv
11. Visualizing Results

PHO generates three graphs:

results/execution_time.png
results/speedup.png
results/efficiency.png

They represent:

Execution Time

Shows how execution time changes with the number of workers.

Speedup

Shows the performance improvement obtained through parallel execution.

Efficiency

Shows how effectively the worker processes are being utilized.

To regenerate the graphs:

python src/pho/benchmarking/plot_results.py
12. Jupyter Notebook Demo

A Jupyter Notebook can be used to demonstrate the framework interactively.

Start Jupyter from the project directory:

jupyter notebook

Then open:

notebooks/PHO_Demo.ipynb

The notebook can display:

Dataset
Machine learning model
Hyperparameter search space
Sequential search
Parallel search
Execution-time comparison
Speedup
Efficiency
Scalability graphs

This provides a visual demonstration of how PHO works.

13. Running Tests

PHO includes automated tests for the main components.

Run:

pytest -v

The test suite covers:

Sequential search
Parallel grid search
Parallel random search
Worker pool
Task scheduler
Retry and failure handling

Example:

6 passed
14. Fault Handling

PHO includes retry support in the worker pool.

If a task fails, the framework attempts to execute the task again.

If the task continues to fail after the configured number of retries,
the framework reports the failure.

This helps demonstrate fault handling in the parallel execution layer.

15. Key Features
Feature	Description
Grid Search	Exhaustive hyperparameter evaluation
Random Search	Random sampling of configurations
Parallel Execution	Multiple configurations execute simultaneously
Worker Pool	Manages parallel worker processes
Task Scheduler	Creates and distributes search tasks
Retry Handling	Retries failed tasks
Benchmarking	Measures performance
Speedup	Measures parallel performance gain
Efficiency	Measures worker utilization
Scalability	Tests different worker counts
Visualization	Generates performance graphs
CLI	Provides simple command-line execution
Testing	Automated test suite
16. Advantages
Reduces hyperparameter search execution time for suitable workloads
Utilizes multiple CPU cores
Supports different search strategies
Provides performance measurements
Provides scalability analysis
Includes fault/retry handling
Can be extended with additional optimization strategies
Provides both CLI and notebook-based demonstrations
17. Limitations
Parallel performance depends on available CPU resources.
Process creation and communication introduce overhead.
Increasing the number of workers does not guarantee linear speedup.
Efficiency can decrease as worker count increases.
Very small workloads may not benefit significantly from parallelization.
The current framework focuses primarily on CPU-based task parallelism.
18. Future Enhancements

Possible future improvements include:

Bayesian Optimization
Genetic Algorithm-based optimization
Adaptive scheduling
Dynamic worker allocation
Distributed execution across multiple machines
GPU-based evaluation
Advanced fault recovery
Result caching
Early stopping
Resource-aware scheduling
Web-based monitoring dashboard
19. Project Outcome

PHO demonstrates how task-level parallelism can be applied to machine learning
hyperparameter optimization.

The project provides an experimental comparison between sequential and
parallel execution and measures the resulting execution time, speedup,
efficiency and scalability.

The framework also demonstrates practical parallel computing concepts such as:

Task decomposition
Process-based parallelism
Worker pools
Task scheduling
Asynchronous task completion
Fault handling
Performance measurement
Scalability analysis
20. Quick Start

For the shortest way to run the project:

git clone https://github.com/saniya-malik/pho_framework.git
cd pho_framework
python -m pip install -r requirements.txt
python -m pip install -e .
pho --help

Then run:

pho grid-search

or:

pho random-search

or:

pho benchmark

For graphs:

python src/pho/benchmarking/plot_results.py

For Jupyter:

jupyter notebook
