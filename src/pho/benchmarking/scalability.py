import csv
import os


class ScalabilityAnalyzer:
    """Stores and analyzes scalability experiment results."""

    def __init__(self):
        self.results = []

    def add_result(
        self,
        workers,
        execution_time,
        speedup,
        efficiency
    ):
        """Add one scalability measurement."""

        self.results.append({
            "workers": workers,
            "execution_time": execution_time,
            "speedup": speedup,
            "efficiency": efficiency
        })

    def get_results(self):
        """Return all recorded results."""

        return self.results

    def save_csv(self, filepath):
        """Save scalability results to a CSV file."""

        directory = os.path.dirname(filepath)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "workers",
                    "execution_time",
                    "speedup",
                    "efficiency"
                ]
            )

            writer.writeheader()
            writer.writerows(self.results)