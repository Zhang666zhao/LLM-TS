#!/usr/bin/env python3
"""Print the current benchmark comparison from canonical CSV data."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.io import read_variable_metrics
from evaluation.metrics import benchmark_summaries, dataset_summaries, pairwise_comparison


def main() -> None:
    rows = read_variable_metrics(ROOT / "results/benchmark_v1/variable_metrics.csv")
    datasets = dataset_summaries(rows)
    print(json.dumps({
        "benchmark": benchmark_summaries(datasets),
        "comparisons": pairwise_comparison(datasets, "chronos_bolt"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
