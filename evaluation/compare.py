#!/usr/bin/env python3
"""Compare all methods in a canonical variable_metrics.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.io import read_variable_metrics
from evaluation.metrics import benchmark_summaries, dataset_summaries, pairwise_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--baseline", default="chronos_bolt")
    args = parser.parse_args()
    datasets = dataset_summaries(read_variable_metrics(args.input))
    print(json.dumps({
        "benchmark": benchmark_summaries(datasets),
        "comparisons": pairwise_comparison(datasets, args.baseline),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
