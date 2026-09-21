#!/usr/bin/env python3
"""Regenerate benchmark_v1 files from the committed completed workbook."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.import_excel import import_workbook


def main() -> None:
    source = ROOT / "results/benchmark_v1/source/TSRAG_ChronosBolt_variable_metrics.xlsx"
    result = import_workbook(source, ROOT / "results/benchmark_v1")
    print(f"imported {len(result['rows'])} rows from {source.name}; no inference executed")


if __name__ == "__main__":
    main()
