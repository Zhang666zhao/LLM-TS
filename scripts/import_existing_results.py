#!/usr/bin/env python3
"""Regenerate benchmark_v1 files from the committed completed workbook."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.import_excel import import_workbook


def main() -> None:
    source = ROOT / "results/benchmark_v1/source/TSRAG_ChronosBolt_variable_metrics.xlsx"
    dlinear_source = ROOT / "results/benchmark_v1/source/dlinear/variable_metrics.csv"
    result = import_workbook(source, ROOT / "results/benchmark_v1", dlinear_source)
    print(f"imported {len(result['rows'])} rows from completed result sources; no inference executed")


if __name__ == "__main__":
    main()
