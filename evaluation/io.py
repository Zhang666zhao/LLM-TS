"""CSV and JSON serialization for the benchmark result contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from evaluation.schema import REQUIRED_COLUMNS, VariableMetric


def write_variable_metrics(path: Path, rows: Iterable[VariableMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def read_variable_metrics(path: Path) -> list[VariableMetric]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = []
        for raw in reader:
            row = VariableMetric(
                benchmark_id=raw["benchmark_id"], method=raw["method"],
                dataset=raw["dataset"], frequency=raw["frequency"],
                feature_id=int(raw["feature_id"]), variable=raw["variable"],
                seed=int(raw["seed"]), context_length=int(raw["context_length"]),
                prediction_length=int(raw["prediction_length"]),
                test_windows=int(raw["test_windows"]), mse=float(raw["mse"]),
                mae=float(raw["mae"]), source=raw["source"],
            )
            row.validate()
            rows.append(row)
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
