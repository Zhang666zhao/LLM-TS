"""Aggregation routines shared by imports and future methods."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Iterable

from evaluation.schema import VariableMetric


def dataset_summaries(rows: Iterable[VariableMetric]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, int, int, int], list[VariableMetric]] = defaultdict(list)
    for row in rows:
        row.validate()
        key = (
            row.benchmark_id, row.method, row.dataset, row.seed,
            row.context_length, row.prediction_length,
        )
        groups[key].append(row)
    summaries = []
    for (benchmark_id, method, dataset, seed, context_length, prediction_length), values in sorted(groups.items()):
        summaries.append({
            "benchmark_id": benchmark_id,
            "method": method,
            "dataset": dataset,
            "seed": seed,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "variables": len(values),
            "test_windows": sum(value.test_windows for value in values),
            "mse": fmean(value.mse for value in values),
            "mae": fmean(value.mae for value in values),
        })
    return summaries


def benchmark_summaries(dataset_rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in dataset_rows:
        key = (
            str(row["benchmark_id"]), str(row["method"]), int(row["seed"]),
            int(row["context_length"]), int(row["prediction_length"]),
        )
        groups[key].append(row)
    return [
        {
            "benchmark_id": benchmark_id,
            "method": method,
            "seed": seed,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "datasets": len(values),
            "mse": fmean(float(value["mse"]) for value in values),
            "mae": fmean(float(value["mae"]) for value in values),
        }
        for (benchmark_id, method, seed, context_length, prediction_length), values in sorted(groups.items())
    ]


def pairwise_comparison(
    summaries: Iterable[dict[str, object]], baseline: str
) -> list[dict[str, object]]:
    rows = list(summaries)
    baseline_rows = {
        (
            str(row["benchmark_id"]), str(row["dataset"]), int(row["seed"]),
            int(row["context_length"]), int(row["prediction_length"]),
        ): row
        for row in rows if row["method"] == baseline
    }
    output = []
    for row in rows:
        if row["method"] == baseline:
            continue
        comparison_key = (
            str(row["benchmark_id"]), str(row["dataset"]), int(row["seed"]),
            int(row["context_length"]), int(row["prediction_length"]),
        )
        reference = baseline_rows.get(comparison_key)
        if reference is None:
            continue
        current_mse, base_mse = float(row["mse"]), float(reference["mse"])
        current_mae, base_mae = float(row["mae"]), float(reference["mae"])
        output.append({
            "benchmark_id": row["benchmark_id"],
            "method": row["method"],
            "baseline": baseline,
            "dataset": row["dataset"],
            "seed": row["seed"],
            "context_length": row["context_length"],
            "prediction_length": row["prediction_length"],
            "mse_delta": current_mse - base_mse,
            "mse_improvement_percent": (base_mse - current_mse) / base_mse * 100,
            "mae_delta": current_mae - base_mae,
            "mae_improvement_percent": (base_mae - current_mae) / base_mae * 100,
        })
    return output
