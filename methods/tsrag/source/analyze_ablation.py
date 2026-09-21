#!/usr/bin/env python3
"""Aggregate the fixed seven-hour TS-RAG ablation suite."""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "exchange_rate"]
EXPERIMENTS = [
    "causal_correct", "causal_zero", "causal_shuffle", "causal_repeat_top1", "causal_topk1",
    "uniform_s2021_5k", "raw_softmax_s2021_5k", "full_arm_s2022_10k",
    "distance_aware_s2021_10k", "distance_aware_s2022_10k", "distance_aware_s2023_10k",
]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_evaluation(root: Path, experiment: str) -> Optional[Dict[str, object]]:
    rows: Dict[str, object] = {}
    mse: List[float] = []
    mae: List[float] = []
    elapsed = 0.0
    for dataset in DATASETS:
        metrics = load_json(root / "experiments" / experiment / "evaluation" / dataset / "metrics.json")
        if not metrics or metrics.get("status") != "complete":
            return None
        rows[dataset] = metrics
        mse.append(float(metrics["mse"]))
        mae.append(float(metrics["mae"]))
        elapsed += float(metrics["elapsed_seconds"])
    return {
        "average_mse": float(np.mean(mse)),
        "average_mae": float(np.mean(mae)),
        "elapsed_seconds": elapsed,
        "datasets": rows,
    }


def block_bootstrap_delta(
    correct_path: Path,
    variant_path: Path,
    seed: int = 2021,
    block_size: int = 64,
    repetitions: int = 1000,
) -> Optional[Dict[str, float]]:
    if not correct_path.is_file() or not variant_path.is_file():
        return None
    correct = np.load(correct_path)
    variant = np.load(variant_path)
    if not np.array_equal(correct["sample_id"], variant["sample_id"]):
        raise RuntimeError(f"Sample order mismatch: {correct_path} vs {variant_path}")
    delta = variant["mse"].astype(np.float64) - correct["mse"].astype(np.float64)
    feature = correct["feature_id"].astype(np.int64)
    blocks: List[Tuple[float, int]] = []
    for feature_id in np.unique(feature):
        values = delta[feature == feature_id]
        for start in range(0, len(values), block_size):
            block = values[start : start + block_size]
            blocks.append((float(block.sum()), int(block.size)))
    if not blocks:
        return None
    sums = np.asarray([item[0] for item in blocks])
    counts = np.asarray([item[1] for item in blocks])
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        samples[index] = sums[selected].sum() / counts[selected].sum()
    return {
        "variant_minus_correct_mse": float(delta.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "blocks": len(blocks),
        "bootstrap_repetitions": repetitions,
    }


def paired_harmed_reduction(root: Path) -> Optional[Dict[str, float]]:
    baseline = root / "experiments" / "full_arm_s2022_10k" / "evaluation"
    candidate = root / "experiments" / "distance_aware_s2022_10k" / "evaluation"
    baseline_anchor = root / "anchors" / "chronos_bolt_sample_errors"
    if not baseline_anchor.is_dir():
        return None
    baseline_harmed = 0
    candidate_harmed = 0
    count = 0
    for dataset in DATASETS:
        paths = [
            baseline / dataset / "sample_errors.npz",
            candidate / dataset / "sample_errors.npz",
            baseline_anchor / dataset / "sample_errors.npz",
        ]
        if not all(path.is_file() for path in paths):
            return None
        full, distance, chronos = (np.load(path) for path in paths)
        baseline_harmed += int(np.sum(full["mse"] > chronos["mse"]))
        candidate_harmed += int(np.sum(distance["mse"] > chronos["mse"]))
        count += int(len(full["mse"]))
    return {
        "sample_count": count,
        "full_arm_harmed_fraction": baseline_harmed / count,
        "distance_aware_harmed_fraction": candidate_harmed / count,
        "relative_reduction": (baseline_harmed - candidate_harmed) / max(baseline_harmed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.suite_root.resolve()
    report: Dict[str, object] = {
        "suite": root.name,
        "status": "partial",
        "experiments": {},
        "causal_bootstrap": {},
    }
    complete = 0
    for experiment in EXPERIMENTS:
        evaluation = load_evaluation(root, experiment)
        status = load_json(root / "experiments" / experiment / "status.json") or {"status": "missing"}
        record = {"status": status.get("status", "missing")}
        if evaluation:
            record.update(evaluation)
            record["status"] = "complete"
            complete += 1
        report["experiments"][experiment] = record  # type: ignore[index]

    correct_root = root / "experiments" / "causal_correct" / "evaluation"
    for variant in ("causal_zero", "causal_shuffle", "causal_repeat_top1", "causal_topk1"):
        dataset_results = {}
        for dataset in DATASETS:
            result = block_bootstrap_delta(
                correct_root / dataset / "sample_errors.npz",
                root / "experiments" / variant / "evaluation" / dataset / "sample_errors.npz",
            )
            if result:
                dataset_results[dataset] = result
        report["causal_bootstrap"][variant] = dataset_results  # type: ignore[index]

    distance_values = [
        report["experiments"][f"distance_aware_s{seed}_10k"]  # type: ignore[index]
        for seed in (2021, 2022, 2023)
        if report["experiments"][f"distance_aware_s{seed}_10k"].get("status") == "complete"  # type: ignore[index]
    ]
    if distance_values:
        mse = np.asarray([value["average_mse"] for value in distance_values], dtype=float)
        mae = np.asarray([value["average_mae"] for value in distance_values], dtype=float)
        report["distance_aware_seeds"] = {
            "count": len(distance_values),
            "average_mse_mean": float(mse.mean()),
            "average_mse_std": float(mse.std()),
            "average_mae_mean": float(mae.mean()),
            "average_mae_std": float(mae.std()),
        }
    harmed = paired_harmed_reduction(root)
    if harmed:
        report["harmed_sample_analysis"] = harmed
    report["completed_experiments"] = complete
    report["status"] = "complete" if complete == len(EXPERIMENTS) else "partial"

    markdown = [
        "# TS-RAG seven-hour ablation report", "",
        f"- status: {report['status']}",
        f"- completed experiments: {complete}/{len(EXPERIMENTS)}", "",
        "| Experiment | Status | Average MSE | Average MAE |",
        "|---|---|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        value = report["experiments"][experiment]  # type: ignore[index]
        markdown.append(
            f"| {experiment} | {value.get('status')} | "
            f"{value.get('average_mse', float('nan')):.6f} | {value.get('average_mae', float('nan')):.6f} |"
        )
    markdown.extend(["", "## Causal block-bootstrap deltas", ""])
    for variant, values in report["causal_bootstrap"].items():  # type: ignore[union-attr]
        if not values:
            continue
        mean_delta = float(np.mean([row["variant_minus_correct_mse"] for row in values.values()]))
        markdown.append(f"- {variant}: macro variant-correct MSE = {mean_delta:+.6f}")
    if "distance_aware_seeds" in report:
        value = report["distance_aware_seeds"]  # type: ignore[assignment]
        markdown.extend([
            "", "## Distance-aware seeds", "",
            f"- MSE: {value['average_mse_mean']:.6f} ± {value['average_mse_std']:.6f}",
            f"- MAE: {value['average_mae_mean']:.6f} ± {value['average_mae_std']:.6f}",
        ])
    atomic_write(root / "metrics.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write(root / "summary.md", "\n".join(markdown) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
