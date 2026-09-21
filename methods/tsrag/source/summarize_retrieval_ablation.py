#!/usr/bin/env python3
"""Aggregate TS-RAG retrieval ablations, diagnostics, and paired block CIs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "exchange_rate"]
TOPK = {"chronos_t5_k1_s2021": 1, "chronos_t5_k5_s2021": 5, "chronos_t5_k10_s2021": 10,
        "chronos_t5_k15_s2021": 15, "chronos_t5_k20_s2021": 20}
METHODS = {"chronos_t5_k10_s2021": "chronos_t5", "random_k10_s2021": "random",
           "abs_pearson_k10_s2021": "pearson", "chronos_bolt_embed_k10_s2021": "bolt",
           "qwen3_text_embed_k10_s2021": "qwen"}
EXPERIMENTS = list(TOPK) + [name for name in METHODS if name not in TOPK]
ANCHOR = "chronos_t5_k10_s2021"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], default: str = "experiment") -> None:
    keys = list(rows[0]) if rows else [default]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {key: np.asarray(source[key]) for key in source.files}


def paired_block_ci(anchor: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> tuple[float, float, float, int]:
    if not np.array_equal(anchor["sample_id"], candidate["sample_id"]):
        raise ValueError("sample order mismatch")
    delta = candidate["mse"].astype(np.float64) - anchor["mse"].astype(np.float64)
    features = candidate["feature_id"].astype(np.int64)
    block_ids = np.empty(len(delta), dtype=np.int64)
    next_block = 0
    for feature in np.unique(features):
        positions = np.flatnonzero(features == feature)
        block_ids[positions] = next_block + np.arange(len(positions)) // 64
        next_block = int(block_ids[positions[-1]]) + 1
    sums = np.bincount(block_ids, weights=delta)
    counts = np.bincount(block_ids)
    rng = np.random.default_rng(2021)
    draws = rng.integers(0, len(sums), size=(2000, len(sums)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(delta.mean()), float(low), float(high), len(sums)


def training_diagnostics(root: Path, experiment: str) -> dict:
    pointer = root / "experiments" / experiment / "selected_train_dir.txt"
    if not pointer.is_file():
        return {"training_seconds": None, "training_peak_vram_bytes": None, "trainable_parameters": None}
    train_root = Path(pointer.read_text().strip())
    metrics_path = train_root / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    peak = 0
    log_path = train_root / "logs" / "train.jsonl"
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            try:
                values = json.loads(line).get("peak_vram_bytes") or []
                peak = max([peak] + [int(value) for value in values])
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
    return {"training_seconds": metrics.get("elapsed_seconds"), "training_peak_vram_bytes": peak or None,
            "trainable_parameters": metrics.get("trainable_parameters")}


def view_diagnostics(root: Path, method: str, dataset: str, sample_count: int) -> dict:
    views = root.parent.parent / "artifacts" / "retrieval_ablation_v2" / "test_views"
    method = "official" if method == "chronos_t5" else method
    current_root, official_root = views / method / dataset, views / "official" / dataset
    result = {"retrieval_overlap_with_chronos_t5": None, "mean_absolute_pearson": None,
              "pearson_negative_fraction": None}
    if (current_root / "ARTIFACT_COMPLETE").is_file() and (official_root / "ARTIFACT_COMPLETE").is_file():
        current = np.load(current_root / "timestamp_indices_i32.npy", mmap_mode="r")
        official = np.load(official_root / "timestamp_indices_i32.npy", mmap_mode="r")
        rows = min(sample_count, len(current), len(official))
        overlap = (np.asarray(current[:rows])[:, :, None] == np.asarray(official[:rows])[:, None, :]).any(axis=2).mean()
        result["retrieval_overlap_with_chronos_t5"] = float(overlap)
        if method == "pearson":
            distance = np.asarray(np.load(current_root / "distances_f32.npy", mmap_mode="r")[:rows])
            signs = np.asarray(np.load(current_root / "signs_i8.npy", mmap_mode="r")[:rows])
            result["mean_absolute_pearson"] = float(np.clip(1.0 - distance, 0.0, 1.0).mean())
            result["pearson_negative_fraction"] = float((signs < 0).mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "topk"), default="all")
    args = parser.parse_args()
    root = args.suite_root.resolve()
    experiments = list(TOPK) if args.scope == "topk" else EXPERIMENTS
    expected_evaluations = len(experiments) * len(DATASETS)
    records, failures, samples = [], [], {}
    for experiment in experiments:
        for dataset in DATASETS:
            eval_root = root / "experiments" / experiment / "evaluation" / dataset
            metrics_path, sample_path = eval_root / "metrics.json", eval_root / "sample_errors.npz"
            if not metrics_path.is_file():
                failures.append({"experiment": experiment, "dataset": dataset, "reason": "missing metrics"}); continue
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("status") != "complete":
                failures.append({"experiment": experiment, "dataset": dataset, "reason": metrics.get("status")}); continue
            if not sample_path.is_file():
                failures.append({"experiment": experiment, "dataset": dataset, "reason": "missing sample errors"}); continue
            samples[(experiment, dataset)] = load_npz(sample_path)
            records.append({"experiment": experiment, "dataset": dataset, "mse": float(metrics["mse"]),
                            "mae": float(metrics["mae"]), "sample_count": int(metrics["sample_count"]),
                            "inference_seconds": float(metrics["elapsed_seconds"]),
                            "inference_peak_vram_bytes": int(metrics["peak_vram_bytes"])})
    lookup = {(row["experiment"], row["dataset"]): row for row in records}
    delta_rows, bootstrap_rows, diagnostic_rows = [], [], []
    for record in records:
        experiment, dataset = record["experiment"], record["dataset"]
        anchor_record = lookup.get((ANCHOR, dataset))
        delta_rows.append({**record,
            "mse_delta_vs_k10": record["mse"] - anchor_record["mse"] if anchor_record else None,
            "mae_delta_vs_k10": record["mae"] - anchor_record["mae"] if anchor_record else None})
        if anchor_record:
            try:
                mean, low, high, blocks = paired_block_ci(samples[(ANCHOR, dataset)], samples[(experiment, dataset)])
                bootstrap_rows.append({"experiment": experiment, "dataset": dataset, "mse_delta": mean,
                    "ci95_low": low, "ci95_high": high, "block_size": 64, "blocks": blocks})
            except ValueError as error:
                failures.append({"experiment": experiment, "dataset": dataset, "reason": str(error)})
        method, array = METHODS.get(experiment, "chronos_t5"), samples[(experiment, dataset)]
        diagnostic_rows.append({"experiment": experiment, "dataset": dataset, "retrieval_method": method,
            **view_diagnostics(root, method, dataset, len(array["mse"])),
            "future_mean_mse": float(np.nanmean(array["neighbor_future_mse"])),
            "future_oracle_mse": float(np.nanmean(array.get("neighbor_future_oracle_mse", np.full(1, np.nan)))),
            "neighbor_diversity": float(np.nanmean(array["neighbor_agreement"])),
            "gate_entropy": float(np.nanmean(array["alpha_entropy"])),
            "gate_top1_weight": float(np.nanmean(array["alpha_top1"])),
            "gate_query_weight": float(np.nanmean(array["alpha_query"]))})
    summaries = []
    for experiment in experiments:
        values = [record for record in records if record["experiment"] == experiment]
        if values:
            summaries.append({"experiment": experiment, "average_mse": sum(v["mse"] for v in values) / len(values),
                "average_mae": sum(v["mae"] for v in values) / len(values), "datasets_complete": len(values),
                "training_kind": "topk" if experiment in TOPK else "retrieval", "k": TOPK.get(experiment, 10),
                "retrieval_method": METHODS.get(experiment, "chronos_t5"),
                "inference_seconds": sum(v["inference_seconds"] for v in values), **training_diagnostics(root, experiment)})
    anchor_summary = next((row for row in summaries if row["experiment"] == ANCHOR), None)
    for row in summaries:
        row["mse_delta_vs_k10"] = row["average_mse"] - anchor_summary["average_mse"] if anchor_summary else None
        row["mae_delta_vs_k10"] = row["average_mae"] - anchor_summary["average_mae"] if anchor_summary else None
    write_csv(root / "topk_results.csv", sorted((r for r in summaries if r["experiment"] in TOPK), key=lambda r: r["k"]))
    write_csv(root / "retrieval_results.csv", [r for r in summaries if r["experiment"] in METHODS])
    write_csv(root / "dataset_deltas.csv", delta_rows); write_csv(root / "retrieval_diagnostics.csv", diagnostic_rows)
    write_csv(root / "bootstrap_ci.csv", bootstrap_rows); write_csv(root / "failures.csv", failures, "reason")
    payload = {"status": "complete" if not failures and len(records) == expected_evaluations else "partial",
        "scope": args.scope,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(), "experiments": summaries,
        "bootstrap": bootstrap_rows, "failures": failures}
    atomic_json(root / "metrics.json", payload)
    lines = ["# TS-RAG Retrieval Ablation", "", f"- status: {payload['status']}",
        f"- complete evaluations: {len(records)}/{expected_evaluations}",
        "- paired CI: feature-stratified contiguous 64-sample blocks, 2,000 bootstrap draws", "",
        "| experiment | average MSE | average MAE | ΔMSE vs K10 |", "|---|---:|---:|---:|"]
    for row in summaries:
        delta = row["mse_delta_vs_k10"]
        lines.append(f"| {row['experiment']} | {row['average_mse']:.6f} | {row['average_mae']:.6f} | "
                     f"{f'{delta:+.6f}' if delta is not None else 'n/a'} |")
    if failures:
        lines += ["", "## Failures", ""] + [f"- {x['experiment']}/{x['dataset']}: {x['reason']}" for x in failures]
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
