#!/usr/bin/env python3
"""Summarize official shared DLinear runs and compare frozen TS-RAG baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "exchange_rate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    variable_frames = []
    for dataset in DATASETS:
        path = args.run_root / dataset / "metrics.json"
        if not path.is_file():
            rows.append({"dataset": dataset, "status": "missing"})
            continue
        metrics = json.loads(path.read_text())
        row = {
            "dataset": dataset,
            "status": metrics["status"],
            "dlinear_mse": metrics.get("mse"),
            "dlinear_mae": metrics.get("mae"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
            "peak_vram_bytes": metrics.get("peak_vram_bytes"),
            "window_count": metrics.get("window_count"),
        }
        for model, directory in (
            ("tsrag", "20260817_TSRAG_official_inference"),
            ("chronos_bolt", "20260817_ChronosBolt_baseline_inference"),
        ):
            baseline = args.project_root / "runs" / directory / dataset / "metrics.json"
            if baseline.is_file():
                value = json.loads(baseline.read_text())
                row[f"{model}_mse"] = value.get("mse")
                row[f"{model}_mae"] = value.get("mae")
                if metrics.get("mse") is not None:
                    row[f"dlinear_minus_{model}_mse"] = metrics["mse"] - value["mse"]
                if metrics.get("mae") is not None:
                    row[f"dlinear_minus_{model}_mae"] = metrics["mae"] - value["mae"]
        rows.append(row)
        variable_path = args.run_root / dataset / "variable_metrics.csv"
        if variable_path.is_file():
            frame = pd.read_csv(variable_path)
            frame.insert(0, "dataset", dataset)
            variable_frames.append(frame)
    results = pd.DataFrame(rows)
    results.to_csv(args.run_root / "dataset_metrics.csv", index=False)
    if variable_frames:
        pd.concat(variable_frames, ignore_index=True).to_csv(args.run_root / "variable_metrics.csv", index=False)
    complete = results[results.status == "complete"]
    aggregate = {
        "status": "complete" if len(complete) == len(DATASETS) else "partial",
        "completed_datasets": int(len(complete)),
        "expected_datasets": len(DATASETS),
        "average_mse": float(complete.dlinear_mse.mean()) if len(complete) else None,
        "average_mae": float(complete.dlinear_mae.mean()) if len(complete) else None,
        "datasets": rows,
    }
    (args.run_root / "metrics.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n")
    markdown = [
        "# Official shared DLinear, context 512 / prediction 64",
        "",
        f"- status: {aggregate['status']}",
        f"- completed: {aggregate['completed_datasets']}/{aggregate['expected_datasets']}",
        f"- macro average MSE: {aggregate['average_mse']}",
        f"- macro average MAE: {aggregate['average_mae']}",
        "",
        "| Dataset | DLinear MSE | DLinear MAE | TS-RAG MSE | Chronos-Bolt MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['dataset']} | {row.get('dlinear_mse', '')} | {row.get('dlinear_mae', '')} | "
            f"{row.get('tsrag_mse', '')} | {row.get('chronos_bolt_mse', '')} |"
        )
    (args.run_root / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
