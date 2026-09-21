#!/usr/bin/env python3
"""Combine per-dataset reproduction metrics into JSON and Markdown reports."""

import argparse
import json
from pathlib import Path
from typing import Dict


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "exchange_rate"]
TARGETS = {
    "tsrag": {
        "ETTh1": (0.3557, 0.3624), "ETTh2": (0.2451, 0.2982),
        "ETTm1": (0.2906, 0.3114), "ETTm2": (0.1466, 0.2231),
        "weather": (0.1454, 0.1771), "electricity": (0.1120, 0.2002),
        "exchange_rate": (0.0627, 0.1718),
    },
    "chronos_bolt": {
        "ETTh1": (0.3616, 0.3650), "ETTh2": (0.2517, 0.2992),
        "ETTm1": (0.3109, 0.3185), "ETTm2": (0.1487, 0.2236),
        "weather": (0.1525, 0.1825), "electricity": (0.1132, 0.2004),
        "exchange_rate": (0.0673, 0.1780),
    },
}


def load_run(root: Path, model: str) -> Dict[str, Dict[str, object]]:
    result = {}
    for dataset in DATASETS:
        path = root / dataset / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        metrics = json.loads(path.read_text())
        if metrics.get("status") != "complete":
            raise RuntimeError(f"Incomplete metrics: {path}: {metrics.get('status')}")
        if metrics.get("model") != model or metrics.get("dataset") != dataset:
            raise RuntimeError(f"Identity mismatch in {path}")
        result[dataset] = metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsrag-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()

    runs = {
        "tsrag": load_run(args.tsrag_run, "tsrag"),
        "chronos_bolt": load_run(args.baseline_run, "chronos_bolt"),
    }
    report: Dict[str, object] = {"tolerance": args.tolerance, "models": {}}
    markdown = [
        "# TS-RAG inference reproduction",
        "",
        "| Model | Dataset | Reproduced MSE | Paper MSE | Δ MSE | Reproduced MAE | Paper MAE | Δ MAE | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    all_pass = True
    for model, model_runs in runs.items():
        rows = {}
        mse_values = []
        mae_values = []
        for dataset in DATASETS:
            metrics = model_runs[dataset]
            mse = float(metrics["mse"])
            mae = float(metrics["mae"])
            paper_mse, paper_mae = TARGETS[model][dataset]
            delta_mse = abs(mse - paper_mse)
            delta_mae = abs(mae - paper_mae)
            passed = delta_mse <= args.tolerance and delta_mae <= args.tolerance
            all_pass = all_pass and passed
            mse_values.append(mse)
            mae_values.append(mae)
            rows[dataset] = {
                "mse": mse, "mae": mae, "paper_mse": paper_mse, "paper_mae": paper_mae,
                "mse_abs_delta": delta_mse, "mae_abs_delta": delta_mae, "within_tolerance": passed,
                "elapsed_seconds": metrics["elapsed_seconds"], "peak_vram_bytes": metrics["peak_vram_bytes"],
            }
            markdown.append(
                f"| {model} | {dataset} | {mse:.4f} | {paper_mse:.4f} | {delta_mse:.4f} | "
                f"{mae:.4f} | {paper_mae:.4f} | {delta_mae:.4f} | {'yes' if passed else 'no'} |"
            )
        report["models"][model] = {  # type: ignore[index]
            "datasets": rows,
            "average_mse": sum(mse_values) / len(mse_values),
            "average_mae": sum(mae_values) / len(mae_values),
        }

    tsrag_avg = report["models"]["tsrag"]  # type: ignore[index]
    baseline_avg = report["models"]["chronos_bolt"]  # type: ignore[index]
    report["relative_improvement"] = {
        "mse_percent": 100.0 * (baseline_avg["average_mse"] - tsrag_avg["average_mse"]) / baseline_avg["average_mse"],
        "mae_percent": 100.0 * (baseline_avg["average_mae"] - tsrag_avg["average_mae"]) / baseline_avg["average_mae"],
    }
    report["all_within_tolerance"] = all_pass
    markdown.extend([
        "",
        f"- TS-RAG average: MSE {tsrag_avg['average_mse']:.4f}, MAE {tsrag_avg['average_mae']:.4f}",
        f"- Chronos-Bolt average: MSE {baseline_avg['average_mse']:.4f}, MAE {baseline_avg['average_mae']:.4f}",
        f"- Relative improvement: MSE {report['relative_improvement']['mse_percent']:.2f}%, MAE {report['relative_improvement']['mae_percent']:.2f}%",  # type: ignore[index]
        f"- All dataset metrics within ±{args.tolerance:.3f}: {'yes' if all_pass else 'no'}",
        "",
    ])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reproduction_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output / "summary.md").write_text("\n".join(markdown))
    for model, run_root in (("tsrag", args.tsrag_run), ("chronos_bolt", args.baseline_run)):
        model_report = report["models"][model]  # type: ignore[index]
        first_metrics = runs[model][DATASETS[0]]
        aggregate = {
            "run_name": run_root.name,
            "status": "keep",
            "metric_name": "mse",
            "metric_direction": "min",
            "metric_value": model_report["average_mse"],
            "secondary_metrics": {"mae": model_report["average_mae"]},
            "git_commit": first_metrics["git_commit"],
            "data_commit": first_metrics["manifest_sha256"],
            "datasets": model_report["datasets"],
            "all_within_tolerance": all(
                row["within_tolerance"] for row in model_report["datasets"].values()
            ),
            "training_allowed": False,
        }
        (run_root / "metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
