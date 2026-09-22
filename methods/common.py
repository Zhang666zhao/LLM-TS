"""Shared adapter helpers for normalizing native method outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


def normalize_legacy_metrics(
    output: Path,
    *,
    benchmark_id: str,
    run_id: str,
    method: str,
    dataset: str,
    seed: int,
    context_length: int,
    prediction_length: int,
    data_manifest_sha256: str,
) -> None:
    metrics_path = output / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Method completed without {metrics_path}")
    native = json.loads(metrics_path.read_text(encoding="utf-8"))
    if native.get("status") != "complete":
        raise RuntimeError(f"Method result is not complete: {native.get('status')}")
    native_path = output / "native_metrics.json"
    native_path.write_text(json.dumps(native, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    normalized = {
        "schema_version": "1.0",
        "benchmark_id": benchmark_id,
        "run_id": run_id,
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "metrics": {"mse": native["mse"], "mae": native["mae"]},
        "sample_count": native["sample_count"],
        "value_count": native["value_count"],
        "code_commit": native["git_commit"],
        "data_manifest_sha256": data_manifest_sha256,
        "status": "complete",
        "native_result": native_path.name,
    }
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, metrics_path)


def normalize_dlinear_metrics(
    output: Path,
    *,
    benchmark_id: str,
    run_id: str,
    dataset: str,
    seed: int,
    data_manifest_sha256: str,
) -> None:
    metrics_path = output / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Method completed without {metrics_path}")
    native = json.loads(metrics_path.read_text(encoding="utf-8"))
    if native.get("status") != "complete":
        raise RuntimeError(f"Method result is not complete: {native.get('status')}")
    native_path = output / "native_metrics.json"
    native_path.write_text(json.dumps(native, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    code_commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    normalized = {
        "schema_version": "1.0",
        "benchmark_id": benchmark_id,
        "run_id": run_id,
        "method": "dlinear",
        "dataset": dataset,
        "seed": seed,
        "context_length": int(native["context_length"]),
        "prediction_length": int(native["prediction_length"]),
        "metrics": {"mse": native["mse"], "mae": native["mae"]},
        "sample_count": int(native["sample_count_tsrag_semantics"]),
        "value_count": int(native["value_count"]),
        "code_commit": code_commit,
        "data_manifest_sha256": data_manifest_sha256,
        "status": "complete",
        "training_regime": "supervised_per_dataset",
        "native_result": native_path.name,
    }
    temporary = metrics_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, metrics_path)
