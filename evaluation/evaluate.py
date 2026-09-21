#!/usr/bin/env python3
"""Validate a completed method run against the common result schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED = {
    "schema_version", "benchmark_id", "run_id", "method", "dataset", "seed",
    "context_length", "prediction_length", "metrics", "sample_count",
    "value_count", "code_commit", "data_manifest_sha256", "status",
}


def validate(payload: dict[str, object]) -> None:
    missing = REQUIRED - payload.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    if payload["status"] != "complete":
        raise ValueError("Only complete runs can enter a benchmark")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    for name in ("mse", "mae"):
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid metric {name}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_json", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    validate(payload)
    print(f"valid: {args.metrics_json}")


if __name__ == "__main__":
    main()
