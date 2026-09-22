#!/usr/bin/env python3
"""Common method adapter for supervised shared DLinear training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.common import normalize_dlinear_metrics


SETTINGS = {
    "ETTh1": (32, 0.005),
    "ETTh2": (32, 0.05),
    "ETTm1": (8, 0.0001),
    "ETTm2": (32, 0.001),
    "weather": (16, 0.0001),
    "electricity": (16, 0.001),
    "exchange_rate": (8, 0.0005),
}


def build_command(args: argparse.Namespace) -> list[str]:
    batch_size, learning_rate = SETTINGS[args.dataset]
    return [
        args.python,
        str(Path(__file__).resolve().parent / "source/train_dlinear_shared.py"),
        "--dataset", args.dataset,
        "--data-root", str(Path(args.artifact_root).resolve() / "baseline_raw_v1"),
        "--official-repo", str(Path(__file__).resolve().parent / "vendor/ltsf_linear"),
        "--output", str(Path(args.output).resolve()),
        "--gpu", str(args.gpu),
        "--context", "512",
        "--horizon", "64",
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--epochs", "10",
        "--patience", "3",
        "--workers", str(args.num_workers),
        "--seed", str(args.seed),
    ]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", choices=sorted(SETTINGS), required=True)
    result.add_argument("--data-root", required=True)
    result.add_argument("--artifact-root", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--python", default="python")
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=256, help="Ignored; exact reproduced settings are dataset-specific")
    result.add_argument("--num-workers", type=int, default=4)
    result.add_argument("--seed", type=int, default=2021)
    result.add_argument("--benchmark-id", default="benchmark_v1")
    result.add_argument("--run-id", required=True)
    result.add_argument("--data-manifest-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    command = build_command(args)
    if args.dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)
    normalize_dlinear_metrics(
        Path(args.output), benchmark_id=args.benchmark_id, run_id=args.run_id,
        dataset=args.dataset, seed=args.seed,
        data_manifest_sha256=args.data_manifest_sha256,
    )


if __name__ == "__main__":
    main()
