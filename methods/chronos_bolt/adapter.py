#!/usr/bin/env python3
"""Portable adapter for the Chronos-Bolt baseline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from methods.common import normalize_legacy_metrics


def build_command(args: argparse.Namespace) -> list[str]:
    source = Path(__file__).resolve().parents[1] / "tsrag" / "source" / "inference_reproduce.py"
    return [
        args.python,
        str(source),
        "--model", "chronos_bolt",
        "--dataset", args.dataset,
        "--gpu", str(args.gpu),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--data-root", str(Path(args.data_root).resolve()),
        "--baseline-root", str(Path(args.artifact_root).resolve() / "baseline_raw_v1"),
        "--output", str(Path(args.output).resolve()),
        "--seed", str(args.seed),
    ]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--dataset", required=True)
    result.add_argument("--data-root", required=True)
    result.add_argument("--artifact-root", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--python", default="python")
    result.add_argument("--gpu", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=256)
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
    normalize_legacy_metrics(
        Path(args.output), benchmark_id=args.benchmark_id, run_id=args.run_id,
        method="chronos_bolt", dataset=args.dataset, seed=args.seed,
        context_length=512, prediction_length=64,
        data_manifest_sha256=args.data_manifest_sha256,
    )


if __name__ == "__main__":
    main()
