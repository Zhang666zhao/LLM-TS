#!/usr/bin/env python3
"""Launch a registered method through the common adapter interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_paths(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid path config line: {raw}")
        values[key.strip()] = value.strip()
    required = {"data_root", "artifact_root", "run_root", "python"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Missing path settings: {sorted(missing)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / "methods" / args.method / "method.json"
    if not manifest_path.exists():
        raise SystemExit(f"Unknown method: {args.method}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.dataset not in manifest["supported_datasets"]:
        raise SystemExit(f"{args.method} does not declare support for {args.dataset}")
    paths = read_paths(args.paths)
    run_id = args.run_id or f"{args.method}_{args.dataset}_seed{args.seed}"
    output = Path(paths["run_root"]) / run_id / args.dataset
    data_manifest = ROOT / "data/manifests/benchmark_v1.json"
    command = [
        paths["python"], str(ROOT / manifest["adapter"]),
        "--dataset", args.dataset,
        "--data-root", paths["data_root"],
        "--artifact-root", paths["artifact_root"],
        "--output", str(output),
        "--python", paths["python"],
        "--gpu", str(args.gpu),
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--benchmark-id", "benchmark_v1",
        "--run-id", run_id,
        "--data-manifest-sha256", sha256(data_manifest),
    ]
    if args.dry_run:
        command.append("--dry-run")
        print(" ".join(command))
        return
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
