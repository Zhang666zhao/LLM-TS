#!/usr/bin/env python3
"""Create raw CSV views from the official retrieval-augmented CSV files.

The augmented files contain the untouched benchmark columns first, followed by
boundary_idx_*, timestamp_idx_*, and distance_* columns.  This script copies
only the original prefix.  Source files are never modified.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "exchange_rate"]
RETRIEVAL_PREFIXES = ("boundary_idx_", "timestamp_idx_", "distance_")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def source_name(dataset: str) -> str:
    return f"{dataset}_retrieve_{dataset}_512_only_self_train_None.csv"


def create_view(source: Path, destination: Path) -> Dict[str, object]:
    with source.open("r", encoding="utf-8", newline="") as handle:
        header_line = handle.readline().rstrip("\r\n")
    columns = header_line.split(",")
    first_retrieval = next(
        (idx for idx, name in enumerate(columns) if name.startswith(RETRIEVAL_PREFIXES)),
        None,
    )
    if first_retrieval is None:
        raise ValueError(f"No retrieval columns found in {source}")
    raw_columns = columns[:first_retrieval]
    if not raw_columns or raw_columns[0] != "date" or "OT" not in raw_columns:
        raise ValueError(f"Unexpected raw schema in {source}: {raw_columns[:5]}")
    if any(not name.startswith(RETRIEVAL_PREFIXES) for name in columns[first_retrieval:]):
        raise ValueError(f"Retrieval columns are not a contiguous suffix in {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    row_count = 0
    with source.open("r", encoding="utf-8", newline="") as src, temporary.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        for line_number, line in enumerate(src):
            stripped = line.rstrip("\r\n")
            parts = stripped.split(",", first_retrieval)
            if len(parts) != first_retrieval + 1:
                raise ValueError(f"Row {line_number + 1} in {source} has too few columns")
            output = ",".join(parts[:first_retrieval]) + "\n"
            dst.write(output)
            digest.update(output.encode("utf-8"))
            if line_number:
                row_count += 1
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)
    return {
        "source": str(source.resolve()),
        "source_size": source.stat().st_size,
        "output": str(destination.resolve()),
        "output_size": destination.stat().st_size,
        "output_sha256": digest.hexdigest(),
        "rows": row_count,
        "columns": raw_columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()

    augmented_root = args.data_root / "TS-RAG-Data" / "datasets_512"
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest: Dict[str, object] = {
        "schema_version": "baseline_raw_v1",
        "derivation": "original contiguous column prefix before retrieval metadata",
        "datasets": {},
    }
    for dataset in args.datasets:
        source = augmented_root / source_name(dataset)
        destination = args.output_root / f"{dataset}.csv"
        print(f"creating {destination.name} from {source.name}", flush=True)
        manifest["datasets"][dataset] = create_view(source, destination)  # type: ignore[index]

    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps({"status": "complete", "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
