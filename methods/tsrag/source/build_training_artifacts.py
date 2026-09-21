#!/usr/bin/env python3
"""Validate official training parquet files and build the retrieval sequence memmap."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


EXPECTED_RETRIEVAL_ROWS = 2_792_864
EXPECTED_TRAINING_ROWS = 28_013_980
CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 64
SEQUENCE_LENGTH = CONTEXT_LENGTH + PREDICTION_LENGTH
TOP_K = 10


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def fixed_list_numpy(array, expected_length: int) -> np.ndarray:
    lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False)
    if lengths.size == 0 or int(lengths.min()) != expected_length or int(lengths.max()) != expected_length:
        raise ValueError(f"Expected list length {expected_length}, got [{lengths.min()}, {lengths.max()}]")
    values = array.flatten().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(len(array), expected_length)


def build_store(retrieval_path: Path, output_root: Path, force: bool) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    store_path = output_root / "retrieval_sequences_f32.npy"
    manifest_path = output_root / "retrieval_sequences_manifest.json"
    if store_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        array = np.load(store_path, mmap_mode="r")
        if tuple(array.shape) != (EXPECTED_RETRIEVAL_ROWS, SEQUENCE_LENGTH) or array.dtype != np.float32:
            raise ValueError("Existing retrieval store has the wrong shape or dtype")
        if sha256_file(store_path) != manifest["output_sha256"]:
            raise ValueError("Existing retrieval store checksum mismatch")
        return manifest

    parquet = pq.ParquetFile(retrieval_path)
    if parquet.metadata.num_rows != EXPECTED_RETRIEVAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_RETRIEVAL_ROWS} retrieval rows, got {parquet.metadata.num_rows}")
    temporary = store_path.with_suffix(".npy.tmp")
    mmap = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(EXPECTED_RETRIEVAL_ROWS, SEQUENCE_LENGTH)
    )
    offset = 0
    for batch_number, batch in enumerate(parquet.iter_batches(batch_size=4096, columns=["x", "y"])):
        x = fixed_list_numpy(batch.column(0), CONTEXT_LENGTH)
        y = fixed_list_numpy(batch.column(1), PREDICTION_LENGTH)
        end = offset + len(batch)
        mmap[offset:end, :CONTEXT_LENGTH] = x
        mmap[offset:end, CONTEXT_LENGTH:] = y
        offset = end
        if batch_number % 100 == 0:
            print(f"retrieval rows written: {offset}/{EXPECTED_RETRIEVAL_ROWS}", flush=True)
    if offset != EXPECTED_RETRIEVAL_ROWS:
        raise ValueError(f"Wrote {offset} retrieval rows")
    mmap.flush()
    del mmap
    os.replace(temporary, store_path)
    manifest = {
        "schema_version": "retrieval_sequences_f32_v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "source": str(retrieval_path.resolve()),
        "source_size": retrieval_path.stat().st_size,
        "source_rows": EXPECTED_RETRIEVAL_ROWS,
        "output": str(store_path.resolve()),
        "output_size": store_path.stat().st_size,
        "output_shape": [EXPECTED_RETRIEVAL_ROWS, SEQUENCE_LENGTH],
        "output_dtype": "float32",
        "output_sha256": sha256_file(store_path),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def validate_training_data(training_root: Path) -> Dict[str, object]:
    files = list(training_root.iterdir())
    if not files or any(path.suffix != ".parquet" for path in files):
        raise ValueError("Training directory must contain parquet files only")
    total_rows = 0
    min_index = EXPECTED_RETRIEVAL_ROWS
    max_index = -1
    file_records = []
    for path in files:
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        total_rows += rows
        file_min = EXPECTED_RETRIEVAL_ROWS
        file_max = -1
        for batch in parquet.iter_batches(batch_size=32768, columns=["target", "indices", "distances"]):
            target_lengths = pc.list_value_length(batch.column(0)).to_numpy(zero_copy_only=False)
            index_lengths = pc.list_value_length(batch.column(1)).to_numpy(zero_copy_only=False)
            distance_lengths = pc.list_value_length(batch.column(2)).to_numpy(zero_copy_only=False)
            if int(target_lengths.min()) != SEQUENCE_LENGTH or int(target_lengths.max()) != SEQUENCE_LENGTH:
                raise ValueError(f"Invalid target length in {path}")
            if int(index_lengths.min()) < TOP_K or int(distance_lengths.min()) < TOP_K:
                raise ValueError(f"Fewer than {TOP_K} retrievals in {path}")
            indices = batch.column(1).flatten().to_numpy(zero_copy_only=False)
            if indices.size:
                file_min = min(file_min, int(indices.min()))
                file_max = max(file_max, int(indices.max()))
        min_index = min(min_index, file_min)
        max_index = max(max_index, file_max)
        file_records.append({"path": str(path.resolve()), "size": path.stat().st_size, "rows": rows})
        print(f"validated {path.name}: rows={rows} index_range=[{file_min}, {file_max}]", flush=True)
    if total_rows != EXPECTED_TRAINING_ROWS:
        raise ValueError(f"Expected {EXPECTED_TRAINING_ROWS} training rows, got {total_rows}")
    if min_index < 0 or max_index >= EXPECTED_RETRIEVAL_ROWS:
        raise ValueError(f"Retrieval index range [{min_index}, {max_index}] is invalid")
    return {
        "schema_version": "official_pretrain_pairs_ctx512_v1",
        "training_root": str(training_root.resolve()),
        "training_rows": total_rows,
        "files_in_filesystem_order": file_records,
        "retrieval_index_min": min_index,
        "retrieval_index_max": max_index,
        "target_length": SEQUENCE_LENGTH,
        "minimum_retrieval_count": TOP_K,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    official_root = args.data_root / "TS-RAG-Data"
    args.output_root.mkdir(parents=True, exist_ok=True)
    training = validate_training_data(official_root / "pretrain_pairs_ctx512")
    retrieval = build_store(official_root / "retrieval_database_512.parquet", args.output_root, args.force)
    manifest = {"training": training, "retrieval": retrieval}
    atomic_json(args.output_root / "training_data_manifest.json", manifest)
    print(json.dumps({"status": "complete", "manifest": str(args.output_root / 'training_data_manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
