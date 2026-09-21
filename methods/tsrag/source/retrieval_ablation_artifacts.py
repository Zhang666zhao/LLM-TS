#!/usr/bin/env python3
"""Build immutable retrieval-ablation artifacts without modifying official data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import time
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Iterator, Sequence

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


SEED = 2021
CONTEXT = 512
PREDICTION = 64
SEQUENCE = 576
DATABASE_ROWS = 2_792_864
DEFAULT_SCHEDULE_ROWS = 2_560_000
TOP_K = 10
SEARCH_K = 64
MAX_SEARCH_K = 1024
EMBED_DIM = 768


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def complete(path: Path, payload: dict) -> None:
    payload = {**payload, "completed_at": now_iso()}
    atomic_json(path / "manifest.json", payload)
    temporary = path / "ARTIFACT_COMPLETE.tmp"
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path / "ARTIFACT_COMPLETE")


def ranges(total: int, parts: int) -> list[tuple[int, int]]:
    return [(total * part // parts, total * (part + 1) // parts) for part in range(parts)]


def open_atomic_memmap(path: Path, dtype, shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    return array, temporary


def finish_memmap(array, temporary: Path, final: Path) -> None:
    array.flush()
    del array
    os.replace(temporary, final)


def fingerprint_rows(values: np.ndarray, output: Path, batch_size: int = 8192) -> None:
    result, temporary = open_atomic_memmap(output, np.uint64, (len(values),))
    offset_basis = np.uint64(1469598103934665603)
    prime = np.uint64(1099511628211)
    for begin in range(0, len(values), batch_size):
        end = min(len(values), begin + batch_size)
        bits = np.ascontiguousarray(values[begin:end], dtype=np.float32).view(np.uint32)
        hashes = np.full(end - begin, offset_basis, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for column in range(bits.shape[1]):
                hashes ^= bits[:, column].astype(np.uint64)
                hashes *= prime
        result[begin:end] = hashes
        if begin % (batch_size * 100) == 0:
            print(f"fingerprints {begin}/{len(values)}", flush=True)
    finish_memmap(result, temporary, output)


def freeze_schedule(args: argparse.Namespace) -> None:
    from train_arm_reproduce import OfficialPretrainDataset

    output = args.output.resolve()
    if (output / "ARTIFACT_COMPLETE").is_file() and not args.force:
        print(f"schedule already complete: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    dataset = OfficialPretrainDataset(args.training_root.resolve(), 20, args.shuffle_buffer)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    targets, target_tmp = open_atomic_memmap(output / "targets_f32.npy", np.float32, (args.rows, SEQUENCE))
    indices, indices_tmp = open_atomic_memmap(output / "official_indices_i32.npy", np.int32, (args.rows, 20))
    distances, distances_tmp = open_atomic_memmap(
        output / "official_distances_f32.npy", np.float32, (args.rows, 20)
    )
    query_ids, ids_tmp = open_atomic_memmap(output / "query_ids.npy", np.int64, (args.rows,))
    checksums = []
    offset = 0
    for batch_index, batch in enumerate(loader):
        take = min(len(batch["x"]), args.rows - offset)
        if take <= 0:
            break
        target = np.concatenate(
            [batch["x"][:take].numpy(), batch["y"][:take].numpy()], axis=1
        ).astype(np.float32, copy=False)
        ind = batch["indices"][:take].numpy().astype(np.int32, copy=False)
        dist = batch["distances"][:take].numpy().astype(np.float32, copy=False)
        if ind.shape[1] != 20 or dist.shape[1] != 20:
            raise ValueError("Official schedule does not contain Top-20")
        if np.any(np.diff(dist, axis=1) < -1e-7):
            raise ValueError(f"Official distances are not ordered at batch {batch_index}")
        end = offset + take
        targets[offset:end] = target
        indices[offset:end] = ind
        distances[offset:end] = dist
        query_ids[offset:end] = np.arange(offset, end, dtype=np.int64)
        checksums.append(
            hashlib.sha256(target.tobytes() + ind.tobytes() + dist.tobytes()).hexdigest()
        )
        offset = end
        if batch_index % 100 == 0:
            print(f"schedule rows {offset}/{args.rows}", flush=True)
        if offset == args.rows:
            break
    if offset != args.rows:
        raise RuntimeError(f"Frozen only {offset}/{args.rows} schedule rows")
    finish_memmap(targets, target_tmp, output / "targets_f32.npy")
    finish_memmap(indices, indices_tmp, output / "official_indices_i32.npy")
    finish_memmap(distances, distances_tmp, output / "official_distances_f32.npy")
    finish_memmap(query_ids, ids_tmp, output / "query_ids.npy")
    atomic_json(output / "batch_checksums.json", {"sha256": checksums})
    fingerprint_rows(np.load(output / "targets_f32.npy", mmap_mode="r")[:, :CONTEXT], output / "query_hashes_u64.npy")
    complete(
        output,
        {
            "schema": "tsrag_frozen_query_schedule_v2",
            "rows": args.rows,
            "target_shape": [args.rows, SEQUENCE],
            "official_top_k": 20,
            "seed": SEED,
            "shuffle_buffer": args.shuffle_buffer,
            "source_files": dataset.filesystem_order,
            "shuffle_generator_initial_seed": dataset.dataset.generator.initial_seed(),
            "targets_sha256": sha256_file(output / "targets_f32.npy"),
            "indices_sha256": sha256_file(output / "official_indices_i32.npy"),
            "distances_sha256": sha256_file(output / "official_distances_f32.npy"),
        },
    )


def build_arm_init(args: argparse.Namespace) -> None:
    from transformers import AutoConfig
    from models.ChronosBolt import ChronosBoltModelForForecastingWithRetrieval

    output = args.output.resolve()
    if output.is_file() and not args.force:
        print(f"ARM initialization already exists: {output}")
        return
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    base = args.base_model.resolve()
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        base, config=config, augment="moe", mixer_variant="official", local_files_only=True
    )
    model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer])
    names = ("encode_mlp", "mha", "ffn", "gate_layer")
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if any(module in name for module in names)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, output)
    atomic_json(
        output.with_suffix(".json"),
        {"seed": SEED, "parameter_count": sum(v.numel() for v in state.values()), "sha256": sha256_file(output)},
    )


def build_database_hashes(args: argparse.Namespace) -> None:
    store = np.load(args.retrieval_store.resolve(), mmap_mode="r")
    if store.shape != (DATABASE_ROWS, SEQUENCE):
        raise ValueError(f"Retrieval store shape mismatch: {store.shape}")
    fingerprint_rows(store[:, :CONTEXT], args.output.resolve())


def splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> np.uint64(31))


def build_random(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if (output / "ARTIFACT_COMPLETE").is_file() and not args.force:
        print(f"random artifact already complete: {output}")
        return
    schedule = args.schedule.resolve()
    query_ids = np.load(schedule / "query_ids.npy", mmap_mode="r")
    query_hashes = np.load(schedule / "query_hashes_u64.npy", mmap_mode="r")
    database_hashes = np.load(args.database_hashes.resolve(), mmap_mode="r")
    rows = len(query_ids)
    indices, ind_tmp = open_atomic_memmap(output / "indices_i32.npy", np.int32, (rows, TOP_K))
    distances, dist_tmp = open_atomic_memmap(output / "distances_f32.npy", np.float32, (rows, TOP_K))
    signs, sign_tmp = open_atomic_memmap(output / "signs_i8.npy", np.int8, (rows, TOP_K))
    for begin in range(0, rows, args.batch_size):
        end = min(rows, begin + args.batch_size)
        q = query_ids[begin:end].astype(np.uint64)
        chosen = np.empty((end - begin, TOP_K), dtype=np.int64)
        for rank in range(TOP_K):
            counter = 0
            candidate = splitmix64(q ^ np.uint64(SEED + rank * 104729 + counter)) % np.uint64(DATABASE_ROWS)
            invalid = database_hashes[candidate.astype(np.int64)] == query_hashes[begin:end]
            if rank:
                invalid |= np.any(chosen[:, :rank] == candidate[:, None], axis=1)
            while np.any(invalid):
                counter += 1
                replacement = splitmix64(
                    q[invalid] ^ np.uint64(SEED + rank * 104729 + counter * 130363)
                ) % np.uint64(DATABASE_ROWS)
                candidate[invalid] = replacement
                invalid = database_hashes[candidate.astype(np.int64)] == query_hashes[begin:end]
                if rank:
                    invalid |= np.any(chosen[:, :rank] == candidate[:, None], axis=1)
                if counter > 100:
                    raise RuntimeError("Could not generate unique deterministic random neighbors")
            chosen[:, rank] = candidate.astype(np.int64)
        indices[begin:end] = chosen.astype(np.int32)
        distances[begin:end] = 0
        signs[begin:end] = 1
        if begin % (args.batch_size * 100) == 0:
            print(f"random rows {begin}/{rows}", flush=True)
    finish_memmap(indices, ind_tmp, output / "indices_i32.npy")
    finish_memmap(distances, dist_tmp, output / "distances_f32.npy")
    finish_memmap(signs, sign_tmp, output / "signs_i8.npy")
    complete(
        output,
        {
            "schema": "tsrag_retrieval_sidecar_v2",
            "method": "random",
            "rows": rows,
            "top_k": TOP_K,
            "seed": SEED,
            "distance_semantics": "random_none",
        },
    )


def source_array(args: argparse.Namespace) -> np.ndarray:
    if args.source == "database":
        return np.load(args.retrieval_store.resolve(), mmap_mode="r")[:, :CONTEXT]
    return np.load(args.schedule.resolve() / "targets_f32.npy", mmap_mode="r")[:, :CONTEXT]


def standardize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-8)


class BoltEmbedder:
    def __init__(self, model_path: Path, device: torch.device):
        from transformers import AutoConfig
        from models.ChronosBolt import ChronosBoltModelForForecastingWithRetrieval

        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.model = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
            model_path, config=config, augment="moe", mixer_variant="official", local_files_only=True
        ).to(device)
        self.model.requires_grad_(False).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, values: np.ndarray) -> np.ndarray:
        model = self.model
        context = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(self.device)
        mask = torch.isfinite(context).to(context.dtype)
        context, _ = model.instance_norm(context)
        patched_context = model.patch(context)
        patched_mask = torch.nan_to_num(model.patch(mask), nan=0.0)
        patched_context[~(patched_mask > 0)] = 0.0
        inputs = model.input_patch_embedding(torch.cat([patched_context, patched_mask], dim=-1).to(model.dtype))
        attention_mask = patched_mask.sum(dim=-1) > 0
        if model.chronos_config.use_reg_token:
            reg_ids = torch.full((len(context), 1), model.config.reg_token_id, device=self.device)
            inputs = torch.cat([inputs, model.shared(reg_ids)], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones_like(reg_ids)], dim=-1)
        hidden = model.encoder(attention_mask=attention_mask, inputs_embeds=inputs)[0]
        result = hidden[:, -1, :].float().cpu().numpy()
        if result.shape[1] != EMBED_DIM or not np.isfinite(result).all():
            raise ValueError("Invalid Chronos-Bolt embedding")
        return result


def qwen_text(values: np.ndarray) -> list[str]:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    normalized = np.divide(values - mean, std, out=np.zeros_like(values), where=std > 1e-8)
    bins = np.rint((np.clip(normalized, -4, 4) + 4.0) * (255.0 / 8.0)).astype(np.uint8)
    prefix = (
        "Represent this normalized time series for similarity retrieval. "
        "There are exactly 512 chronological integer bins; 0=-4 sigma, 128=0, 255=+4 sigma. Values: "
    )
    return [prefix + " ".join(map(str, row.tolist())) for row in bins]


class QwenEmbedder:
    def __init__(self, model_path: Path, device: torch.device):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16).to(device)
        self.model.requires_grad_(False).eval()
        self.device = device
        self.max_seen_tokens = 0

    @torch.inference_mode()
    def __call__(self, values: np.ndarray) -> np.ndarray:
        texts = qwen_text(values)
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        lengths = encoded["attention_mask"].sum(dim=1)
        self.max_seen_tokens = max(self.max_seen_tokens, int(lengths.max().item()))
        model_limit = int(getattr(self.model.config, "max_position_embeddings", 32768))
        if int(lengths.max()) > model_limit:
            raise ValueError(f"Qwen input would be truncated: {int(lengths.max())}>{model_limit}")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.model(**encoded).last_hidden_state
        if bool((encoded["attention_mask"][:, -1] == 1).all()):
            pooled = hidden[:, -1]
        else:
            positions = encoded["attention_mask"].sum(dim=1) - 1
            pooled = hidden[torch.arange(len(hidden), device=self.device), positions]
        pooled = torch.nn.functional.normalize(pooled[:, :EMBED_DIM].float(), p=2, dim=1)
        result = pooled.cpu().numpy()
        if result.shape[1] != EMBED_DIM or not np.isfinite(result).all():
            raise ValueError("Invalid Qwen embedding")
        return result


def build_embedder(name: str, model_path: Path | None, device: torch.device):
    if name == "bolt":
        return BoltEmbedder(model_path, device)
    if name == "qwen":
        return QwenEmbedder(model_path, device)
    if name == "pearson":
        return standardize_rows
    raise ValueError(name)


def encode_shard(args: argparse.Namespace) -> None:
    values = source_array(args)
    if args.max_rows is not None:
        values = values[: args.max_rows]
    begin, end = ranges(len(values), args.num_shards)[args.shard_index]
    output = args.output.resolve() / args.encoder / args.source / f"part_{args.shard_index:04d}.npy"
    marker = output.with_suffix(".done")
    if marker.is_file() and output.is_file() and not args.force:
        print(f"embedding shard already complete: {output}")
        return
    device = torch.device(f"cuda:{args.gpu}") if args.encoder in ("bolt", "qwen") else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    embedder = build_embedder(args.encoder, args.model_path.resolve() if args.model_path else None, device)
    dimension = CONTEXT if args.encoder == "pearson" else EMBED_DIM
    result, temporary = open_atomic_memmap(output, np.float32, (end - begin, dimension))
    started = time.monotonic()
    for offset in range(begin, end, args.batch_size):
        stop = min(end, offset + args.batch_size)
        result[offset - begin : stop - begin] = embedder(np.asarray(values[offset:stop], dtype=np.float32))
        if (offset - begin) % (args.batch_size * 100) == 0:
            print(
                json.dumps(
                    {
                        "encoder": args.encoder,
                        "source": args.source,
                        "shard": args.shard_index,
                        "rows": offset - begin,
                        "total": end - begin,
                        "rows_per_second": (offset - begin + args.batch_size) / max(time.monotonic() - started, 1e-6),
                        "max_tokens": getattr(embedder, "max_seen_tokens", None),
                    }
                ),
                flush=True,
            )
    finish_memmap(result, temporary, output)
    marker.write_text(sha256_file(output) + "\n")


def require_faiss_gpu():
    import faiss

    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("GPU FAISS is required for full retrieval artifact generation")
    return faiss


def embedding_parts(root: Path, encoder: str, source: str, count: int) -> list[Path]:
    result = [root / encoder / source / f"part_{index:04d}.npy" for index in range(count)]
    missing = [str(path) for path in result if not path.is_file() or not path.with_suffix(".done").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing embedding shards: {missing[:5]}")
    return result


def build_index(args: argparse.Namespace) -> None:
    faiss = require_faiss_gpu()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.with_suffix(".json").is_file() and not args.force:
        print(f"index already complete: {output}")
        return
    parts = embedding_parts(args.embeddings.resolve(), args.encoder, "database", args.num_shards)
    dimension = CONTEXT if args.encoder == "pearson" else EMBED_DIM
    metric = faiss.METRIC_INNER_PRODUCT if args.encoder == "pearson" else faiss.METRIC_L2
    quantizer = faiss.IndexFlatIP(dimension) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, args.nlist, metric)
    rng = np.random.default_rng(SEED)
    sample_indices = np.sort(rng.choice(DATABASE_ROWS, size=min(args.train_rows, DATABASE_ROWS), replace=False))
    samples = np.empty((len(sample_indices), dimension), dtype=np.float32)
    offsets = np.cumsum([0] + [len(np.load(path, mmap_mode="r")) for path in parts])
    for part_index, path in enumerate(parts):
        mask = (sample_indices >= offsets[part_index]) & (sample_indices < offsets[part_index + 1])
        if mask.any():
            local = sample_indices[mask] - offsets[part_index]
            samples[mask] = np.load(path, mmap_mode="r")[local]
    resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(resources, args.gpu, index)
    gpu_index.train(samples)
    for path in parts:
        array = np.load(path, mmap_mode="r")
        for begin in range(0, len(array), args.add_batch):
            gpu_index.add(np.asarray(array[begin : begin + args.add_batch], dtype=np.float32))
            print(f"index ntotal={gpu_index.ntotal}/{DATABASE_ROWS}", flush=True)
    if gpu_index.ntotal != DATABASE_ROWS:
        raise RuntimeError(f"Index contains {gpu_index.ntotal}/{DATABASE_ROWS} vectors")
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    temporary = output.with_suffix(output.suffix + ".tmp")
    faiss.write_index(cpu_index, str(temporary))
    os.replace(temporary, output)
    atomic_json(
        output.with_suffix(".json"),
        {"encoder": args.encoder, "dimension": dimension, "metric": int(metric), "nlist": args.nlist, "ntotal": DATABASE_ROWS},
    )


def load_all_parts(paths: Sequence[Path]) -> np.ndarray:
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    result = np.empty((sum(len(array) for array in arrays), arrays[0].shape[1]), dtype=np.float32)
    offset = 0
    for array in arrays:
        result[offset : offset + len(array)] = array
        offset += len(array)
    return result


def audit_index(args: argparse.Namespace) -> None:
    faiss = require_faiss_gpu()
    index = faiss.read_index(str(args.index.resolve()))
    database = load_all_parts(
        embedding_parts(args.embeddings.resolve(), args.encoder, "database", args.num_shards)
    )
    query_part = np.load(
        embedding_parts(args.embeddings.resolve(), args.encoder, "queries", args.num_shards)[0], mmap_mode="r"
    )
    queries = np.asarray(query_part[: args.audit_rows], dtype=np.float32)
    dimension = queries.shape[1]
    resources = faiss.StandardGpuResources()
    exact_cpu = faiss.IndexFlatIP(dimension) if args.encoder == "pearson" else faiss.IndexFlatL2(dimension)
    exact = faiss.index_cpu_to_gpu(resources, args.gpu, exact_cpu)
    exact.add(database)
    if args.encoder == "pearson":
        _, positive = exact.search(queries, TOP_K)
        _, negative = exact.search(-queries, TOP_K)
        exact_ids = np.concatenate([positive, negative], axis=1)
    else:
        _, exact_ids = exact.search(queries, TOP_K)
    del exact, database
    gpu_index = faiss.index_cpu_to_gpu(resources, args.gpu, index)
    selected = None
    recalls = {}
    for nprobe in (32, 64, 128, 256):
        gpu_index.nprobe = nprobe
        if args.encoder == "pearson":
            _, pos = gpu_index.search(queries, TOP_K)
            _, neg = gpu_index.search(-queries, TOP_K)
            approximate = np.concatenate([pos, neg], axis=1)
        else:
            _, approximate = gpu_index.search(queries, TOP_K)
        recall = float(
            np.mean([len(set(exact_ids[row]) & set(approximate[row])) / len(set(exact_ids[row])) for row in range(len(queries))])
        )
        recalls[str(nprobe)] = recall
        if recall >= args.minimum_recall and selected is None:
            selected = nprobe
            break
    if selected is None:
        raise RuntimeError(f"ANN recall below {args.minimum_recall}: {recalls}")
    atomic_json(
        args.output.resolve(),
        {"encoder": args.encoder, "audit_rows": len(queries), "minimum_recall": args.minimum_recall, "recall": recalls, "nprobe": selected},
    )


def select_filtered_partial(
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    query_hashes: np.ndarray,
    database_hashes: np.ndarray,
    metric: str,
    candidate_signs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(candidate_ids)
    out_ids = np.full((rows, TOP_K), -1, dtype=np.int32)
    out_dist = np.full((rows, TOP_K), np.inf, dtype=np.float32)
    out_signs = np.ones((rows, TOP_K), dtype=np.int8)
    counts = np.zeros(rows, dtype=np.int32)
    exact_context_rejections = np.zeros(rows, dtype=np.int32)
    for row in range(rows):
        seen = set()
        write = 0
        for column, candidate in enumerate(candidate_ids[row]):
            candidate = int(candidate)
            if candidate < 0 or candidate in seen:
                continue
            if database_hashes[candidate] == query_hashes[row]:
                exact_context_rejections[row] += 1
                continue
            seen.add(candidate)
            out_ids[row, write] = candidate
            if metric == "abs_pearson":
                similarity = abs(float(candidate_scores[row, column]))
                out_dist[row, write] = 1.0 - similarity
                out_signs[row, write] = int(candidate_signs[row, column])
            else:
                out_dist[row, write] = float(candidate_scores[row, column])
            write += 1
            if write == TOP_K:
                break
        counts[row] = write
    return out_ids, out_dist, out_signs, counts, exact_context_rejections


def select_filtered(
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    query_hashes: np.ndarray,
    database_hashes: np.ndarray,
    metric: str,
    candidate_signs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_ids, out_dist, out_signs, counts, _ = select_filtered_partial(
        candidate_ids,
        candidate_scores,
        query_hashes,
        database_hashes,
        metric,
        candidate_signs,
    )
    insufficient = np.flatnonzero(counts != TOP_K)
    if len(insufficient):
        row = int(insufficient[0])
        raise RuntimeError(f"Only {int(counts[row])} valid neighbors remained for query row {row}")
    return out_ids, out_dist, out_signs


def rank_candidates(index, queries: np.ndarray, encoder: str, search_k: int):
    if encoder == "pearson":
        positive_scores, positive_ids = index.search(queries, search_k)
        negative_scores, negative_ids = index.search(-queries, search_k)
        candidate_ids = np.concatenate([positive_ids, negative_ids], axis=1)
        raw_scores = np.concatenate([positive_scores, -negative_scores], axis=1)
        candidate_signs = np.concatenate(
            [np.ones_like(positive_ids, dtype=np.int8), -np.ones_like(negative_ids, dtype=np.int8)], axis=1
        )
        order = np.argsort(-np.abs(raw_scores), axis=1)
        return (
            np.take_along_axis(candidate_ids, order, axis=1),
            np.take_along_axis(raw_scores, order, axis=1),
            np.take_along_axis(candidate_signs, order, axis=1),
            "abs_pearson",
        )
    candidate_scores, candidate_ids = index.search(queries, search_k)
    return candidate_ids, candidate_scores, None, "l2"


def adaptive_filtered_search(
    index,
    queries: np.ndarray,
    query_hashes: np.ndarray,
    database_hashes: np.ndarray,
    encoder: str,
    initial_search_k: int = SEARCH_K,
    max_search_k: int = MAX_SEARCH_K,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    rows = len(queries)
    out_ids = np.full((rows, TOP_K), -1, dtype=np.int32)
    out_dist = np.full((rows, TOP_K), np.inf, dtype=np.float32)
    out_signs = np.ones((rows, TOP_K), dtype=np.int8)
    unresolved = np.arange(rows, dtype=np.int64)
    search_k = min(initial_search_k, int(index.ntotal))
    hard_cap = min(max_search_k, int(index.ntotal))
    attempts = []
    retried_rows = 0

    while len(unresolved):
        candidate_ids, candidate_scores, candidate_signs, metric = rank_candidates(
            index, np.asarray(queries[unresolved], dtype=np.float32), encoder, search_k
        )
        selected_ids, selected_dist, selected_signs, counts, rejected = select_filtered_partial(
            candidate_ids,
            candidate_scores,
            query_hashes[unresolved],
            database_hashes,
            metric,
            candidate_signs,
        )
        complete_rows = counts == TOP_K
        completed_global = unresolved[complete_rows]
        out_ids[completed_global] = selected_ids[complete_rows]
        out_dist[completed_global] = selected_dist[complete_rows]
        out_signs[completed_global] = selected_signs[complete_rows]
        next_unresolved = unresolved[~complete_rows]
        attempts.append(
            {
                "search_k": int(search_k),
                "rows": int(len(unresolved)),
                "completed": int(np.count_nonzero(complete_rows)),
                "insufficient": int(len(next_unresolved)),
                "max_exact_context_rejections": int(rejected.max(initial=0)),
            }
        )
        if not len(next_unresolved):
            break
        if search_k >= hard_cap:
            local = int(np.flatnonzero(~complete_rows)[0])
            global_row = int(unresolved[local])
            raise RuntimeError(
                f"Only {int(counts[local])} valid neighbors remained for query row {global_row} "
                f"after adaptive search up to k={search_k}"
            )
        retried_rows += len(next_unresolved)
        unresolved = next_unresolved
        search_k = min(search_k * 2, hard_cap)

    diagnostics = {
        "initial_search_k": int(min(initial_search_k, int(index.ntotal))),
        "final_search_k": int(search_k),
        "max_search_k": int(hard_cap),
        "retried_row_attempts": int(retried_rows),
        "attempts": attempts,
    }
    return (out_ids, out_dist, out_signs), diagnostics


def search_shard(args: argparse.Namespace) -> None:
    faiss = require_faiss_gpu()
    output = args.output.resolve() / "shards"
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"part_{args.shard_index:04d}"
    marker = prefix.with_suffix(".done")
    if marker.is_file() and not args.force:
        print(f"search shard already complete: {prefix}")
        return
    query_path = embedding_parts(args.embeddings.resolve(), args.encoder, "queries", args.num_shards)[args.shard_index]
    queries = np.load(query_path, mmap_mode="r")
    begin_global, _ = ranges(args.total_queries, args.num_shards)[args.shard_index]
    query_hashes_all = np.load(args.query_hashes.resolve(), mmap_mode="r")
    query_hashes = query_hashes_all[begin_global : begin_global + len(queries)]
    database_hashes = np.load(args.database_hashes.resolve(), mmap_mode="r")
    audit = json.loads(args.audit.resolve().read_text())
    cpu_index = faiss.read_index(str(args.index.resolve()))
    resources = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(resources, args.gpu, cpu_index)
    index.nprobe = int(audit["nprobe"])
    ids, ids_tmp = open_atomic_memmap(prefix.with_name(prefix.name + "_indices_i32.npy"), np.int32, (len(queries), TOP_K))
    distances, dist_tmp = open_atomic_memmap(prefix.with_name(prefix.name + "_distances_f32.npy"), np.float32, (len(queries), TOP_K))
    signs, sign_tmp = open_atomic_memmap(prefix.with_name(prefix.name + "_signs_i8.npy"), np.int8, (len(queries), TOP_K))
    aggregate_attempts: dict[int, dict[str, int]] = {}
    retried_batches = 0
    retried_row_attempts = 0
    for begin in range(0, len(queries), args.batch_size):
        end = min(len(queries), begin + args.batch_size)
        q = np.asarray(queries[begin:end], dtype=np.float32)
        selected, diagnostics = adaptive_filtered_search(
            index,
            q,
            query_hashes[begin:end],
            database_hashes,
            args.encoder,
            args.initial_search_k,
            args.max_search_k,
        )
        if diagnostics["final_search_k"] > diagnostics["initial_search_k"]:
            retried_batches += 1
        retried_row_attempts += diagnostics["retried_row_attempts"]
        for attempt in diagnostics["attempts"]:
            item = aggregate_attempts.setdefault(
                attempt["search_k"],
                {"calls": 0, "rows": 0, "completed": 0, "insufficient": 0, "max_exact_context_rejections": 0},
            )
            item["calls"] += 1
            for key in ("rows", "completed", "insufficient"):
                item[key] += attempt[key]
            item["max_exact_context_rejections"] = max(
                item["max_exact_context_rejections"], attempt["max_exact_context_rejections"]
            )
        ids[begin:end], distances[begin:end], signs[begin:end] = selected
        if begin % (args.batch_size * 100) == 0:
            print(f"search {args.encoder} shard={args.shard_index} rows={begin}/{len(queries)}", flush=True)
    id_path = prefix.with_name(prefix.name + "_indices_i32.npy")
    dist_path = prefix.with_name(prefix.name + "_distances_f32.npy")
    sign_path = prefix.with_name(prefix.name + "_signs_i8.npy")
    finish_memmap(ids, ids_tmp, id_path)
    finish_memmap(distances, dist_tmp, dist_path)
    finish_memmap(signs, sign_tmp, sign_path)
    atomic_json(
        prefix.with_name(prefix.name + "_search_diagnostics.json"),
        {
            "encoder": args.encoder,
            "shard_index": args.shard_index,
            "rows": len(queries),
            "initial_search_k": args.initial_search_k,
            "max_search_k": args.max_search_k,
            "retried_batches": retried_batches,
            "retried_row_attempts": retried_row_attempts,
            "attempts": {str(key): value for key, value in sorted(aggregate_attempts.items())},
        },
    )
    marker.write_text(json.dumps({"indices": sha256_file(id_path), "distances": sha256_file(dist_path), "signs": sha256_file(sign_path)}) + "\n")


def merge_search(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if (output / "ARTIFACT_COMPLETE").is_file() and not args.force:
        print(f"artifact already merged: {output}")
        return
    rows = args.total_queries
    ids, ids_tmp = open_atomic_memmap(output / "indices_i32.npy", np.int32, (rows, TOP_K))
    distances, dist_tmp = open_atomic_memmap(output / "distances_f32.npy", np.float32, (rows, TOP_K))
    signs, sign_tmp = open_atomic_memmap(output / "signs_i8.npy", np.int8, (rows, TOP_K))
    offset = 0
    for shard in range(args.num_shards):
        prefix = output / "shards" / f"part_{shard:04d}"
        if not prefix.with_suffix(".done").is_file():
            raise FileNotFoundError(prefix.with_suffix(".done"))
        part_ids = np.load(prefix.with_name(prefix.name + "_indices_i32.npy"), mmap_mode="r")
        part_dist = np.load(prefix.with_name(prefix.name + "_distances_f32.npy"), mmap_mode="r")
        part_sign = np.load(prefix.with_name(prefix.name + "_signs_i8.npy"), mmap_mode="r")
        end = offset + len(part_ids)
        ids[offset:end], distances[offset:end], signs[offset:end] = part_ids, part_dist, part_sign
        offset = end
    if offset != rows:
        raise RuntimeError(f"Merged {offset}/{rows} rows")
    finish_memmap(ids, ids_tmp, output / "indices_i32.npy")
    finish_memmap(distances, dist_tmp, output / "distances_f32.npy")
    finish_memmap(signs, sign_tmp, output / "signs_i8.npy")
    complete(
        output,
        {
            "schema": "tsrag_retrieval_sidecar_v2",
            "method": args.method,
            "rows": rows,
            "top_k": TOP_K,
            "indices_sha256": sha256_file(output / "indices_i32.npy"),
            "distances_sha256": sha256_file(output / "distances_f32.npy"),
            "signs_sha256": sha256_file(output / "signs_i8.npy"),
        },
    )


def load_test_dataset(project_root: Path, dataset_name: str):
    from data_provider.data_factory import data_provider

    specs = {
        "ETTh1": ("ett_h", "hour"), "ETTh2": ("ett_h", "hour"),
        "ETTm1": ("ett_m", "minute"), "ETTm2": ("ett_m", "minute"),
        "weather": ("custom", "10minutes"), "electricity": ("custom", "hour"),
        "exchange_rate": ("custom", "hour"),
    }
    loader_name, frequency = specs[dataset_name]
    data_root = project_root / "Data"
    data_path = data_root / "TS-RAG-Data" / "datasets_512" / (
        f"{dataset_name}_retrieve_{dataset_name}_512_only_self_train_None.csv"
    )
    database_path = data_root / "TS-RAG-Data" / "database_512" / (
        f"{dataset_name}_{frequency}_512.pkl"
    )
    with database_path.open("rb") as handle:
        database = pickle.load(handle)
    raw_values = np.asarray([database[name]["raw_data"] for name in database]).T
    raw = StandardScaler().fit_transform(raw_values).T
    data_args = Namespace(
        model_id=f"{dataset_name}_zeroshot_512_pred_64_512_retrieve_64",
        root_path=str(data_path.parent) + os.sep,
        data_path=data_path.name,
        data=loader_name + "_retrieve",
        features="M", freq="h", target="OT", embed="timeF", percent=100,
        max_len=-1, seq_len=512, label_len=0, pred_len=64, batch_size=256,
        num_workers=0, top_k=10, mode="only_self_train", return_feature_id=False,
    )
    dataset, _ = data_provider(data_args, "test", retriever_rawdata=raw)
    return dataset, raw


def exact_search_numpy_or_faiss(
    method: str,
    candidate_vectors: np.ndarray,
    query_vectors: np.ndarray,
    gpu: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faiss = require_faiss_gpu()
    dimension = candidate_vectors.shape[1]
    metric_ip = method == "pearson"
    cpu = faiss.IndexFlatIP(dimension) if metric_ip else faiss.IndexFlatL2(dimension)
    resources = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(resources, gpu, cpu)
    index.add(np.asarray(candidate_vectors, dtype=np.float32))
    if metric_ip:
        pos_scores, pos_ids = index.search(np.asarray(query_vectors, dtype=np.float32), min(SEARCH_K, len(candidate_vectors)))
        neg_scores, neg_ids = index.search(np.asarray(-query_vectors, dtype=np.float32), min(SEARCH_K, len(candidate_vectors)))
        ids = np.concatenate([pos_ids, neg_ids], axis=1)
        scores = np.concatenate([pos_scores, -neg_scores], axis=1)
        signs = np.concatenate([np.ones_like(pos_ids, dtype=np.int8), -np.ones_like(neg_ids, dtype=np.int8)], axis=1)
        order = np.argsort(-np.abs(scores), axis=1)[:, :TOP_K]
        ids = np.take_along_axis(ids, order, axis=1)
        scores = np.take_along_axis(scores, order, axis=1)
        signs = np.take_along_axis(signs, order, axis=1)
        return ids.astype(np.int32), (1.0 - np.abs(scores)).astype(np.float32), signs
    distances, ids = index.search(np.asarray(query_vectors, dtype=np.float32), TOP_K)
    return ids.astype(np.int32), distances.astype(np.float32), np.ones_like(ids, dtype=np.int8)


def test_view(args: argparse.Namespace) -> None:
    output = args.output.resolve() / args.method / args.dataset
    if (output / "ARTIFACT_COMPLETE").is_file() and not args.force:
        print(f"test view already complete: {output}")
        return
    dataset, raw = load_test_dataset(args.project_root.resolve(), args.dataset)
    total = len(dataset)
    timestamps, ts_tmp = open_atomic_memmap(output / "timestamp_indices_i32.npy", np.int32, (total, TOP_K))
    distances, dist_tmp = open_atomic_memmap(output / "distances_f32.npy", np.float32, (total, TOP_K))
    signs, sign_tmp = open_atomic_memmap(output / "signs_i8.npy", np.int8, (total, TOP_K))
    if args.max_samples:
        total_to_build = min(total, args.max_samples)
    else:
        total_to_build = total
    device = torch.device(f"cuda:{args.gpu}")
    embedder = None
    if args.method in ("bolt", "qwen"):
        embedder = build_embedder(args.method, args.model_path.resolve(), device)
    for feature in range(dataset.enc_in):
        global_begin = feature * dataset.tot_len
        global_end = min(global_begin + dataset.tot_len, total_to_build)
        if global_begin >= total_to_build:
            break
        query_count = global_end - global_begin
        query_contexts = np.lib.stride_tricks.sliding_window_view(
            dataset.data_x[:, feature], CONTEXT
        )[:query_count]
        # Match the official only_self_train candidate boundary while requiring a complete future.
        if args.dataset in ("ETTh1", "ETTh2"):
            train_end = 12 * 30 * 24
        elif args.dataset in ("ETTm1", "ETTm2"):
            train_end = 12 * 30 * 24 * 4
        else:
            full_rows = len(raw[feature])
            train_end = int(full_rows * 0.7)
        candidate_count = train_end - SEQUENCE + 1
        candidate_contexts = np.lib.stride_tricks.sliding_window_view(raw[feature], CONTEXT)[:candidate_count]
        if args.method == "official":
            local_ids = np.asarray(dataset.timestamp_idx[:query_count, :TOP_K, feature], dtype=np.int32)
            local_dist = np.asarray(dataset.distance[:query_count, :TOP_K, feature], dtype=np.float32)
            local_sign = np.ones((query_count, TOP_K), dtype=np.int8)
        elif args.method == "random":
            local_ids = np.empty((query_count, TOP_K), dtype=np.int32)
            base_ids = np.arange(global_begin, global_end, dtype=np.uint64)
            for rank in range(TOP_K):
                counter = 0
                candidate = splitmix64(base_ids ^ np.uint64(SEED + rank * 104729)) % np.uint64(candidate_count)
                invalid = np.zeros(query_count, dtype=bool)
                if rank:
                    invalid = np.any(local_ids[:, :rank] == candidate[:, None], axis=1)
                while invalid.any():
                    counter += 1
                    candidate[invalid] = splitmix64(
                        base_ids[invalid] ^ np.uint64(SEED + rank * 104729 + counter * 130363)
                    ) % np.uint64(candidate_count)
                    invalid = np.any(local_ids[:, :rank] == candidate[:, None], axis=1)
                local_ids[:, rank] = candidate.astype(np.int32)
            local_dist = np.zeros((query_count, TOP_K), dtype=np.float32)
            local_sign = np.ones((query_count, TOP_K), dtype=np.int8)
        else:
            if args.method == "pearson":
                candidate_vectors = standardize_rows(candidate_contexts)
                query_vectors = standardize_rows(query_contexts)
            else:
                candidate_chunks = [
                    embedder(candidate_contexts[begin : begin + args.batch_size])
                    for begin in range(0, candidate_count, args.batch_size)
                ]
                query_chunks = [
                    embedder(query_contexts[begin : begin + args.batch_size])
                    for begin in range(0, query_count, args.batch_size)
                ]
                candidate_vectors = np.concatenate(candidate_chunks)
                query_vectors = np.concatenate(query_chunks)
            local_ids, local_dist, local_sign = exact_search_numpy_or_faiss(
                args.method, candidate_vectors, query_vectors, args.gpu
            )
        timestamps[global_begin:global_end] = local_ids
        distances[global_begin:global_end] = local_dist
        signs[global_begin:global_end] = local_sign
        print(f"test-view method={args.method} dataset={args.dataset} feature={feature+1}/{dataset.enc_in}", flush=True)
    if total_to_build < total:
        timestamps[total_to_build:] = 0
        distances[total_to_build:] = 0
        signs[total_to_build:] = 1
    finish_memmap(timestamps, ts_tmp, output / "timestamp_indices_i32.npy")
    finish_memmap(distances, dist_tmp, output / "distances_f32.npy")
    finish_memmap(signs, sign_tmp, output / "signs_i8.npy")
    complete(
        output,
        {"schema": "tsrag_test_retrieval_view_v2", "method": args.method, "dataset": args.dataset, "rows": total, "built_rows": total_to_build},
    )


def validate(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    required = ["indices_i32.npy", "distances_f32.npy", "signs_i8.npy", "ARTIFACT_COMPLETE"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    indices = np.load(root / "indices_i32.npy", mmap_mode="r")
    distances = np.load(root / "distances_f32.npy", mmap_mode="r")
    signs = np.load(root / "signs_i8.npy", mmap_mode="r")
    if indices.shape != distances.shape or indices.shape != signs.shape or indices.shape[1] != TOP_K:
        raise ValueError("Artifact shape mismatch")
    if indices.min() < 0 or indices.max() >= DATABASE_ROWS:
        raise ValueError("Artifact index out of range")
    if not np.isfinite(distances).all() or not np.isin(signs, (-1, 1)).all():
        raise ValueError("Artifact contains invalid distance/sign")
    print(json.dumps({"status": "valid", "root": str(root), "shape": indices.shape}))


def smoke_sidecars(args: argparse.Namespace) -> None:
    schedule = args.schedule.resolve()
    indices = np.asarray(np.load(schedule / "official_indices_i32.npy", mmap_mode="r")[:, :TOP_K], dtype=np.int32)
    distances = np.asarray(np.load(schedule / "official_distances_f32.npy", mmap_mode="r")[:, :TOP_K], dtype=np.float32)
    for method in ("random_k10", "abs_pearson_k10", "chronos_bolt_k10", "qwen3_text_k10"):
        output = args.output.resolve() / method
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "indices_i32.npy", indices)
        np.save(output / "distances_f32.npy", np.zeros_like(distances) if method == "random_k10" else distances)
        signs = np.ones_like(indices, dtype=np.int8)
        if method == "abs_pearson_k10":
            signs[:, 1::2] = -1
        np.save(output / "signs_i8.npy", signs)
        complete(output, {"schema": "tsrag_smoke_sidecar_v2", "method": method, "rows": len(indices), "top_k": TOP_K})


def smoke_faiss_gpu(_: argparse.Namespace) -> None:
    import faiss

    rng = np.random.default_rng(SEED)
    database = rng.standard_normal((64, EMBED_DIM), dtype=np.float32)
    queries = database[:4].copy()
    resources = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatL2(EMBED_DIM))
    index.add(database)
    distances, indices = index.search(queries, TOP_K)
    if indices.shape != (4, TOP_K) or distances.shape != (4, TOP_K):
        raise AssertionError("GPU FAISS returned an invalid result shape")
    if not np.array_equal(indices[:, 0], np.arange(4)) or not np.isfinite(distances).all():
        raise AssertionError("GPU FAISS nearest-neighbor result is invalid")
    print("GPU FAISS smoke passed")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-schedule")
    freeze.add_argument("--training-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--rows", type=int, default=DEFAULT_SCHEDULE_ROWS)
    freeze.add_argument("--batch-size", type=int, default=4096)
    freeze.add_argument("--shuffle-buffer", type=int, default=10000)
    freeze.add_argument("--force", action="store_true")
    freeze.set_defaults(func=freeze_schedule)

    arm = sub.add_parser("arm-init")
    arm.add_argument("--base-model", type=Path, required=True)
    arm.add_argument("--output", type=Path, required=True)
    arm.add_argument("--force", action="store_true")
    arm.set_defaults(func=build_arm_init)

    hashes = sub.add_parser("database-hashes")
    hashes.add_argument("--retrieval-store", type=Path, required=True)
    hashes.add_argument("--output", type=Path, required=True)
    hashes.set_defaults(func=build_database_hashes)

    random_parser = sub.add_parser("random")
    random_parser.add_argument("--schedule", type=Path, required=True)
    random_parser.add_argument("--database-hashes", type=Path, required=True)
    random_parser.add_argument("--output", type=Path, required=True)
    random_parser.add_argument("--batch-size", type=int, default=65536)
    random_parser.add_argument("--force", action="store_true")
    random_parser.set_defaults(func=build_random)

    encode = sub.add_parser("encode")
    encode.add_argument("--encoder", choices=["pearson", "bolt", "qwen"], required=True)
    encode.add_argument("--source", choices=["database", "queries"], required=True)
    encode.add_argument("--retrieval-store", type=Path, required=True)
    encode.add_argument("--schedule", type=Path, required=True)
    encode.add_argument("--model-path", type=Path)
    encode.add_argument("--output", type=Path, required=True)
    encode.add_argument("--gpu", type=int, default=0)
    encode.add_argument("--num-shards", type=int, default=64)
    encode.add_argument("--shard-index", type=int, required=True)
    encode.add_argument("--batch-size", type=int, default=256)
    encode.add_argument("--max-rows", type=int)
    encode.add_argument("--force", action="store_true")
    encode.set_defaults(func=encode_shard)

    index = sub.add_parser("build-index")
    index.add_argument("--encoder", choices=["pearson", "bolt", "qwen"], required=True)
    index.add_argument("--embeddings", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--gpu", type=int, default=0)
    index.add_argument("--num-shards", type=int, default=64)
    index.add_argument("--nlist", type=int, default=16384)
    index.add_argument("--train-rows", type=int, default=1_000_000)
    index.add_argument("--add-batch", type=int, default=131072)
    index.add_argument("--force", action="store_true")
    index.set_defaults(func=build_index)

    audit = sub.add_parser("audit-index")
    audit.add_argument("--encoder", choices=["pearson", "bolt", "qwen"], required=True)
    audit.add_argument("--embeddings", type=Path, required=True)
    audit.add_argument("--index", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--gpu", type=int, default=0)
    audit.add_argument("--num-shards", type=int, default=64)
    audit.add_argument("--audit-rows", type=int, default=2048)
    audit.add_argument("--minimum-recall", type=float, default=0.95)
    audit.set_defaults(func=audit_index)

    search = sub.add_parser("search")
    search.add_argument("--encoder", choices=["pearson", "bolt", "qwen"], required=True)
    search.add_argument("--embeddings", type=Path, required=True)
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--audit", type=Path, required=True)
    search.add_argument("--query-hashes", type=Path, required=True)
    search.add_argument("--database-hashes", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--gpu", type=int, default=0)
    search.add_argument("--num-shards", type=int, default=64)
    search.add_argument("--shard-index", type=int, required=True)
    search.add_argument("--total-queries", type=int, default=DEFAULT_SCHEDULE_ROWS)
    search.add_argument("--batch-size", type=int, default=4096)
    search.add_argument("--initial-search-k", type=int, default=SEARCH_K)
    search.add_argument("--max-search-k", type=int, default=MAX_SEARCH_K)
    search.add_argument("--force", action="store_true")
    search.set_defaults(func=search_shard)

    merge = sub.add_parser("merge-search")
    merge.add_argument("--method", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--num-shards", type=int, default=64)
    merge.add_argument("--total-queries", type=int, default=DEFAULT_SCHEDULE_ROWS)
    merge.add_argument("--force", action="store_true")
    merge.set_defaults(func=merge_search)

    test = sub.add_parser("test-view")
    test.add_argument("--project-root", type=Path, required=True)
    test.add_argument("--method", choices=["official", "random", "pearson", "bolt", "qwen"], required=True)
    test.add_argument("--dataset", required=True)
    test.add_argument("--model-path", type=Path)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--gpu", type=int, default=0)
    test.add_argument("--batch-size", type=int, default=128)
    test.add_argument("--max-samples", type=int)
    test.add_argument("--force", action="store_true")
    test.set_defaults(func=test_view)

    check = sub.add_parser("validate")
    check.add_argument("--root", type=Path, required=True)
    check.set_defaults(func=validate)

    smoke = sub.add_parser("smoke-sidecars")
    smoke.add_argument("--schedule", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.set_defaults(func=smoke_sidecars)

    faiss_smoke = sub.add_parser("smoke-faiss-gpu")
    faiss_smoke.set_defaults(func=smoke_faiss_gpu)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    parsed.func(parsed)
