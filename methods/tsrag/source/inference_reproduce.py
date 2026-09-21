#!/usr/bin/env python3
"""Deterministic, inference-only reproduction runner for TS-RAG."""

import argparse
import gc
import hashlib
import json
import os
import pickle
import platform
import subprocess
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
import transformers
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig

from data_provider.data_factory import data_provider
from models.ChronosBolt import (
    ChronosBoltModelForForecastingWithRetrieval,
    ChronosBoltPipeline,
)


DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "ETTh1": {"loader": "ett_h", "frequency": "hour"},
    "ETTh2": {"loader": "ett_h", "frequency": "hour"},
    "ETTm1": {"loader": "ett_m", "frequency": "minute"},
    "ETTm2": {"loader": "ett_m", "frequency": "minute"},
    "weather": {"loader": "custom", "frequency": "10minutes"},
    "electricity": {"loader": "custom", "frequency": "hour"},
    "exchange_rate": {"loader": "custom", "frequency": "hour"},
}
PAPER_TARGETS = {
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


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def git_commit(repo_root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_retriever_rawdata(database_path: Path) -> np.ndarray:
    print(f"loading retrieval database {database_path}", flush=True)
    with database_path.open("rb") as handle:
        database = pickle.load(handle)
    raw_series = [database[name]["raw_data"] for name in database]
    del database
    gc.collect()
    raw = np.asarray(raw_series).T
    scaler = StandardScaler()
    raw = scaler.fit_transform(raw).T
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite values in retrieval raw data: {database_path}")
    return raw


def build_data_args(args: argparse.Namespace, data_path: Path, retrieve: bool) -> Namespace:
    spec = DATASET_SPECS[args.dataset]
    return Namespace(
        model_id=f"{args.dataset}_zeroshot_512_pred_64_512_{'retrieve' if retrieve else 'baseline'}_64",
        root_path=str(data_path.parent) + os.sep,
        data_path=data_path.name,
        data=(spec["loader"] + "_retrieve") if retrieve else spec["loader"],
        features="M",
        freq="h",
        target="OT",
        embed="timeF",
        percent=100,
        max_len=-1,
        seq_len=512,
        label_len=0,
        pred_len=64,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=args.effective_top_k if retrieve and args.retrieval_artifact is None else 10,
        mode="only_self_train",
        return_feature_id=False,
    )


def load_model(args: argparse.Namespace, data_root: Path, device: torch.device):
    base_path = data_root / "chronos-bolt-base"
    if args.model == "tsrag":
        checkpoint = (args.checkpoint or data_root / "TS-RAG-ChronosBolt" / "pytorch_model.bin").resolve()
        config = AutoConfig.from_pretrained(base_path, local_files_only=True)
        model = ChronosBoltModelForForecastingWithRetrieval(
            config, augment="moe", mixer_variant=args.mixer_variant
        )
        state = torch.load(checkpoint, map_location="cpu")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
        model.retrieval_intervention = args.retrieval_intervention
        weight_source = checkpoint
    else:
        pipeline = ChronosBoltPipeline.from_pretrained(base_path, local_files_only=True)
        model = pipeline.model
        weight_source = base_path / "model.safetensors"
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Inference invariant violated: trainable parameters remain")
    return model, weight_source


def central_prediction(model, context: torch.Tensor, retrieved: Optional[torch.Tensor], distances: Optional[torch.Tensor]):
    if retrieved is None:
        model_output = model(context=context)
    else:
        model_output = model(context=context, retrieved_seq=retrieved, distances=distances)
    quantiles = model.config.chronos_config["quantiles"]
    central_idx = quantiles.index(0.5)
    return model_output.quantile_preds[:, central_idx, -64:], model_output.attentions


def intervention_batches(loader: Iterable, intervention: str) -> Iterator:
    """Yield deterministic batches, including a cross-batch cyclic future shift."""
    iterator = iter(loader)
    try:
        current = next(iterator)
    except StopIteration:
        return
    if intervention != "shuffle_future":
        yield current
        yield from iterator
        return
    if len(current) != 6:
        raise ValueError("shuffle_future requires retrieval batches")
    first_retrieval = current[4][:1].clone()
    for following in iterator:
        shifted = torch.cat([current[4][1:], following[4][:1]], dim=0)
        output = list(current)
        output[4] = shifted
        yield output
        current = following
    shifted = torch.cat([current[4][1:], first_retrieval], dim=0)
    output = list(current)
    output[4] = shifted
    yield output


def safe_corr(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else None


class TestRetrievalArtifact:
    """Fixed, sample-order-aligned test retrieval metadata sidecar."""

    def __init__(self, root: Path, rawdata: np.ndarray, dataset) -> None:
        if not (root / "ARTIFACT_COMPLETE").is_file():
            raise FileNotFoundError(f"Incomplete test retrieval artifact: {root}")
        self.timestamps = np.load(root / "timestamp_indices_i32.npy", mmap_mode="r")
        self.distances = np.load(root / "distances_f32.npy", mmap_mode="r")
        self.signs = np.load(root / "signs_i8.npy", mmap_mode="r")
        expected = (len(dataset), 10)
        if self.timestamps.shape != expected or self.distances.shape != expected or self.signs.shape != expected:
            raise ValueError(
                f"Test retrieval artifact shape mismatch expected={expected} "
                f"got={self.timestamps.shape}/{self.distances.shape}/{self.signs.shape}"
            )
        if not np.isin(self.signs, (-1, 1)).all():
            raise ValueError("Test retrieval artifact contains signs other than -1/+1")
        self.rawdata = rawdata
        self.tot_len = int(dataset.tot_len)
        self.root = root

    def batch(self, offset: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stop = offset + batch_size
        timestamps = np.asarray(self.timestamps[offset:stop], dtype=np.int64)
        signs = np.asarray(self.signs[offset:stop], dtype=np.float32)
        result = np.empty((batch_size, timestamps.shape[1], 576), dtype=np.float32)
        for row, sample_id in enumerate(range(offset, stop)):
            feature_id = sample_id // self.tot_len
            for neighbor, timestamp in enumerate(timestamps[row]):
                sequence = self.rawdata[feature_id][timestamp : timestamp + 576]
                if len(sequence) != 576:
                    raise ValueError(
                        f"Invalid fixed retrieval window sample={sample_id} feature={feature_id} timestamp={timestamp}"
                    )
                result[row, neighbor] = sequence * signs[row, neighbor]
        return torch.from_numpy(result), torch.from_numpy(np.asarray(self.distances[offset:stop], dtype=np.float32))


def evaluate(
    model,
    loader,
    dataset,
    device: torch.device,
    max_batches: Optional[int],
    intervention: str,
    effective_top_k: int,
    sample_output: Optional[Path],
    retrieval_artifact: Optional[TestRetrievalArtifact] = None,
) -> Dict[str, object]:
    squared_error = 0.0
    absolute_error = 0.0
    value_count = 0
    sample_count = 0
    sample_records: Dict[str, List[np.ndarray]] = {
        name: []
        for name in (
            "sample_id", "feature_id", "mse", "mae", "distance_min", "distance_mean",
            "distance_std", "neighbor_future_mse", "neighbor_future_oracle_mse",
            "neighbor_agreement", "alpha_entropy",
            "alpha_top1", "alpha_query",
        )
    }
    dataset_offset = 0
    total_length = int(getattr(dataset, "tot_len", len(dataset)))
    with torch.inference_mode():
        if torch.is_grad_enabled():
            raise RuntimeError("torch.inference_mode did not disable gradients")
        for batch_index, batch in enumerate(intervention_batches(loader, intervention)):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch_x, batch_y = batch[0], batch[1]
            context = batch_x[..., 0].to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch_y[..., 0].to(device=device, dtype=torch.float32, non_blocking=True)[:, -64:]
            if retrieval_artifact is not None:
                retrieved_cpu, distances_cpu = retrieval_artifact.batch(dataset_offset, len(batch_x))
                retrieved_cpu = retrieved_cpu[:, :effective_top_k]
                distances_cpu = distances_cpu[:, :effective_top_k]
                retrieved = retrieved_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                distances = distances_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            elif len(batch) == 6:
                retrieved_cpu = batch[4].float()
                distances_cpu = batch[5].float()
                if intervention == "repeat_top1":
                    retrieved_cpu = retrieved_cpu[:, :1].expand(-1, 10, -1).clone()
                    distances_cpu = distances_cpu[:, :1].expand(-1, 10).clone()
                retrieved_cpu = retrieved_cpu[:, :effective_top_k]
                distances_cpu = distances_cpu[:, :effective_top_k]
                retrieved = retrieved_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                distances = distances_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                retrieved = None
                distances = None
                retrieved_cpu = None
                distances_cpu = None
            prediction, alpha = central_prediction(model, context, retrieved, distances)
            if prediction.shape != target.shape:
                raise RuntimeError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise RuntimeError("Non-finite prediction or target detected")
            error = prediction - target
            batch_size = error.shape[0]
            ids = np.arange(dataset_offset, dataset_offset + batch_size, dtype=np.int64)
            dataset_offset += batch_size
            sample_mse = error.float().square().mean(dim=1).cpu().numpy()
            sample_mae = error.float().abs().mean(dim=1).cpu().numpy()
            sample_records["sample_id"].append(ids)
            sample_records["feature_id"].append(ids // max(total_length, 1))
            sample_records["mse"].append(sample_mse)
            sample_records["mae"].append(sample_mae)
            if retrieved_cpu is not None and distances_cpu is not None:
                future = retrieved_cpu[..., -64:]
                target_cpu = target.detach().cpu().unsqueeze(1)
                sample_records["distance_min"].append(distances_cpu.min(dim=1).values.numpy())
                sample_records["distance_mean"].append(distances_cpu.mean(dim=1).numpy())
                sample_records["distance_std"].append(distances_cpu.std(dim=1, unbiased=False).numpy())
                neighbor_errors = (future - target_cpu).square().mean(dim=2)
                sample_records["neighbor_future_mse"].append(neighbor_errors.mean(dim=1).numpy())
                sample_records["neighbor_future_oracle_mse"].append(neighbor_errors.min(dim=1).values.numpy())
                sample_records["neighbor_agreement"].append(
                    future.std(dim=1, unbiased=False).mean(dim=1).numpy()
                )
            if alpha is not None:
                alpha_cpu = alpha.detach().float().squeeze(-1).cpu()
                sample_records["alpha_entropy"].append(
                    (-(alpha_cpu * alpha_cpu.clamp_min(1e-12).log()).sum(dim=1)).numpy()
                )
                sample_records["alpha_top1"].append(alpha_cpu.max(dim=1).values.numpy())
                sample_records["alpha_query"].append(alpha_cpu[:, 0].numpy())
            squared_error += error.double().square().sum().item()
            absolute_error += error.double().abs().sum().item()
            value_count += error.numel()
            sample_count += error.shape[0]
            if batch_index % 100 == 0:
                print(f"batch={batch_index} samples={sample_count}", flush=True)
    if value_count == 0:
        raise RuntimeError("No evaluation samples were produced")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Inference invariant violated: parameter gradients were created")
    arrays = {
        key: np.concatenate(values) if values else np.full(sample_count, np.nan, dtype=np.float32)
        for key, values in sample_records.items()
    }
    if sample_output is not None:
        sample_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_output, **arrays)
    return {
        "mse": squared_error / value_count,
        "mae": absolute_error / value_count,
        "sample_count": sample_count,
        "value_count": value_count,
        "diagnostics": {
            "alpha_entropy_mean": float(np.nanmean(arrays["alpha_entropy"])),
            "alpha_top1_mean": float(np.nanmean(arrays["alpha_top1"])),
            "alpha_query_mean": float(np.nanmean(arrays["alpha_query"])),
            "distance_error_correlation": safe_corr(arrays["distance_min"], arrays["mse"]),
            "neighbor_quality_error_correlation": safe_corr(arrays["neighbor_future_mse"], arrays["mse"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TS-RAG inference-only reproduction")
    parser.add_argument("--model", choices=["tsrag", "chronos_bolt"], required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional TS-RAG checkpoint override")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument(
        "--mixer-variant",
        choices=["official", "uniform", "raw_softmax", "distance_aware"],
        default="official",
    )
    parser.add_argument(
        "--retrieval-intervention",
        choices=["correct", "zero", "shuffle_future", "repeat_top1"],
        default="correct",
    )
    parser.add_argument("--effective-top-k", type=int, choices=[1, 5, 10, 15, 20], default=10)
    parser.add_argument(
        "--retrieval-artifact",
        type=Path,
        help="Fixed test-view directory containing timestamp indices, distances, and signs",
    )
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--save-sample-errors", action="store_true")
    args = parser.parse_args()

    code_root = Path(__file__).resolve().parent
    repo_root = code_root.parent
    data_root = (args.data_root or repo_root / "Data").resolve()
    baseline_root = (args.baseline_root or repo_root / "artifacts" / "baseline_raw_v1").resolve()
    output_path = args.output.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    started_at = now_iso()
    started_clock = time.monotonic()

    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA device {args.gpu} is unavailable")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.model == "tsrag":
        data_path = data_root / "TS-RAG-Data" / "datasets_512" / (
            f"{args.dataset}_retrieve_{args.dataset}_512_only_self_train_None.csv"
        )
        database_path = data_root / "TS-RAG-Data" / "database_512" / (
            f"{args.dataset}_{DATASET_SPECS[args.dataset]['frequency']}_512.pkl"
        )
        retriever_rawdata = load_retriever_rawdata(database_path)
        data_args = build_data_args(args, data_path, retrieve=True)
    else:
        data_path = baseline_root / f"{args.dataset}.csv"
        database_path = None
        retriever_rawdata = None
        data_args = build_data_args(args, data_path, retrieve=False)

    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    dataset, loader = data_provider(data_args, "test", retriever_rawdata=retriever_rawdata)
    fixed_retrieval = None
    if args.retrieval_artifact:
        if args.effective_top_k != 10:
            raise ValueError("New retrieval-method artifacts are fixed at top_k=10")
        fixed_retrieval = TestRetrievalArtifact(args.retrieval_artifact.resolve(), retriever_rawdata, dataset)
    else:
        del retriever_rawdata
        gc.collect()
    model, weight_source = load_model(args, data_root, device)
    paper_mse, paper_mae = PAPER_TARGETS[args.model][args.dataset]

    running = {
        "status": "running", "model": args.model, "dataset": args.dataset,
        "started_at": started_at, "batch_size": args.batch_size,
        "max_batches": args.max_batches, "training_allowed": False,
    }
    atomic_json(output_path / "metrics.json", running)
    evaluation = evaluate(
        model,
        loader,
        dataset,
        device,
        args.max_batches,
        args.retrieval_intervention,
        args.effective_top_k,
        output_path / "sample_errors.npz" if args.save_sample_errors else None,
        fixed_retrieval,
    )
    mse = float(evaluation["mse"])
    mae = float(evaluation["mae"])
    sample_count = int(evaluation["sample_count"])
    value_count = int(evaluation["value_count"])
    elapsed = time.monotonic() - started_clock
    peak_vram = torch.cuda.max_memory_allocated(device)

    manifest = data_root / "download_manifest.json"
    metrics: Dict[str, object] = {
        **running,
        "status": "smoke_complete" if args.max_batches is not None else "complete",
        "finished_at": now_iso(),
        "mse": mse,
        "mae": mae,
        "paper_mse": paper_mse,
        "paper_mae": paper_mae,
        "mse_abs_delta": abs(mse - paper_mse),
        "mae_abs_delta": abs(mae - paper_mae),
        "sample_count": sample_count,
        "value_count": value_count,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": peak_vram,
        "mixer_variant": args.mixer_variant,
        "retrieval_intervention": args.retrieval_intervention,
        "effective_top_k": args.effective_top_k,
        "retrieval_artifact": str(args.retrieval_artifact.resolve()) if args.retrieval_artifact else None,
        "seed": args.seed,
        "diagnostics": evaluation["diagnostics"],
        "sample_errors_path": str(output_path / "sample_errors.npz") if args.save_sample_errors else None,
        "git_commit": git_commit(repo_root),
        "data_path": str(data_path),
        "data_size": data_path.stat().st_size,
        "database_path": str(database_path) if database_path else None,
        "weight_source": str(weight_source),
        "weight_size": weight_source.stat().st_size,
        "manifest_sha256": sha256_file(manifest),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "invariants": {
            "model_eval": not model.training,
            "all_requires_grad_false": not any(p.requires_grad for p in model.parameters()),
            "all_parameter_grads_none": all(p.grad is None for p in model.parameters()),
            "optimizer_created": False,
        },
    }
    atomic_json(output_path / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
