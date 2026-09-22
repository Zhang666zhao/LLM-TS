#!/usr/bin/env python3
"""Train the untouched official shared DLinear on the frozen TS-RAG benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


OFFICIAL_DLINEAR_COMMIT = "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
OFFICIAL_DLINEAR_SHA256 = "0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "exchange_rate")


@dataclass(frozen=True)
class SplitSpec:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def load_official_model(repo: Path):
    source = repo / "models" / "DLinear.py"
    # A full external checkout is supported, but the benchmark repository also
    # vendors the exact licensed model file for portable reproduction.
    if (repo / ".git").exists():
        if git_commit(repo) != OFFICIAL_DLINEAR_COMMIT:
            raise RuntimeError("Official LTSF-Linear commit changed")
        if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
            raise RuntimeError("Official LTSF-Linear worktree is dirty")
    if sha256_file(source) != OFFICIAL_DLINEAR_SHA256:
        raise RuntimeError("Official DLinear.py checksum changed")
    spec = importlib.util.spec_from_file_location("official_ltsf_dlinear", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load official DLinear module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def split_spec(dataset: str, rows: int, context: int) -> SplitSpec:
    if dataset in ("ETTh1", "ETTh2"):
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = train_end + 8 * 30 * 24
        if rows < test_end:
            raise ValueError(f"{dataset} has too few rows: {rows}")
        return SplitSpec(0, train_end, train_end - context, val_end, val_end - context, test_end)
    if dataset in ("ETTm1", "ETTm2"):
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = train_end + 8 * 30 * 24 * 4
        if rows < test_end:
            raise ValueError(f"{dataset} has too few rows: {rows}")
        return SplitSpec(0, train_end, train_end - context, val_end, val_end - context, test_end)
    train_end = int(rows * 0.7)
    test_rows = int(rows * 0.2)
    val_end = rows - test_rows
    return SplitSpec(0, train_end, train_end - context, val_end, val_end - context, rows)


class WindowDataset(Dataset):
    def __init__(self, values: np.ndarray, begin: int, end: int, context: int, horizon: int) -> None:
        self.values = values
        self.begin = begin
        self.end = end
        self.context = context
        self.horizon = horizon
        self.length = end - begin - context - horizon + 1
        if self.length <= 0:
            raise ValueError("Empty window dataset")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.begin + index
        middle = start + self.context
        finish = middle + self.horizon
        return torch.from_numpy(self.values[start:middle]), torch.from_numpy(self.values[middle:finish])


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int, drop_last: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=drop_last,
        pin_memory=True,
        generator=generator,
    )


def evaluate_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: Optional[int]) -> float:
    model.eval()
    squared = 0.0
    count = 0
    with torch.inference_mode():
        for batch_index, (context, target) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            context = context.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            prediction = model(context)
            if prediction.shape != target.shape:
                raise RuntimeError(f"Prediction shape {prediction.shape} != target shape {target.shape}")
            if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise RuntimeError("Non-finite validation prediction or target")
            squared += (prediction.double() - target.double()).square().sum().item()
            count += target.numel()
    if count == 0:
        raise RuntimeError("No validation values")
    return squared / count


def evaluate_test(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output: Path,
    variables: list[str],
    horizon: int,
    max_batches: Optional[int],
) -> Dict[str, object]:
    model.eval()
    channel_count = len(variables)
    sample_limit = len(loader.dataset)
    if max_batches is not None:
        sample_limit = min(sample_limit, max_batches * loader.batch_size)
    predictions = np.lib.format.open_memmap(
        output / "predictions_f32.npy", mode="w+", dtype=np.float32, shape=(sample_limit, horizon, channel_count)
    )
    targets = np.lib.format.open_memmap(
        output / "targets_f32.npy", mode="w+", dtype=np.float32, shape=(sample_limit, horizon, channel_count)
    )
    squared_by_variable = np.zeros(channel_count, dtype=np.float64)
    absolute_by_variable = np.zeros(channel_count, dtype=np.float64)
    squared_by_horizon = np.zeros(horizon, dtype=np.float64)
    absolute_by_horizon = np.zeros(horizon, dtype=np.float64)
    offset = 0
    with torch.inference_mode():
        for batch_index, (context, target) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            context = context.to(device=device, dtype=torch.float32, non_blocking=True)
            target_gpu = target.to(device=device, dtype=torch.float32, non_blocking=True)
            prediction = model(context)
            if prediction.shape != target_gpu.shape:
                raise RuntimeError("Test prediction shape mismatch")
            if not torch.isfinite(prediction).all() or not torch.isfinite(target_gpu).all():
                raise RuntimeError("Non-finite test prediction or target")
            error = prediction.double() - target_gpu.double()
            squared_by_variable += error.square().sum(dim=(0, 1)).cpu().numpy()
            absolute_by_variable += error.abs().sum(dim=(0, 1)).cpu().numpy()
            squared_by_horizon += error.square().sum(dim=(0, 2)).cpu().numpy()
            absolute_by_horizon += error.abs().sum(dim=(0, 2)).cpu().numpy()
            batch_count = len(context)
            stop = offset + batch_count
            predictions[offset:stop] = prediction.cpu().numpy()
            targets[offset:stop] = target.numpy()
            offset = stop
    predictions.flush()
    targets.flush()
    if offset != sample_limit:
        raise RuntimeError(f"Expected {sample_limit} test samples, wrote {offset}")
    per_variable_count = offset * horizon
    per_horizon_count = offset * channel_count
    variable_frame = pd.DataFrame(
        {
            "feature_id": np.arange(channel_count),
            "variable": variables,
            "mse": squared_by_variable / per_variable_count,
            "mae": absolute_by_variable / per_variable_count,
            "sample_count": offset,
            "value_count": per_variable_count,
        }
    )
    variable_frame.to_csv(output / "variable_metrics.csv", index=False)
    pd.DataFrame(
        {
            "horizon": np.arange(1, horizon + 1),
            "mse": squared_by_horizon / per_horizon_count,
            "mae": absolute_by_horizon / per_horizon_count,
        }
    ).to_csv(output / "horizon_metrics.csv", index=False)
    total_values = offset * horizon * channel_count
    return {
        "mse": float(squared_by_variable.sum() / total_values),
        "mae": float(absolute_by_variable.sum() / total_values),
        "window_count": offset,
        "channel_count": channel_count,
        "sample_count_tsrag_semantics": offset * channel_count,
        "value_count": total_values,
    }


def cross_channel_invariance(model: torch.nn.Module, channels: int, context: int, device: torch.device) -> float:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(12345)
    baseline = torch.randn((2, context, channels), generator=generator, device=device)
    changed = baseline.clone()
    if channels > 1:
        changed[:, :, 1] += 100.0
    with torch.inference_mode():
        first = model(baseline)[:, :, 0]
        second = model(changed)[:, :, 0]
    return float((first - second).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    args = parser.parse_args()

    started_at = now_iso()
    started_clock = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    atomic_json(output / "config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    atomic_json(output / "status.json", {"status": "running", "dataset": args.dataset, "pid": os.getpid(), "started_at": started_at})

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
        raise RuntimeError(f"GPU {args.gpu} unavailable")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    official_repo = args.official_repo.resolve()
    OfficialModel = load_official_model(official_repo)
    source = (args.data_root / f"{args.dataset}.csv").resolve()
    frame = pd.read_csv(source)
    if frame.columns[0] != "date" or "OT" not in frame.columns:
        raise ValueError("Frozen CSV schema mismatch")
    variables = list(frame.columns[1:])
    numeric = frame[variables].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Frozen CSV contains NaN/Inf")
    splits = split_spec(args.dataset, len(frame), args.context)
    scaler = StandardScaler().fit(numeric[splits.train_start:splits.train_end])
    values = np.asarray(scaler.transform(numeric), dtype=np.float32)
    del numeric, frame

    train_set = WindowDataset(values, splits.train_start, splits.train_end, args.context, args.horizon)
    val_set = WindowDataset(values, splits.val_start, splits.val_end, args.context, args.horizon)
    test_set = WindowDataset(values, splits.test_start, splits.test_end, args.context, args.horizon)
    train_loader = make_loader(train_set, args.batch_size, True, args.workers, args.seed, True)
    val_loader = make_loader(val_set, args.batch_size, False, args.workers, args.seed, False)
    test_loader = make_loader(test_set, args.batch_size, False, args.workers, args.seed, False)

    model = OfficialModel(SimpleNamespace(seq_len=args.context, pred_len=args.horizon, individual=False, enc_in=len(variables)))
    model.to(device)
    channel_delta = cross_channel_invariance(model, len(variables), args.context, device)
    if channel_delta != 0.0:
        raise RuntimeError(f"Official shared DLinear mixed channels unexpectedly: delta={channel_delta}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint = output / "checkpoints" / "best.pth"
    best_val = math.inf
    stale_epochs = 0
    training_log = output / "logs" / "train.jsonl"

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_squared = 0.0
            train_count = 0
            grad_norm_sum = 0.0
            batches = 0
            epoch_start = time.monotonic()
            for batch_index, (context, target) in enumerate(train_loader):
                if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                    break
                context = context.to(device=device, dtype=torch.float32, non_blocking=True)
                target = target.to(device=device, dtype=torch.float32, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(context)
                loss = torch.nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                grad_squared = torch.zeros((), device=device, dtype=torch.float64)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise RuntimeError("Non-finite gradient")
                        grad_squared += parameter.grad.double().square().sum()
                grad_norm_sum += float(grad_squared.sqrt().item())
                optimizer.step()
                train_squared += float(loss.item()) * target.numel()
                train_count += target.numel()
                batches += 1
            if train_count == 0:
                raise RuntimeError("No training batches")
            train_loss = train_squared / train_count
            val_loss = evaluate_loss(model, val_loader, device, args.max_eval_batches)
            record = {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mean_grad_norm": grad_norm_sum / max(batches, 1),
                "batches": batches,
                "elapsed_seconds": time.monotonic() - epoch_start,
            }
            with training_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if val_loss < best_val:
                best_val = val_loss
                stale_epochs = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"early_stop epoch={epoch}", flush=True)
                    break
            # Match official type1 scheduling: after epoch 1 the LR remains at
            # the base value; it is halved after each subsequent epoch.
            next_lr = args.learning_rate * (0.5 ** max(epoch - 1, 0))
            for group in optimizer.param_groups:
                group["lr"] = next_lr

        model.load_state_dict(torch.load(checkpoint, map_location=device))
        test_metrics = evaluate_test(model, test_loader, device, output, variables, args.horizon, args.max_eval_batches)
        elapsed = time.monotonic() - started_clock
        smoke = args.max_train_batches is not None or args.max_eval_batches is not None
        metrics = {
            "status": "smoke_complete" if smoke else "complete",
            "model": "DLinear_shared_official",
            "dataset": args.dataset,
            "started_at": started_at,
            "finished_at": now_iso(),
            "elapsed_seconds": elapsed,
            "best_val_mse": best_val,
            **test_metrics,
            "context_length": args.context,
            "prediction_length": args.horizon,
            "seed": args.seed,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
            "official_repo_commit": OFFICIAL_DLINEAR_COMMIT,
            "official_model_sha256": OFFICIAL_DLINEAR_SHA256,
            "data_path": str(source),
            "data_sha256": sha256_file(source),
            "variables": variables,
            "split": asdict(splits),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "invariants": {
                "individual": False,
                "cross_channel_invariance_max_delta": channel_delta,
                "official_model_unmodified": True,
                "all_predictions_finite": True,
                "test_drop_last": False,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
                "gpu": torch.cuda.get_device_name(device),
            },
        }
        atomic_json(output / "metrics.json", metrics)
        atomic_json(output / "status.json", {"status": metrics["status"], "dataset": args.dataset, "finished_at": metrics["finished_at"]})
        print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    except Exception as error:
        atomic_json(output / "status.json", {"status": "failed", "dataset": args.dataset, "error": repr(error), "failed_at": now_iso()})
        raise


if __name__ == "__main__":
    main()
