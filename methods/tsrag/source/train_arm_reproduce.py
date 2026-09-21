#!/usr/bin/env python3
"""Reproduce the official TS-RAG ARM-only training without FAISS or W&B."""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import signal
import subprocess
import time
from datetime import datetime, timezone
from itertools import chain, cycle
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
import transformers
import yaml
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoConfig

from models.ChronosBolt import ChronosBoltModelForForecastingWithRetrieval


BASE_ARM_MODULES = ("encode_mlp", "mha", "ffn", "gate_layer")
EXPECTED_TOTAL_PARAMETERS = 210_068_353
EXPECTED_TRAINABLE_PARAMETERS = 4_775_425
EXPECTED_STATE_KEYS = 287


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


class PseudoShuffledIterableDataset(IterableDataset):
    """Official shuffle-buffer behavior with its private deterministic generator."""

    def __init__(self, base_dataset: Iterable[dict], shuffle_buffer_length: int) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self) -> Iterator[dict]:
        buffer: List[dict] = []
        for element in self.base_dataset:
            buffer.append(element)
            if len(buffer) >= self.shuffle_buffer_length:
                index = int(torch.randint(len(buffer), size=(), generator=self.generator))
                yield buffer.pop(index)
        while buffer:
            index = int(torch.randint(len(buffer), size=(), generator=self.generator))
            yield buffer.pop(index)


class OfficialPretrainDataset(IterableDataset):
    def __init__(self, path: Path, top_k: int, shuffle_buffer_length: int) -> None:
        super().__init__()
        files = list(path.iterdir())
        if not files or any(file.suffix != ".parquet" for file in files):
            raise ValueError("Official pretrain directory must contain parquet files only")
        self.filesystem_order = [str(file.resolve()) for file in files]
        dataset = Cyclic(FileDataset(path, freq="1H"))
        self.dataset = PseudoShuffledIterableDataset(dataset, shuffle_buffer_length)
        self.top_k = top_k

    def __iter__(self) -> Iterator[dict]:
        for entry in self.dataset:
            target = np.asarray(entry["target"], dtype=np.float32)
            indices = np.asarray(entry["indices"][: self.top_k], dtype=np.int64)
            distances = np.asarray(entry["distances"][: self.top_k], dtype=np.float32)
            if target.shape != (576,) or indices.shape != (self.top_k,) or distances.shape != (self.top_k,):
                raise ValueError(
                    f"Invalid sample shapes target={target.shape}, indices={indices.shape}, distances={distances.shape}"
                )
            yield {
                "x": target[:512],
                "y": target[512:],
                "indices": indices,
                "distances": distances,
                "signs": np.ones(self.top_k, dtype=np.int8),
            }


class FrozenScheduleDataset(Dataset):
    """Map-style view of the exact, pre-materialized official 10k sample schedule."""

    def __init__(self, schedule_root: Path, top_k: int, retrieval_artifact: Optional[Path]) -> None:
        marker = schedule_root / "ARTIFACT_COMPLETE"
        if not marker.is_file():
            raise FileNotFoundError(f"Frozen query schedule is incomplete: {marker}")
        self.targets = np.load(schedule_root / "targets_f32.npy", mmap_mode="r")
        self.query_ids = np.load(schedule_root / "query_ids.npy", mmap_mode="r")
        if retrieval_artifact is None:
            self.indices = np.load(schedule_root / "official_indices_i32.npy", mmap_mode="r")
            self.distances = np.load(schedule_root / "official_distances_f32.npy", mmap_mode="r")
            self.signs = None
            self.retrieval_source = "official_chronos_t5"
        else:
            if not (retrieval_artifact / "ARTIFACT_COMPLETE").is_file():
                raise FileNotFoundError(f"Retrieval artifact is incomplete: {retrieval_artifact}")
            self.indices = np.load(retrieval_artifact / "indices_i32.npy", mmap_mode="r")
            self.distances = np.load(retrieval_artifact / "distances_f32.npy", mmap_mode="r")
            self.signs = np.load(retrieval_artifact / "signs_i8.npy", mmap_mode="r")
            self.retrieval_source = retrieval_artifact.name
        self.top_k = top_k
        rows = len(self.targets)
        if self.targets.shape[1:] != (576,) or self.query_ids.shape != (rows,):
            raise ValueError("Frozen query schedule shape mismatch")
        if self.indices.shape[0] != rows or self.distances.shape[0] != rows:
            raise ValueError("Retrieval artifact row count does not match frozen schedule")
        if self.indices.shape[1] < top_k or self.distances.shape[1] < top_k:
            raise ValueError(f"Retrieval artifact does not contain top_k={top_k}")
        if self.signs is not None and self.signs.shape != self.indices.shape:
            raise ValueError("Retrieval signs shape mismatch")
        self.filesystem_order = [str(schedule_root.resolve())]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        target = np.asarray(self.targets[index], dtype=np.float32)
        indices = np.asarray(self.indices[index, : self.top_k], dtype=np.int64)
        distances = np.asarray(self.distances[index, : self.top_k], dtype=np.float32)
        signs = (
            np.ones(self.top_k, dtype=np.int8)
            if self.signs is None
            else np.asarray(self.signs[index, : self.top_k], dtype=np.int8)
        )
        if not np.isin(signs, (-1, 1)).all():
            raise ValueError(f"Invalid retrieval sign at frozen query {index}")
        return {
            "x": target[:512],
            "y": target[512:],
            "indices": indices,
            "distances": distances,
            "signs": signs,
            "query_id": np.int64(self.query_ids[index]),
        }


def hash_parameters(model: torch.nn.Module, trainable: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad != trainable:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def trainable_snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def parameter_update_ratio(model: torch.nn.Module, previous: Dict[str, torch.Tensor]) -> float:
    delta_sq = 0.0
    base_sq = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        current = parameter.detach().cpu()
        old = previous[name]
        delta_sq += (current - old).double().square().sum().item()
        base_sq += old.double().square().sum().item()
        previous[name].copy_(current)
    return math.sqrt(delta_sq) / max(math.sqrt(base_sq), 1e-12)


def gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    zero_count = 0
    finite_count = 0
    nonfinite_count = 0
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = torch.isfinite(grad)
        finite_count += int(finite.sum().item())
        nonfinite_count += int((~finite).sum().item())
        zero_count += int(((grad == 0) & finite).sum().item())
    return {
        "gradient_zero_fraction": zero_count / max(finite_count, 1),
        "gradient_nonfinite_count": nonfinite_count,
    }


def weight_stats(model: torch.nn.Module, trainable_modules: Iterable[str]) -> Dict[str, Dict[str, float]]:
    result = {}
    for module_name in trainable_modules:
        tensors = [
            parameter.detach().float().reshape(-1)
            for name, parameter in model.named_parameters()
            if module_name in name and parameter.requires_grad
        ]
        if not tensors:
            continue
        values = torch.cat(tensors)
        result[module_name] = {"mean": values.mean().item(), "std": values.std(unbiased=False).item()}
    return result


def save_model(model: torch.nn.Module, path: Path, expected_state_keys: int) -> None:
    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
    state = unwrapped.state_dict()
    if len(state) != expected_state_keys:
        raise RuntimeError(f"Expected {expected_state_keys} state keys, got {len(state)}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--top-k", type=int, choices=[1, 5, 10, 15, 20])
    parser.add_argument("--query-schedule", type=Path)
    parser.add_argument("--retrieval-artifact", type=Path)
    parser.add_argument("--arm-init-checkpoint", type=Path)
    parser.add_argument(
        "--mixer-variant",
        choices=["official", "uniform", "raw_softmax", "distance_aware"],
    )
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "diagnostics").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    log_path = run_dir / "logs" / "train.jsonl"
    metrics_path = run_dir / "metrics.json"

    seed = args.seed if args.seed is not None else int(config["run"]["seed"])
    mixer_variant = args.mixer_variant or config["model"].get("mixer_variant", "official")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    execution = config["execution"]
    max_updates = args.max_updates or int(execution["optimizer_updates"])
    batch_size = args.batch_size or int(execution["global_batch_size"])
    accumulation = args.gradient_accumulation or int(execution["gradient_accumulation_steps"])
    if batch_size * accumulation != int(execution["global_batch_size"]):
        raise ValueError("micro batch times gradient accumulation must equal global batch 256")
    devices = [args.gpu] if args.gpu is not None else [int(device) for device in execution["devices"]]
    if torch.cuda.device_count() < len(devices):
        raise RuntimeError(f"Need {len(devices)} CUDA devices, found {torch.cuda.device_count()}")
    device = torch.device(f"cuda:{devices[0]}")

    data_root = project_root / "Data" / "TS-RAG-Data"
    artifact_root = project_root / "artifacts" / "training_v1"
    manifest = json.loads((artifact_root / "training_data_manifest.json").read_text())
    retrieval_store = np.load(artifact_root / "retrieval_sequences_f32.npy", mmap_mode="r")
    if retrieval_store.shape != (2_792_864, 576) or retrieval_store.dtype != np.float32:
        raise ValueError("Retrieval store contract mismatch")

    top_k = args.top_k if args.top_k is not None else int(config["model"]["top_k"])
    if args.query_schedule:
        dataset = FrozenScheduleDataset(
            args.query_schedule.resolve(),
            top_k=top_k,
            retrieval_artifact=args.retrieval_artifact.resolve() if args.retrieval_artifact else None,
        )
    else:
        if args.retrieval_artifact:
            raise ValueError("--retrieval-artifact requires --query-schedule")
        dataset = OfficialPretrainDataset(
            data_root / "pretrain_pairs_ctx512",
            top_k=top_k,
            shuffle_buffer_length=int(config["data"]["shuffle_buffer_length"]),
        )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    iterator = iter(loader)

    base_path = project_root / "Data" / "chronos-bolt-base"
    model_config = AutoConfig.from_pretrained(base_path, local_files_only=True)
    model, loading_info = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        base_path,
        config=model_config,
        augment=config["model"]["augment_mode"],
        mixer_variant=mixer_variant,
        local_files_only=True,
        output_loading_info=True,
    )
    unexpected = loading_info.get("unexpected_keys", [])
    illegal_missing = [
        key
        for key in loading_info.get("missing_keys", [])
        if not any(module in key for module in (*BASE_ARM_MODULES, "distance_beta_raw"))
    ]
    if unexpected or illegal_missing:
        raise RuntimeError(f"Base checkpoint mismatch: unexpected={unexpected}, illegal_missing={illegal_missing}")
    model.init_extra_weights([model.encode_mlp, model.mha, model.ffn, model.gate_layer])
    if args.arm_init_checkpoint:
        arm_state = torch.load(args.arm_init_checkpoint.resolve(), map_location="cpu")
        expected_arm_keys = {
            name for name, _ in model.named_parameters() if any(module in name for module in BASE_ARM_MODULES)
        }
        if set(arm_state) != expected_arm_keys:
            raise RuntimeError(
                f"ARM initialization keys mismatch missing={sorted(expected_arm_keys - set(arm_state))} "
                f"unexpected={sorted(set(arm_state) - expected_arm_keys)}"
            )
        parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in arm_state.items():
                parameters[name].copy_(value)
    if mixer_variant == "distance_aware":
        model.distance_beta_raw.data.fill_(0.541324854612918)
    trainable_modules = ["encode_mlp", "mha", "ffn"]
    if mixer_variant != "uniform":
        trainable_modules.append("gate_layer")
    if mixer_variant == "distance_aware":
        trainable_modules.append("distance_beta_raw")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if any(module in name for module in trainable_modules):
            parameter.requires_grad = True
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected_total_parameters = EXPECTED_TOTAL_PARAMETERS + (1 if mixer_variant == "distance_aware" else 0)
    expected_trainable_parameters = EXPECTED_TRAINABLE_PARAMETERS
    if mixer_variant == "uniform":
        expected_trainable_parameters -= 591_361
    elif mixer_variant == "distance_aware":
        expected_trainable_parameters += 1
    expected_state_keys = EXPECTED_STATE_KEYS + (1 if mixer_variant == "distance_aware" else 0)
    if total_parameters != expected_total_parameters or trainable_parameters != expected_trainable_parameters:
        raise RuntimeError(f"Parameter contract mismatch total={total_parameters} trainable={trainable_parameters}")
    frozen_hash_before = hash_parameters(model, trainable=False)
    trainable_hash_before = hash_parameters(model, trainable=True)
    previous_trainable = trainable_snapshot(model)

    gate_capture: List[Dict[str, float]] = []
    alpha_capture: List[Dict[str, float]] = []
    capture_gate = {"enabled": False}

    def gate_hook(_module, _inputs, output):
        if capture_gate["enabled"]:
            gate = torch.sigmoid(output.detach().float())
            gate_capture.append(
                {
                    "mean": gate.mean().item(),
                    "std": gate.std().item(),
                    "saturation": ((gate < 0.01) | (gate > 0.99)).float().mean().item(),
                }
            )

    if mixer_variant != "uniform":
        model.gate_layer.register_forward_hook(gate_hook)
    model.to(device)
    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=devices)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(optimizer_config["scheduler_tmax"]), eta_min=1e-8
    )

    stop_requested = {"value": False, "signal": None}

    def request_stop(signum, _frame):
        stop_requested["value"] = True
        stop_requested["signal"] = signum

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    started_at = now_iso()
    started_clock = time.monotonic()
    atomic_json(
        metrics_path,
        {
            "run_name": run_dir.name,
            "status": "running",
            "draft": args.draft,
            "started_at": started_at,
            "max_updates": max_updates,
            "global_batch_size": batch_size * accumulation,
            "git_commit": git_commit(project_root),
            "data_commit": manifest["retrieval"]["output_sha256"],
        },
    )

    model.train()
    log_every = int(execution["log_every"])
    checkpoint_updates = set(int(value) for value in execution["checkpoint_updates"])
    rolling_loss: List[float] = []
    last_log_clock = time.monotonic()
    official_checkpoint = None
    last_completed_update = 0

    for update in range(1, max_updates + 1):
        if stop_requested["value"]:
            break
        if max_updates == 10_000 and update == 10_000:
            official_checkpoint = run_dir / "checkpoints" / "official_pre_update_10000.pth"
            save_model(model, official_checkpoint, expected_state_keys)
            scheduler.step()

        diagnostic_step = update % log_every == 0 or update == 1 or update == max_updates
        capture_gate["enabled"] = diagnostic_step
        gate_capture.clear()
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for _ in range(accumulation):
            batch = next(iterator)
            indices = batch["indices"].numpy()
            if indices.min() < 0 or indices.max() >= retrieval_store.shape[0]:
                raise ValueError(f"Retrieval indices out of range [{indices.min()}, {indices.max()}]")
            retrieved = torch.from_numpy(np.asarray(retrieval_store[indices], dtype=np.float32))
            signs = batch.get("signs")
            if signs is not None:
                signs = signs.to(dtype=torch.float32)
                if not torch.all((signs == 1) | (signs == -1)):
                    raise ValueError("Retrieval signs must be exactly -1 or +1")
                retrieved = retrieved * signs.unsqueeze(-1)
            context = batch["x"].float()
            target = batch["y"].float()
            distances = batch["distances"].float()
            if not isinstance(model, torch.nn.DataParallel):
                context = context.to(device)
                target = target.to(device)
                retrieved = retrieved.to(device)
                distances = distances.to(device)
            outputs = model(
                context=context,
                target=target,
                retrieved_seq=retrieved,
                distances=distances,
            )
            if diagnostic_step and outputs.attentions is not None:
                alpha = outputs.attentions.detach().float().squeeze(-1)
                alpha_capture.append(
                    {
                        "entropy": (-(alpha * alpha.clamp_min(1e-12).log()).sum(dim=1)).mean().item(),
                        "top1": alpha.max(dim=1).values.mean().item(),
                        "query": alpha[:, 0].mean().item(),
                    }
                )
            loss = outputs.loss.mean() / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {update}: {loss.item()}")
            loss.backward()
            update_loss += loss.item()
        grad_norm = float(clip_grad_norm_(model.parameters(), float(optimizer_config["grad_clip"])).item())
        diagnostic_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        grad_diagnostics = gradient_stats(diagnostic_model)
        if grad_diagnostics["gradient_nonfinite_count"]:
            raise FloatingPointError(f"Non-finite gradients at update {update}")
        optimizer.step()
        last_completed_update = update
        rolling_loss.append(update_loss)

        if update in checkpoint_updates:
            save_model(model, run_dir / "checkpoints" / f"post_update_{update:05d}.pth", expected_state_keys)

        if diagnostic_step:
            current_clock = time.monotonic()
            elapsed_interval = current_clock - last_log_clock
            gate = {
                key: float(np.mean([record[key] for record in gate_capture])) if gate_capture else None
                for key in ("mean", "std", "saturation")
            }
            alpha_stats = {
                key: float(np.mean([record[key] for record in alpha_capture])) if alpha_capture else None
                for key in ("entropy", "top1", "query")
            }
            event = {
                "timestamp": now_iso(),
                "update": update,
                "loss": update_loss,
                "rolling_loss": float(np.mean(rolling_loss)),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_global_norm": grad_norm,
                **grad_diagnostics,
                "parameter_update_ratio": parameter_update_ratio(diagnostic_model, previous_trainable),
                "gate": gate,
                "alpha": alpha_stats,
                "distance_beta": (
                    torch.nn.functional.softplus(diagnostic_model.distance_beta_raw).item()
                    if mixer_variant == "distance_aware"
                    else None
                ),
                "weights": weight_stats(diagnostic_model, trainable_modules),
                "examples_per_second": (batch_size * accumulation * max(len(rolling_loss), 1)) / max(elapsed_interval, 1e-9),
                "peak_vram_bytes": [torch.cuda.max_memory_allocated(index) for index in devices],
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            rolling_loss.clear()
            last_log_clock = current_clock
        capture_gate["enabled"] = False
        alpha_capture.clear()

    post_checkpoint = run_dir / "checkpoints" / f"post_update_{last_completed_update:05d}.pth"
    save_model(model, post_checkpoint, expected_state_keys)
    unwrapped = model.module if isinstance(model, torch.nn.DataParallel) else model
    frozen_hash_after = hash_parameters(unwrapped, trainable=False)
    trainable_hash_after = hash_parameters(unwrapped, trainable=True)
    frozen_unchanged = frozen_hash_before == frozen_hash_after
    if not frozen_unchanged:
        raise RuntimeError("Frozen backbone parameters changed")
    status = "draft_complete" if args.draft else ("complete" if last_completed_update == max_updates else "interrupted")
    final_metrics = {
        "run_name": run_dir.name,
        "mixer_variant": mixer_variant,
        "seed": seed,
        "status": status,
        "draft": args.draft,
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds": time.monotonic() - started_clock,
        "max_updates": max_updates,
        "completed_updates": last_completed_update,
        "global_batch_size": batch_size * accumulation,
        "micro_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_sha256_before": frozen_hash_before,
        "frozen_sha256_after": frozen_hash_after,
        "frozen_unchanged": frozen_unchanged,
        "trainable_sha256_before": trainable_hash_before,
        "trainable_sha256_after": trainable_hash_after,
        "trainable_changed": trainable_hash_before != trainable_hash_after,
        "official_checkpoint": str(official_checkpoint) if official_checkpoint else None,
        "post_checkpoint": str(post_checkpoint),
        "state_key_count": expected_state_keys,
        "git_commit": git_commit(project_root),
        "data_commit": manifest["retrieval"]["output_sha256"],
        "top_k": top_k,
        "query_schedule": str(args.query_schedule.resolve()) if args.query_schedule else None,
        "retrieval_artifact": str(args.retrieval_artifact.resolve()) if args.retrieval_artifact else None,
        "arm_init_checkpoint": str(args.arm_init_checkpoint.resolve()) if args.arm_init_checkpoint else None,
        "filesystem_order": dataset.filesystem_order,
        "shuffle_generator_initial_seed": (
            dataset.dataset.generator.initial_seed() if isinstance(dataset, OfficialPretrainDataset) else None
        ),
        "stop_signal": stop_requested["signal"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(index) for index in devices],
        },
    }
    atomic_json(metrics_path, final_metrics)
    print(json.dumps(final_metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
