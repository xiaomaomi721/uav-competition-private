"""Step-based trainer shared by every controlled YAML experiment."""

from __future__ import annotations

import csv
import gc
import math
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import build_datasets
from .inference import evaluate_model, save_evaluation_artifacts
from .losses import build_loss
from .models import build_model, load_model_checkpoint, parameter_count
from .utils import environment_info, save_json, save_yaml, set_seed, utc_now


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _checkpoint_payload(
    model: torch.nn.Module,
    step: int,
    best_miou: float,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": int(step),
        "best_miou": float(best_miou),
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def _peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / 1024**2)


def _current_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device) / 1024**2)


def _required_artifacts(output_dir: Path) -> list[Path]:
    return [
        output_dir / "config.yaml",
        output_dir / "environment.json",
        output_dir / "train_log.csv",
        output_dir / "metrics.json",
        output_dir / "per_class_iou.csv",
        output_dir / "confusion_matrix.npy",
        output_dir / "best_miou.pt",
        output_dir / "last.pt",
    ]


def train_once(
    config: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    project_root: Path,
    device: torch.device,
    *,
    resume: Path | None = None,
    oom_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = deepcopy(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    config["dataset_root_runtime"] = str(dataset_root.resolve())
    config["output_dir_runtime"] = str(output_dir.resolve())
    config["device_runtime"] = str(device)
    config["oom_events"] = list(oom_events or [])
    quick_pilot = bool(config.get("quick_pilot", False))
    save_yaml(output_dir / "config.yaml", config)
    save_json(output_dir / "environment.json", environment_info())

    train_dataset, val_dataset, cropper = build_datasets(config, dataset_root, project_root)
    batch_size = int(config.get("batch_size", 1))
    workers = int(config.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("validation_batch_size", 1)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    rare_enabled = bool(config.get("rare_crop", {}).get("enabled", False))
    domain_enabled = bool(config.get("transforms", {}).get("domain_aug", {}).get("enabled", False))
    if rare_enabled or domain_enabled:
        train_dataset.save_training_previews(
            output_dir / "visualizations" / "augmentation",
            count=int(config.get("augmentation_preview_count", 20)),
            force_rare=rare_enabled,
            force_domain=domain_enabled,
        )

    model = build_model(config).to(device)
    parameters = parameter_count(model)
    criterion, class_weights = build_loss(config, device, project_root)
    if class_weights is not None:
        config.setdefault("loss", {})["resolved_class_weights"] = class_weights
        save_yaml(output_dir / "config.yaml", config)

    optimizer_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("learning_rate", 6.0e-5)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
    )
    max_steps = int(config["max_steps"])
    power = float(config.get("scheduler", {}).get("power", 1.0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(0.0, (1.0 - min(step, max_steps) / max_steps) ** power)
    )
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = _grad_scaler(use_amp)
    accumulation = int(config.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    start_step = 0
    best_miou = -math.inf
    if resume is not None:
        checkpoint = load_model_checkpoint(model, resume, device)
        start_step = int(checkpoint.get("step", 0))
        best_miou = float(checkpoint.get("best_miou", -math.inf))
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])

    log_path = output_dir / "train_log.csv"
    append_log = resume is not None and log_path.is_file()
    log_handle = log_path.open("a" if append_log else "w", encoding="utf-8-sig", newline="")
    fieldnames = ["step", "loss", "learning_rate", "gpu_memory_mb", "peak_vram_mb", "step_time_seconds", "mIoU"]
    writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
    if not append_log:
        writer.writeheader()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    step_times: list[float] = []
    final_step = start_step
    validation_interval = int(config.get("validation_interval", 50))
    log_interval = int(config.get("log_interval", 1))
    try:
        model.train()
        for step in range(start_step + 1, max_steps + 1):
            started = time.perf_counter()
            accumulated_loss = 0.0
            for _ in range(accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                images = batch["pixel_values"].to(device, non_blocking=True)
                targets = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(pixel_values=images).logits
                    logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                    loss = criterion(logits, targets) / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}: {float(loss.detach())}")
                scaler.scale(loss).backward()
                accumulated_loss += float(loss.detach())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            step_times.append(elapsed)
            final_step = step
            current_miou: float | None = None

            if validation_interval > 0 and (step % validation_interval == 0 or step == max_steps):
                validation = evaluate_model(
                    model,
                    val_loader,
                    device,
                    num_classes=int(config.get("num_classes", 8)),
                    ignore_index=int(config.get("ignore_index", 255)),
                    amp=use_amp,
                )
                current_miou = float(validation["mIoU"])
                if current_miou > best_miou:
                    best_miou = current_miou
                    torch.save(
                        _checkpoint_payload(model, step, best_miou, config),
                        output_dir / "best_miou.pt",
                    )
                model.train()

            row = {
                "step": step,
                "loss": accumulated_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gpu_memory_mb": _current_vram_mb(device),
                "peak_vram_mb": _peak_vram_mb(device),
                "step_time_seconds": elapsed,
                "mIoU": "" if current_miou is None else current_miou,
            }
            writer.writerow(row)
            log_handle.flush()
            if step % log_interval == 0 or current_miou is not None:
                print(
                    f"step={step}/{max_steps} loss={accumulated_loss:.6f} "
                    f"lr={row['learning_rate']:.8g} gpu_mb={row['gpu_memory_mb']:.1f} "
                    f"step_s={elapsed:.3f} mIoU={'-' if current_miou is None else f'{current_miou:.6f}'}",
                    flush=True,
                )
    finally:
        log_handle.close()

    torch.save(
        _checkpoint_payload(model, final_step, best_miou, config, optimizer, scheduler, scaler),
        output_dir / "last.pt",
    )
    best_path = output_dir / "best_miou.pt"
    if not best_path.is_file():
        validation = evaluate_model(model, val_loader, device, amp=use_amp)
        best_miou = float(validation["mIoU"])
        torch.save(_checkpoint_payload(model, final_step, best_miou, config), best_path)
    load_model_checkpoint(model, best_path, device)
    final_metrics = evaluate_model(
        model,
        val_loader,
        device,
        num_classes=int(config.get("num_classes", 8)),
        ignore_index=int(config.get("ignore_index", 255)),
        amp=use_amp,
        output_dir=output_dir,
        save_artifacts=True,
        visualization_limit=int(config.get("prediction_visualization_count", 8)),
    )
    final_metrics.update(
        {
            "experiment_id": config.get("experiment_id"),
            "step": config.get("step"),
            "model_name": config.get("model_name"),
            "parameter_count": parameters,
            "crop_size": int(config["crop_size"]),
            "loss_name": config.get("loss", {}).get("name"),
            "max_steps": max_steps,
            "completed_steps": final_step,
            "train_sample_count": len(train_dataset),
            "validation_sample_count": len(val_dataset),
            "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": batch_size * accumulation,
            "peak_vram_mb": _peak_vram_mb(device),
            "average_step_time": sum(step_times) / len(step_times) if step_times else None,
            "best_miou": best_miou,
            "class_weights": class_weights,
            "loss_is_finite": True,
            "rare_crop_enabled": rare_enabled,
            "domain_aug_enabled": domain_enabled,
            "crop_statistics": cropper.statistics.to_dict(),
            "oom_events": list(oom_events or []),
            "quick_pilot": quick_pilot,
        }
    )
    save_evaluation_artifacts(output_dir, final_metrics)
    save_json(output_dir / "crop_statistics.json", cropper.statistics.to_dict())

    missing = [str(path) for path in _required_artifacts(output_dir) if not path.is_file()]
    visualization_dir = output_dir / "visualizations" / "predictions"
    if not visualization_dir.is_dir() or len(list(visualization_dir.glob("*.png"))) < 8:
        missing.append("at least 8 prediction visualizations")
    if missing:
        raise RuntimeError(f"experiment artifacts incomplete: {missing}")
    completion = {
        "status": "completed",
        "completed_at_utc": utc_now(),
        "experiment_id": config.get("experiment_id"),
        "quick_pilot": quick_pilot,
        "required_artifacts": [str(path) for path in _required_artifacts(output_dir)],
    }
    save_json(output_dir / "completion.json", completion)
    save_json(output_dir / "completed.json", completion)
    return final_metrics


def run_training_with_oom_retry(
    config: dict[str, Any],
    dataset_root: Path,
    output_dir: Path,
    project_root: Path,
    device: torch.device,
    *,
    resume: Path | None = None,
) -> dict[str, Any]:
    current = deepcopy(config)
    oom_events: list[dict[str, Any]] = []
    for attempt in range(2):
        try:
            return train_once(
                current,
                dataset_root,
                output_dir,
                project_root,
                device,
                resume=resume if attempt == 0 else None,
                oom_events=oom_events,
            )
        except torch.cuda.OutOfMemoryError as exc:
            event = {
                "attempt": attempt + 1,
                "batch_size": int(current.get("batch_size", 1)),
                "gradient_accumulation_steps": int(current.get("gradient_accumulation_steps", 1)),
                "gradient_checkpointing": bool(current.get("gradient_checkpointing", False)),
                "message": str(exc),
            }
            oom_events.append(event)
            if attempt >= 1:
                raise
            old_batch = int(current.get("batch_size", 1))
            if old_batch > 1:
                current["batch_size"] = max(1, old_batch // 2)
                current["gradient_accumulation_steps"] = int(current.get("gradient_accumulation_steps", 1)) * 2
            elif not bool(current.get("gradient_checkpointing", False)):
                current["gradient_checkpointing"] = True
            else:
                raise
            print(f"CUDA OOM: applying one recorded retry with config {event}", flush=True)
            for name in ("best_miou.pt", "last.pt", "completion.json", "completed.json"):
                candidate = output_dir / name
                if candidate.is_file():
                    candidate.unlink()
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError("unreachable")
