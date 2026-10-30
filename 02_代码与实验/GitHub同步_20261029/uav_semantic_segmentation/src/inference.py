"""Unified validation and artifact export for all experiments."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import SegmentationMetrics
from .utils import CLASS_NAMES, denormalize_image, save_json, save_triptych


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    num_classes: int = 8,
    ignore_index: int = 255,
    amp: bool = True,
    output_dir: Path | None = None,
    save_artifacts: bool = False,
    visualization_limit: int = 8,
) -> dict[str, Any]:
    model.eval()
    meter = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    rare_visuals: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    fallback_visuals: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    total_images = 0
    total_inference_seconds = 0.0

    for batch in data_loader:
        images = batch["pixel_values"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with _autocast(device, amp):
            logits = model(pixel_values=images).logits
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        predictions = logits.argmax(dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_inference_seconds += time.perf_counter() - started
        total_images += int(images.shape[0])
        meter.update(predictions, targets)

        if save_artifacts:
            for index in range(images.shape[0]):
                stem = str(batch["stem"][index])
                image_np = denormalize_image(images[index])
                target_np = targets[index].detach().cpu().numpy().astype(np.uint8)
                prediction_np = predictions[index].detach().cpu().numpy().astype(np.uint8)
                record = (stem, image_np, target_np, prediction_np)
                contains_priority = bool(np.any(target_np == 7) or np.any(target_np == 4))
                if contains_priority and len(rare_visuals) < visualization_limit:
                    rare_visuals.append(record)
                elif len(fallback_visuals) < visualization_limit:
                    fallback_visuals.append(record)

    result = meter.compute()
    result["inference_images"] = total_images
    result["inference_seconds"] = total_inference_seconds
    result["inference_images_per_second"] = (
        total_images / total_inference_seconds if total_inference_seconds > 0 else None
    )
    if save_artifacts:
        if output_dir is None:
            raise ValueError("output_dir is required when save_artifacts=True")
        save_evaluation_artifacts(output_dir, result, rare_visuals + fallback_visuals, visualization_limit)
    return result


def save_evaluation_artifacts(
    output_dir: Path,
    metrics: dict[str, Any],
    visualizations: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] | None = None,
    visualization_limit: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "metrics.json", metrics)
    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    np.save(output_dir / "confusion_matrix.npy", confusion)
    np.savetxt(output_dir / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",")
    rows = []
    absent = set(int(x) for x in metrics.get("absent_class_ids", []))
    for class_id, name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "iou": metrics["per_class_iou"][class_id],
                "absent": class_id in absent,
            }
        )
    with (output_dir / "per_class_iou.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "iou", "absent"])
        writer.writeheader()
        writer.writerows(rows)
    if visualizations:
        directory = output_dir / "visualizations" / "predictions"
        for index, (stem, image, target, prediction) in enumerate(visualizations[:visualization_limit]):
            save_triptych(directory / f"prediction_{index + 1:02d}_{stem}.png", image, target, prediction)
