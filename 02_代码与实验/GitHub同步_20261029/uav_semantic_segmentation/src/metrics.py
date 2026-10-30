"""Confusion-matrix metrics with explicit ignore and absent-class handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class SegmentationMetrics:
    num_classes: int = 8
    ignore_index: int = 255
    confusion_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion_matrix.fill(0)

    def update(self, prediction: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        pred = torch.as_tensor(prediction, dtype=torch.int64).reshape(-1)
        truth = torch.as_tensor(target, dtype=torch.int64).reshape(-1)
        if pred.numel() != truth.numel():
            raise ValueError(f"prediction/target size mismatch: {pred.numel()} vs {truth.numel()}")
        valid = truth != self.ignore_index
        pred = pred[valid]
        truth = truth[valid]
        if truth.numel() == 0:
            return
        if torch.any((truth < 0) | (truth >= self.num_classes)):
            bad = torch.unique(truth[(truth < 0) | (truth >= self.num_classes)]).tolist()
            raise ValueError(f"target contains invalid non-ignore IDs: {bad}")
        if torch.any((pred < 0) | (pred >= self.num_classes)):
            bad = torch.unique(pred[(pred < 0) | (pred >= self.num_classes)]).tolist()
            raise ValueError(f"prediction contains invalid IDs: {bad}")
        encoded = truth * self.num_classes + pred
        counts = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes).cpu().numpy()

    def compute(self) -> dict[str, Any]:
        cm = self.confusion_matrix.astype(np.float64)
        intersection = np.diag(cm)
        target_count = cm.sum(axis=1)
        prediction_count = cm.sum(axis=0)
        union = target_count + prediction_count - intersection
        false_positive = prediction_count - intersection
        false_negative = target_count - intersection
        per_class_iou = np.full(self.num_classes, np.nan, dtype=np.float64)
        per_class_precision = np.full(self.num_classes, np.nan, dtype=np.float64)
        per_class_recall = np.full(self.num_classes, np.nan, dtype=np.float64)
        per_class_false_positive_rate = np.full(self.num_classes, np.nan, dtype=np.float64)
        present = union > 0
        per_class_iou[present] = intersection[present] / union[present]
        total = cm.sum()
        predicted = prediction_count > 0
        target_present = target_count > 0
        negative_count = total - target_count
        has_negative = negative_count > 0
        per_class_precision[predicted] = intersection[predicted] / prediction_count[predicted]
        per_class_recall[target_present] = intersection[target_present] / target_count[target_present]
        per_class_false_positive_rate[has_negative] = false_positive[has_negative] / negative_count[has_negative]
        pixel_accuracy = float(intersection.sum() / total) if total else float("nan")
        miou = float(np.nanmean(per_class_iou)) if np.any(present) else float("nan")
        absent = np.flatnonzero(~present).astype(int).tolist()
        return {
            "mIoU": miou,
            "pixel_accuracy": pixel_accuracy,
            "per_class_iou": [None if np.isnan(x) else float(x) for x in per_class_iou],
            "per_class_precision": [None if np.isnan(x) else float(x) for x in per_class_precision],
            "per_class_recall": [None if np.isnan(x) else float(x) for x in per_class_recall],
            "per_class_false_positive_rate": [
                None if np.isnan(x) else float(x) for x in per_class_false_positive_rate
            ],
            "per_class_false_positive_pixels": [int(x) for x in false_positive],
            "per_class_false_negative_pixels": [int(x) for x in false_negative],
            "per_class_target_pixels": [int(x) for x in target_count],
            "per_class_prediction_pixels": [int(x) for x in prediction_count],
            "absent_class_ids": absent,
            "evaluated_class_count": int(present.sum()),
            "valid_pixel_count": int(total),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
