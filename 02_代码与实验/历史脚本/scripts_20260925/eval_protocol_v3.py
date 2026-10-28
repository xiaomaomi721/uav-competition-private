"""EVAL_PROTOCOL_V3: Correct confusion matrix evaluator for AIC2026.

Handles ignore_index=255 explicitly. No guesswork on label mapping.
"""
import numpy as np
from typing import Dict, Optional, Tuple

NUM_CLASSES = 8
IGNORE_INDEX = 255


class ConfusionMatrix:
    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.valid_pixels = 0
        self.ignore_pixels = 0

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        gt = np.asarray(gt)
        pred = np.asarray(pred)
        valid = gt != self.ignore_index
        self.ignore_pixels += int((~valid).sum())
        if not valid.any():
            return
        gt_v = gt[valid].astype(np.int64)
        pred_v = pred[valid].astype(np.int64)
        if gt_v.min() < 0 or gt_v.max() >= self.num_classes:
            raise ValueError(f"GT outside 0..{self.num_classes-1}: min={gt_v.min()}, max={gt_v.max()}")
        if pred_v.min() < 0 or pred_v.max() >= self.num_classes:
            raise ValueError(f"pred outside 0..{self.num_classes-1}: min={pred_v.min()}, max={pred_v.max()}")
        self.valid_pixels += int(valid.sum())
        self.matrix += np.bincount(
            gt_v * self.num_classes + pred_v,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)

    def compute_iou(self) -> Dict:
        cm = self.matrix.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / np.maximum(denom, 1e-12), np.nan)
        present = denom > 0
        miou = float(np.nanmean(iou[present])) if present.any() else float("nan")
        precision = np.where(cm.sum(axis=0) > 0, tp / np.maximum(cm.sum(axis=0), 1e-12), np.nan)
        recall = np.where(cm.sum(axis=1) > 0, tp / np.maximum(cm.sum(axis=1), 1e-12), np.nan)
        return {
            "mIoU": miou,
            "per_class_iou": {i: (float(iou[i]) if present[i] else None) for i in range(self.num_classes)},
            "per_class_precision": {i: (float(precision[i]) if present[i] else None) for i in range(self.num_classes)},
            "per_class_recall": {i: (float(recall[i]) if present[i] else None) for i in range(self.num_classes)},
            "valid_pixels": self.valid_pixels,
            "ignore_pixels": self.ignore_pixels,
            "absent_classes": [i for i in range(self.num_classes) if not present[i]],
            "confusion_matrix": cm.tolist(),
        }

    def merge(self, other: "ConfusionMatrix") -> None:
        self.matrix += other.matrix
        self.valid_pixels += other.valid_pixels
        self.ignore_pixels += other.ignore_pixels


def official_to_train(official_mask: np.ndarray) -> np.ndarray:
    """Official labels 1..8 -> train labels 0..7. Official 0 -> ignore 255."""
    out = np.full_like(official_mask, IGNORE_INDEX, dtype=np.int64)
    for c in range(1, 9):
        out[official_mask == c] = c - 1
    return out


def evaluate_pair(gt: np.ndarray, pred: np.ndarray, use_official_mapping: bool = False) -> Dict:
    cm = ConfusionMatrix()
    if use_official_mapping:
        gt = official_to_train(gt)
    cm.update(gt, pred)
    return cm.compute_iou()
