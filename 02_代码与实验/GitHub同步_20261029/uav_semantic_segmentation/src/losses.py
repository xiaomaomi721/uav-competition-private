"""Loss functions used by the controlled quick-pilot experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_json


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 8, ignore_index: int = 255, smooth: float = 1.0e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        safe_target = target.clone()
        safe_target[~valid] = 0
        one_hot = F.one_hot(safe_target.long(), self.num_classes).permute(0, 3, 1, 2).to(logits.dtype)
        valid_mask = valid.unsqueeze(1)
        one_hot = one_hot * valid_mask
        probabilities = F.softmax(logits, dim=1) * valid_mask
        dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dims)
        denominator = probabilities.sum(dims) + one_hot.sum(dims)
        target_present = one_hot.sum(dims) > 0
        if not torch.any(target_present):
            return logits.sum() * 0.0
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice[target_present].mean()


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, ce: nn.Module, dice: DiceLoss, ce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.ce = ce
        self.dice = dice
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = 255, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, target, weight=self.alpha, ignore_index=self.ignore_index, reduction="none")
        ce = ce[valid]
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def class_weights_from_split(
    statistics_path: Path,
    *,
    minimum: float = 0.5,
    maximum: float = 5.0,
) -> list[float]:
    stats = load_json(statistics_path)
    counts_by_official = stats["train"]["pixel_counts_official"]
    counts = torch.tensor([float(counts_by_official[str(i)]) for i in range(1, 9)], dtype=torch.float64)
    if torch.any(counts <= 0):
        raise ValueError(f"all training classes need positive pixel counts, got {counts.tolist()}")
    frequency = counts / counts.sum()
    weights = torch.rsqrt(frequency)
    weights = weights / weights.mean()
    weights = weights.clamp(min=minimum, max=maximum)
    return [float(x) for x in weights]


def build_loss(config: dict[str, Any], device: torch.device, project_root: Path) -> tuple[nn.Module, list[float] | None]:
    loss_cfg = config.get("loss", {})
    name = str(loss_cfg.get("name", "cross_entropy")).lower()
    ignore_index = int(config.get("ignore_index", 255))
    num_classes = int(config.get("num_classes", 8))
    weights: list[float] | None = None
    weight_tensor: torch.Tensor | None = None
    if name == "weighted_cross_entropy" or bool(loss_cfg.get("use_class_weights", False)):
        if loss_cfg.get("class_weights"):
            weights = [float(x) for x in loss_cfg["class_weights"]]
        else:
            stats_path = project_root / str(loss_cfg.get("split_statistics", "splits/split_statistics.json"))
            clip = loss_cfg.get("weight_clip", [0.5, 5.0])
            weights = class_weights_from_split(stats_path, minimum=float(clip[0]), maximum=float(clip[1]))
        if len(weights) != num_classes or any(not math.isfinite(x) or x <= 0 for x in weights):
            raise ValueError(f"invalid class weights: {weights}")
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    ce = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index)
    if name == "cross_entropy" or name == "weighted_cross_entropy":
        return ce, weights
    if name == "ce_dice":
        dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        return CrossEntropyDiceLoss(
            ce,
            dice,
            ce_weight=float(loss_cfg.get("ce_weight", 0.5)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        ), weights
    if name == "focal":
        return FocalLoss(gamma=float(loss_cfg.get("gamma", 2.0)), ignore_index=ignore_index, alpha=weight_tensor), weights
    raise ValueError(f"unsupported loss: {name}")
