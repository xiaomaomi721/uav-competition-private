#!/usr/bin/env python3
"""Fail-closed MMSeg Mask2Former adapter for SHBR donor probabilities.

The donor score space is kept separate from SegFormer logits. Bstar is never
accepted or applied here.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

import shbr_common as common


def _normalize_class_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _unwrap_data(value: object) -> torch.Tensor:
    if hasattr(value, "data"):
        value = value.data
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    return value


class M2FDonorAdapter:
    def __init__(self, config_path: Path, checkpoint_path: Path, settings: Mapping, device: str = "cuda:0"):
        if settings.get("score_kind") != "nonnegative_semantic_mixture":
            raise RuntimeError("donor score_kind must explicitly be nonnegative_semantic_mixture")
        if settings.get("bstar_applied", False):
            raise RuntimeError("Bstar is forbidden in the M2F donor score space")
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"missing donor config/checkpoint: {config_path} {checkpoint_path}")
        from mmseg.apis import init_model

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = init_model(str(config_path), str(checkpoint_path), device=device)
        self.model.eval()
        expected = tuple(settings.get("class_order", common.CLASS_NAMES))
        if len(expected) != common.NUM_CLASSES:
            raise RuntimeError(f"expected exactly eight donor class names, got {expected}")
        metadata = getattr(self.model, "dataset_meta", None) or {}
        actual = metadata.get("classes")
        if actual is None:
            raise RuntimeError("donor model has no dataset_meta.classes; class order cannot be verified")
        if tuple(map(_normalize_class_name, actual)) != tuple(map(_normalize_class_name, expected)):
            raise RuntimeError(f"donor class order mismatch: actual={actual} expected={expected}")
        self.class_order = tuple(str(item) for item in actual)

    def _one_view(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        from mmseg.apis import inference_model

        result = inference_model(self.model, image_bgr)
        if isinstance(result, (tuple, list)):
            if len(result) != 1:
                raise RuntimeError(f"unexpected donor result count: {len(result)}")
            result = result[0]
        if not hasattr(result, "seg_logits") or not hasattr(result, "pred_sem_seg"):
            raise RuntimeError("donor API must expose both seg_logits and pred_sem_seg; hard labels alone are rejected")
        scores = _unwrap_data(result.seg_logits).float()
        prediction = _unwrap_data(result.pred_sem_seg).long()
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if scores.ndim != 3 or scores.shape[0] != common.NUM_CLASSES:
            raise RuntimeError(f"donor dense score shape must be [8,H,W], got {tuple(scores.shape)}")
        if prediction.ndim != 2 or tuple(prediction.shape) != tuple(scores.shape[-2:]):
            raise RuntimeError(f"donor prediction shape mismatch: score={tuple(scores.shape)} pred={tuple(prediction.shape)}")
        if not torch.isfinite(scores).all():
            raise FloatingPointError("non-finite donor semantic mixture")
        minimum = float(scores.min().item())
        if minimum < -1e-7:
            raise RuntimeError(f"declared nonnegative donor mixture contains negative value {minimum}")
        scores = scores.clamp_min(0)
        denominator = scores.sum(0, keepdim=True)
        if torch.any(denominator <= 0):
            raise RuntimeError("donor semantic mixture has zero class sum")
        probabilities = scores / denominator
        mismatch = int((probabilities.argmax(0) != prediction).sum().item())
        if mismatch:
            raise RuntimeError(f"donor argmax parity failed on {mismatch} pixels")
        return probabilities, prediction

    def infer_d4(self, image_path: Path, native_size: tuple[int, int] = (256, 256)) -> dict[str, np.ndarray]:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required to preserve MMSeg BGR preprocessing") from error
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"failed to decode donor image: {image_path}")
        base = torch.from_numpy(np.ascontiguousarray(image_bgr)).permute(2, 0, 1).unsqueeze(0)
        canonical_probabilities = []
        canonical_votes = []
        for view in range(8):
            transformed = common.d4_forward(base, view)[0].permute(1, 2, 0).contiguous().cpu().numpy()
            probabilities, _ = self._one_view(transformed)
            canonical = common.d4_inverse(probabilities[None], view)
            canonical = F.interpolate(canonical, size=native_size, mode="bilinear", align_corners=False)[0]
            canonical = canonical / canonical.sum(0, keepdim=True).clamp_min(1e-12)
            canonical_probabilities.append(canonical)
            canonical_votes.append(canonical.argmax(0))
        stacked = torch.stack(canonical_probabilities)
        mean = stacked.mean(0, dtype=torch.float32)
        mean = mean / mean.sum(0, keepdim=True).clamp_min(1e-12)
        prediction = mean.argmax(0)
        votes = torch.stack(canonical_votes)
        agreement = (votes == prediction[None]).to(torch.uint8).sum(0)
        top2 = mean.topk(2, dim=0).values
        margin = top2[0] - top2[1]
        if not torch.isfinite(mean).all() or not torch.isfinite(margin).all():
            raise FloatingPointError("non-finite D4 donor outputs")
        return {
            "probabilities": mean.cpu().numpy().astype(np.float32),
            "prediction": prediction.cpu().numpy().astype(np.uint8),
            "agreement": agreement.cpu().numpy().astype(np.uint8),
            "margin": margin.cpu().numpy().astype(np.float32),
        }

    def receipt(self) -> dict:
        return {
            "adapter": "mmseg-mask2former-nonnegative-semantic-mixture",
            "config": str(self.config_path),
            "config_sha256": common.sha256_file(self.config_path),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": common.sha256_file(self.checkpoint_path),
            "class_order": list(self.class_order),
            "score_kind": "nonnegative_semantic_mixture_normalized_across_8_classes",
            "no_object_handling": "already removed by the model semantic mixture; exactly eight channels required",
            "d4": "8 views; exact inverse; bilinear to native 256; FP32 probability mean",
            "bstar_applied": False,
        }
