#!/usr/bin/env python3
"""Standalone, bounded-memory D4 TTA anatomy with device-side aggregation.

The eleven historical schemes and D4 ordering are unchanged. Only the standard
mean-logit cache crosses to the CPU as a full float16 HWC array (once per image).
All view logits, scheme reductions, predictions and running statistics stay on
the selected device. No predictions or logits accumulate across images.

Production CUDA calls require CUDA_VISIBLE_DEVICES=2 and --device cuda:0. This
module never connects to another host or selects a GPU for the caller. Helpers
also accept CPU tensors for independent tests; --self-test never touches CUDA.

The model implementation is loaded from the same snapshot's src package as v1;
this file does not import tta_anatomy.py. Override --snapshot-root/--data-root to
use another local checkout. No output files are overwritten or cleaned up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
NUM_CLASSES = 8
IGNORE = 255
CROP = 1024
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
VIEW_NAMES = [f"rot{k*90}" + ("_mirror" if hf else "") for k, hf in VIEWS]
SCHEMES: dict[str, tuple[tuple[int, ...], str]] = {
    "single_canonical": ((0,), "logit"),
    "single_rot90": ((2,), "logit"),
    "single_rot180": ((4,), "logit"),
    "single_rot270": ((6,), "logit"),
    "mirror_pair": ((0, 1), "logit"),
    "rotations4": ((0, 2, 4, 6), "logit"),
    "subset2_mirror90": ((0, 1, 2, 3), "logit"),
    "subset4_mirror0_90": ((0, 1, 2, 3, 4, 5), "logit"),
    "subset6": ((0, 1, 2, 3, 4, 5, 6, 7)[:6], "logit"),
    "tta8_mean_logits": (tuple(range(8)), "logit"),
    "tta8_mean_probs": (tuple(range(8)), "prob"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def assert_cuda_mapping(device) -> None:
    """Fail closed before CUDA initialization; CPU is for helper tests only."""
    name = str(device)
    if name == "cpu":
        return
    if (name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "2"
            or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
        raise RuntimeError("CUDA requires CUDA_DEVICE_ORDER=PCI_BUS_ID, "
                           "CUDA_VISIBLE_DEVICES=2 and device cuda:0; "
                           "physical GPUs 0, 1 and 3 must not be used")


def _snapshot_imports() -> None:
    if str(SNAP) not in sys.path:
        sys.path.insert(0, str(SNAP))


def build_model(config_path, checkpoint_path, device):
    """Load the snapshot model and strict checkpoint; return an eval model only."""
    assert_cuda_mapping(device)
    import torch
    _snapshot_imports()
    from src.utils import load_config
    from src.models import build_model as make_model

    cfg = load_config(str(config_path))
    cfg["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0,
                   "norm_eps": 1.0e-12, "start_step": 1}
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = None
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if (isinstance(candidate, dict) and candidate
                    and all(isinstance(v, torch.Tensor) for v in candidate.values())):
                state = candidate
                break
        if state is None and checkpoint and all(isinstance(v, torch.Tensor)
                                                for v in checkpoint.values()):
            state = checkpoint
    if state is None:
        raise ValueError("Checkpoint has no nonempty tensor model state dictionary")
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model = make_model(cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def forward_views(model, px, view_batch_size):
    """Normalized 1x3xHxW -> canonical float32 8x8xHxW on px.device.

    Inverse mirror/rotation precedes bilinear resizing, exactly as in v1. Batches
    of 1/2/4/8 views are supported. Rectangular inputs are grouped by transformed
    shape within each batch rather than padded or geometrically altered.
    """
    import torch
    import torch.nn.functional as F

    assert_cuda_mapping(px.device)
    if view_batch_size not in (1, 2, 4, 8):
        raise ValueError("view_batch_size must be 1, 2, 4 or 8")
    if px.ndim != 4 or px.shape[:2] != (1, 3) or not px.is_floating_point():
        raise ValueError("px must be a normalized floating-point 1x3xHxW tensor")
    height, width = px.shape[-2:]
    if min(height, width) < 1:
        raise ValueError("px spatial dimensions must be positive")
    with torch.inference_mode():
        views = torch.empty((8, NUM_CLASSES, height, width), device=px.device,
                            dtype=torch.float32)
        for start in range(0, 8, view_batch_size):
            groups = {}
            for index in range(start, start + view_batch_size):
                rotation, mirror = VIEWS[index]
                transformed = torch.rot90(px, rotation, (-2, -1)) if rotation else px
                if mirror:
                    transformed = transformed.flip((-1,))
                groups.setdefault(tuple(transformed.shape[-2:]), []).append((index, transformed))
            for group in groups.values():
                batch = torch.cat([item[1] for item in group], dim=0)
                logits = model(pixel_values=batch).logits
                if (logits.ndim != 4 or logits.shape[:2] != (len(group), NUM_CLASSES)
                        or logits.device != px.device):
                    raise ValueError("Model must return Bx8xhxw logits on the input device")
                for offset, (index, _) in enumerate(group):
                    rotation, mirror = VIEWS[index]
                    aligned = logits[offset:offset + 1]
                    if mirror:
                        aligned = aligned.flip((-1,))
                    if rotation:
                        aligned = torch.rot90(aligned, -rotation, (-2, -1))
                    if tuple(aligned.shape[-2:]) != (height, width):
                        aligned = F.interpolate(aligned, size=(height, width), mode="bilinear",
                                                align_corners=False)
                    views[index].copy_(aligned[0].float())
        if not bool(torch.isfinite(views).all()):
            raise ValueError("Nonfinite model logits; no cache may be certified")
    return views


def _validate_views(views) -> None:
    import torch
    assert_cuda_mapping(views.device)
    if views.ndim != 4 or tuple(views.shape[:2]) != (8, NUM_CLASSES):
        raise ValueError("views must have shape 8x8xHxW")
    if views.dtype != torch.float32:
        raise ValueError("views must be float32")


def scheme_scores(views, mean_logits=None):
    """Yield (name, CHW scores) on device without retaining eleven score maps.

    Pass the already computed float32 all-view mean to reuse it for caching.
    Logit schemes return means of logits; the probability scheme averages each
    view's class softmax, never the softmax of the mean logits.
    """
    _validate_views(views)
    if mean_logits is None:
        mean_logits = views.mean(dim=0)
    for name, (members, mode) in SCHEMES.items():
        if mode == "prob":
            scores = views.softmax(dim=1).mean(dim=0)
        elif name == "tta8_mean_logits":
            scores = mean_logits
        elif len(members) == 1:
            scores = views[members[0]]
        else:
            scores = views[list(members)].mean(dim=0)
        yield name, scores


def scheme_predictions(views):
    """Return all eleven name -> uint8 HxW predictions on the views' device."""
    import torch
    with torch.inference_mode():
        return {name: scores.argmax(dim=0).to(torch.uint8)
                for name, scores in scheme_scores(views)}


def confusion_tensor(pred, gt):
    """Return device-side int64 [GT,pred] counts, ignoring invalid GT labels."""
    import torch
    valid = (gt >= 0) & (gt < NUM_CLASSES) & (gt != IGNORE)
    indices = gt[valid].long() * NUM_CLASSES + pred[valid].long()
    return torch.bincount(indices, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    diagonal = np.diag(cm).astype(np.float64)
    rows = cm.sum(axis=1).astype(np.float64)
    columns = cm.sum(axis=0).astype(np.float64)
    union = rows + columns - diagonal
    iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union > 0)
    recall = np.divide(diagonal, rows, out=np.zeros_like(diagonal), where=rows > 0)
    return {
        "mIoU": float(iou[union > 0].mean()) if (union > 0).any() else None,
        "pixel_accuracy": float(diagonal.sum() / cm.sum()) if cm.sum() else None,
        "per_class_iou": [float(v) if n > 0 else None for v, n in zip(iou, union)],
        "per_class_recall": [float(v) if n > 0 else None for v, n in zip(recall, rows)],
        "gt_pixels": [int(v) for v in rows],
        "pred_pixels": [int(v) for v in columns],
    }


class RunningStatistics:
    """Fixed-size GPU counters; pixel comparisons are discarded each image.

    Confusion uses valid GT only. Disagreement, entropy (natural logarithm) and
    confidence use all image pixels, preserving v1's disagreement denominator.
    """
    def __init__(self, device, domains=False):
        import torch
        self.names = list(SCHEMES)
        self.pairs = [(a, b) for a in range(len(self.names))
                      for b in range(a + 1, len(self.names))]
        self.cms = torch.zeros((len(self.names), NUM_CLASSES, NUM_CLASSES),
                               dtype=torch.int64, device=device)
        self.disagreements = torch.zeros(len(self.pairs), dtype=torch.int64, device=device)
        self.uncertainty = torch.zeros((len(self.names), 2), dtype=torch.float64, device=device)
        self.pixels = 0
        self.n_images = 0
        self.has_gt = False
        self.domain_cms = torch.zeros((3, len(self.names), NUM_CLASSES, NUM_CLASSES),
                                     dtype=torch.int64, device=device) if domains else None
        self.domain_images = [0, 0, 0]

    def update(self, views, mean_logits, gt=None, domain=None):
        import torch
        if self.domain_cms is not None and (domain not in (0, 1, 2) or gt is None):
            raise ValueError("Domain statistics require GT and a domain in 0,1,2")
        predictions = {}
        with torch.inference_mode():
            for index, (name, scores) in enumerate(scheme_scores(views, mean_logits)):
                prediction = scores.argmax(dim=0).to(torch.uint8)
                predictions[name] = prediction
                if gt is not None:
                    if tuple(gt.shape) != tuple(prediction.shape):
                        raise ValueError("GT and prediction spatial dimensions differ")
                    image_cm = confusion_tensor(prediction, gt)
                    self.cms[index] += image_cm
                    if self.domain_cms is not None:
                        self.domain_cms[domain, index] += image_cm
                probabilities = scores if SCHEMES[name][1] == "prob" else scores.softmax(dim=0)
                entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=0)
                self.uncertainty[index, 0] += entropy.sum(dtype=torch.float64)
                self.uncertainty[index, 1] += probabilities.amax(dim=0).sum(dtype=torch.float64)
            for index, (a, b) in enumerate(self.pairs):
                self.disagreements[index] += (predictions[self.names[a]] !=
                                              predictions[self.names[b]]).sum()
        self.has_gt |= gt is not None
        self.pixels += views.shape[-2] * views.shape[-1]
        self.n_images += 1
        if self.domain_cms is not None:
            self.domain_images[domain] += 1
        return predictions["tta8_mean_logits"]

    def result(self):
        cms = self.cms.cpu().numpy()
        disagreement_counts = self.disagreements.cpu().tolist()
        uncertainty = self.uncertainty.cpu().tolist()
        denominator = max(self.pixels, 1)
        domain_cms = self.domain_cms.cpu().numpy() if self.domain_cms is not None else None
        return {
            "domains": {f"D{d}": {"n_images": self.domain_images[d],
                                    "schemes": {name: metrics_from_confusion(domain_cms[d, i])
                                                for i, name in enumerate(self.names)}}
                        for d in range(3)} if domain_cms is not None else None,
            "schemes": {name: metrics_from_confusion(cms[i]) for i, name in enumerate(self.names)},
            "confusion_matrices": {name: cms[i].tolist() for i, name in enumerate(self.names)},
            "pairwise_disagreement": {
                f"{self.names[a]}_vs_{self.names[b]}": disagreement_counts[i] / denominator
                for i, (a, b) in enumerate(self.pairs)},
            "uncertainty": {
                name: {"mean_entropy_nats": uncertainty[i][0] / denominator,
                       "mean_confidence": uncertainty[i][1] / denominator}
                for i, name in enumerate(self.names)},
            "statistics_pixels": self.pixels,
            "statistics_scope": {"confusion": "valid_gt_only", "disagreement": "all_pixels",
                                 "entropy_confidence": "all_pixels"},
        }


BIAS_VARIANTS = {
    "no_bias": [0.0] * NUM_CLASSES,
    "FIXED8D_EXP41_STYLE": [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5],
    "LOCKED8D_69_276229": [-0.2, -0.0384, 0.0709, 0.7, 3.6, 0.1932, 3.3215, -0.5],
}


class PrecisionAudit:
    """True fp32 vs fp16 gate, with all three preregistered class biases.

    The optional reference_logits preserves the original pred-only test API,
    but a PASS is impossible unless original logits were supplied for every
    image. Confusion/count/error accumulators have constant dataset-size memory.
    """
    def __init__(self, device):
        import torch
        self.names = list(BIAS_VARIANTS)
        self.biases = {name: torch.tensor(values, dtype=torch.float32, device=device)[:, None, None]
                       for name, values in BIAS_VARIANTS.items()}
        self.sums = torch.zeros((3, 2), dtype=torch.float64, device=device)
        self.maxima = torch.zeros_like(self.sums)
        self.metric_images = torch.zeros((), dtype=torch.int64, device=device)
        self.class_changes = torch.zeros((3, NUM_CLASSES), dtype=torch.int64, device=device)
        self.class_pixels = torch.zeros_like(self.class_changes)
        self.cms32 = torch.zeros((3, NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
        self.cms16 = torch.zeros_like(self.cms32)
        self.logit_error_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.logit_error_max = torch.zeros_like(self.logit_error_sum)
        self.logit_values = 0
        self.n_images = 0
        self.reference_logits_n = 0
        self.disk_reload_n = 0

    def update(self, reference_pred, quantized_chw, gt, reference_logits=None):
        import torch
        if reference_logits is not None:
            if reference_logits.shape != quantized_chw.shape:
                raise ValueError("Precision reference and quantized logits must have identical shape")
            errors = (reference_logits - quantized_chw).abs()
            self.logit_error_sum += errors.sum(dtype=torch.float64)
            self.logit_error_max = torch.maximum(self.logit_error_max, errors.amax().double())
            self.logit_values += errors.numel()
            self.reference_logits_n += 1
        for index, name in enumerate(self.names):
            if index and reference_logits is None:
                continue
            bias = self.biases[name]
            original = (reference_logits + bias).argmax(0) if reference_logits is not None else reference_pred
            quantized = (quantized_chw + bias).argmax(0)
            changed = original != quantized
            disagreement = changed.double().mean()
            self.class_changes[index] += torch.bincount(original[changed].long(), minlength=NUM_CLASSES)
            self.class_pixels[index] += torch.bincount(original.reshape(-1).long(), minlength=NUM_CLASSES)
            if gt is None:
                delta = torch.zeros((), dtype=torch.float64, device=original.device)
            else:
                a = confusion_tensor(original, gt)
                b = confusion_tensor(quantized, gt)
                self.cms32[index] += a
                self.cms16[index] += b
                def miou(cm):
                    cm = cm.double()
                    diagonal = cm.diag()
                    union = cm.sum(0) + cm.sum(1) - diagonal
                    present = union > 0
                    return (diagonal / union.clamp_min(1)).sum() / present.sum().clamp_min(1)
                delta = (miou(a) - miou(b)).abs()
                if index == 0:
                    self.metric_images += (a.sum() > 0).long()
            values = torch.stack((disagreement, delta))
            self.sums[index] += values
            self.maxima[index] = torch.maximum(self.maxima[index], values)
        self.n_images += 1

    def result(self, checkpoint_sha, config_sha, cache_dir, required_disk_n):
        sums = self.sums.cpu().tolist()
        maxima = self.maxima.cpu().tolist()
        changes = self.class_changes.cpu().tolist()
        pixels = self.class_pixels.cpu().tolist()
        cms32, cms16 = self.cms32.cpu().numpy(), self.cms16.cpu().numpy()
        metric_images = int(self.metric_images.cpu())
        n = self.n_images
        evaluated = n > 0 and metric_images > 0 and required_disk_n > 0
        complete = n > 0 and self.reference_logits_n == n
        variants = {}
        for index, name in enumerate(self.names):
            a, b = metrics_from_confusion(cms32[index]), metrics_from_confusion(cms16[index])
            covered = n if index == 0 else self.reference_logits_n
            variants[name] = {
                "bias": BIAS_VARIANTS[name], "n_images": covered,
                "argmax_disagreement_mean": sums[index][0] / covered if covered else None,
                "argmax_disagreement_max": maxima[index][0] if covered else None,
                "miou_delta_mean": sums[index][1] / metric_images if metric_images and covered else None,
                "miou_delta_max": maxima[index][1] if metric_images and covered else None,
                "fp32_mIoU": a["mIoU"], "fp16_mIoU": b["mIoU"],
                "aggregate_miou_delta": abs(a["mIoU"] - b["mIoU"]) if a["mIoU"] is not None else None,
                "per_class_iou_delta": [abs(x - y) if x is not None and y is not None else None
                                        for x, y in zip(a["per_class_iou"], b["per_class_iou"])],
                "per_reference_class_changed_pixels": changes[index],
                "per_reference_class_pixels": pixels[index],
                "per_reference_class_disagreement": [c / p if p else None
                                                       for c, p in zip(changes[index], pixels[index])],
            }
        maximum_disagreement = max(values[0] for values in maxima)
        maximum_delta = max(values[1] for values in maxima)
        passed = (evaluated and complete and self.disk_reload_n == required_disk_n
                  and maximum_disagreement < 1e-4 and maximum_delta < 1e-4
                  and all(v["aggregate_miou_delta"] is not None
                          and v["aggregate_miou_delta"] < 1e-4 for v in variants.values()))
        return {
            "status": "PASS" if passed else ("FAIL" if evaluated else "NOT_EVALUATED"),
            "evaluated": evaluated, "n_images": n, "n": n,
            "reference_logits_n": self.reference_logits_n, "variants": variants,
            "mean_abs_logit_error": float(self.logit_error_sum.cpu()) / self.logit_values
                                    if self.logit_values else None,
            "max_abs_logit_error": float(self.logit_error_max.cpu()) if self.logit_values else None,
            "metric_images": metric_images, "disk_reload_n": self.disk_reload_n,
            "disk_reload_expected_n": required_disk_n,
            "argmax_disagreement_mean": max(row[0] for row in sums) / n if n else None,
            "argmax_disagreement_max": maximum_disagreement if n else None,
            "miou_delta_mean": max(row[1] for row in sums) / metric_images if metric_images else None,
            "miou_delta_max": maximum_delta if metric_images else None,
            "threshold": 1e-4, "comparison": "true_fp32_before_quantization_vs_fp16_roundtrip_all3biases",
            "disk_reload_verification": "exact_fp16_array_equality_first_n_images",
            "checkpoint_sha256": checkpoint_sha, "config_sha256": config_sha,
            "cache_dir": str(cache_dir) if cache_dir else None,
            "exact_fallback_available": False,
            "downstream_cache_use_allowed": bool(passed),
        }


def image_list(split: str, snapshot_root=None, data_root=None):
    snapshot_root = Path(snapshot_root) if snapshot_root is not None else SNAP
    data_root = Path(data_root) if data_root is not None else DATA
    if split == "val":
        ids = [line.strip() for line in (snapshot_root / "splits" / "val.txt").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError("Validation split contains duplicate image IDs")
        if any(Path(i).name != i or "/" in i or "\\" in i or i in (".", "..") for i in ids):
            raise ValueError("Validation IDs must be simple file stems")
        imgs = [data_root / "train" / "images" / f"{i}.png" for i in ids]
        masks = [data_root / "train" / "masks" / f"{i}.png" for i in ids]
    else:
        imgs = sorted((data_root / "images").glob("*.png"),
                      key=lambda p: int(p.stem.split("_")[-1]))
        if len(imgs) != 500:
            raise ValueError(f"Expected 500 test images, found {len(imgs)}")
        masks = None
    if not imgs:
        raise ValueError("Empty image split")
    for path in imgs + (masks or []):
        if not path.is_file():
            raise FileNotFoundError(path)
    return imgs, masks


def load_assignments(path, image_ids):
    import csv
    mapping = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"image_name", "split", "domain_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Assignments require image_name,split,domain_id columns")
        for row in reader:
            if row["split"].strip() != "val":
                continue
            name = row["image_name"].strip()
            if name in mapping:
                raise ValueError(f"Duplicate validation domain assignment: {name}")
            domain = int(row["domain_id"])
            if domain not in (0, 1, 2):
                raise ValueError(f"Domain must be 0, 1 or 2: {name}")
            mapping[name] = domain
    missing = set(image_ids) - set(mapping)
    if missing:
        raise ValueError(f"Missing validation domain assignments: {sorted(missing)[:10]}")
    if {mapping[name] for name in image_ids} != {0, 1, 2}:
        raise ValueError("Full validation assignments must include all three domains")
    return mapping


def _verified_sha(path, expected, kind):
    actual = sha256_file(path)
    if expected is not None:
        if len(expected) != 64 or any(c not in "0123456789abcdefABCDEF" for c in expected):
            raise ValueError(f"Expected {kind} SHA must be 64 hexadecimal characters")
        if actual != expected.lower():
            raise ValueError(f"{kind} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def _write_json_exclusive(path, result):
    # Serializing first prevents a nonfinite statistic from creating a partial report.
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(text)


def numpy_self_test() -> int:
    """Dependency-light registration/JSON tests; not a substitute for torch tests."""
    assert len(SCHEMES) == 11 and len(VIEWS) == 8
    assert VIEWS == [(0, False), (0, True), (1, False), (1, True),
                     (2, False), (2, True), (3, False), (3, True)]
    assert SCHEMES["subset6"] == SCHEMES["subset4_mirror0_90"]
    for members, mode in SCHEMES.values():
        assert members and all(0 <= i < 8 for i in members)
        assert mode in ("logit", "prob")
    empty = metrics_from_confusion(np.zeros((8, 8), dtype=np.int64))
    assert empty["mIoU"] is None and empty["pixel_accuracy"] is None
    perfect = metrics_from_confusion(np.eye(8, dtype=np.int64))
    assert perfect["mIoU"] == 1.0 and perfect["pixel_accuracy"] == 1.0
    json.dumps([empty, perfect], allow_nan=False)
    print("NUMPY_SELF_TEST_OK (registration/metrics/strict JSON only; no torch execution)")
    return 0


def self_test() -> int:
    """Exercise all production aggregators and D4 batches on CPU, never CUDA."""
    numpy_self_test()
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace

    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    cpu_views = rng.normal(size=(8, 8, 7, NUM_CLASSES)).astype(np.float32)
    views = torch.from_numpy(cpu_views.transpose(0, 3, 1, 2).copy())
    predictions = scheme_predictions(views)
    for name, scores in scheme_scores(views):
        members, mode = SCHEMES[name]
        stack = np.stack([cpu_views[i] for i in members])
        if mode == "prob":
            exponentials = np.exp(stack - stack.max(-1, keepdims=True))
            stack = exponentials / exponentials.sum(-1, keepdims=True)
        reference = stack.mean(0)
        np.testing.assert_allclose(scores.permute(1, 2, 0).numpy(), reference,
                                   atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(predictions[name].numpy(), reference.argmax(-1))
        assert predictions[name].dtype == torch.uint8

    class ToyModel:
        def __call__(self, pixel_values):
            # Coordinate-sensitive, low-resolution output tests inverse + resize.
            x = F.avg_pool2d(pixel_values, 2, 2)
            ramp = torch.arange(x.shape[-1], dtype=x.dtype).reshape(1, 1, 1, -1) / 13
            logits = torch.cat([x, x[:, :1] * 2, -x, x[:, 1:2] + ramp], dim=1)
            return SimpleNamespace(logits=logits)

    for height, width in ((12, 12), (10, 14)):
        px = torch.from_numpy(rng.normal(size=(1, 3, height, width)).astype(np.float32))
        model = ToyModel()
        reference = []
        for rotation, mirror in VIEWS:
            transformed = torch.rot90(px, rotation, (-2, -1))
            if mirror:
                transformed = transformed.flip((-1,))
            aligned = model(pixel_values=transformed).logits
            if mirror:
                aligned = aligned.flip((-1,))
            aligned = torch.rot90(aligned, -rotation, (-2, -1))
            aligned = F.interpolate(aligned, (height, width), mode="bilinear", align_corners=False)
            reference.append(aligned[0])
        reference = torch.stack(reference)
        for batch_size in (1, 2, 4, 8):
            torch.testing.assert_close(forward_views(model, px, batch_size), reference,
                                       atol=1e-6, rtol=1e-6)

    gt_array = rng.integers(0, NUM_CLASSES, size=(8, 7), dtype=np.int64)
    gt_array[0, :3] = [IGNORE, -1, NUM_CLASSES]
    gt = torch.from_numpy(gt_array)
    stats = RunningStatistics("cpu")
    mean = views.mean(0)
    stats.update(views, mean, gt)
    stats.update(views, mean, gt)
    result = stats.result()
    valid = (gt_array >= 0) & (gt_array < NUM_CLASSES)
    for name, pred in predictions.items():
        indices = gt_array[valid] * NUM_CLASSES + pred.numpy()[valid]
        expected = np.bincount(indices, minlength=64).reshape(8, 8) * 2
        np.testing.assert_array_equal(result["confusion_matrices"][name], expected)
    for a, b in stats.pairs:
        key = f"{stats.names[a]}_vs_{stats.names[b]}"
        expected = float((predictions[stats.names[a]] != predictions[stats.names[b]]).double().mean())
        assert result["pairwise_disagreement"][key] == expected
    assert result["statistics_pixels"] == 112
    json.dumps(result, allow_nan=False)
    domain_stats = RunningStatistics("cpu", domains=True)
    for domain in range(3):
        domain_stats.update(views, mean, gt, domain=domain)
    domain_result = domain_stats.result()
    assert all(domain_result["domains"][f"D{d}"]["n_images"] == 1 for d in range(3))
    np.testing.assert_array_equal(domain_stats.domain_cms.sum(0).numpy(), domain_stats.cms.numpy())

    # Deliberate near-tie changes argmax after fp16. A pre-quantized reference
    # (the old implementation's bug) would wrongly report zero disagreement.
    near_tie = torch.zeros((8, 2, 2), dtype=torch.float32)
    near_tie[0] = 1.0
    near_tie[1] = 1.0001
    audit = PrecisionAudit("cpu")
    audit.update(near_tie.argmax(0), near_tie.half().float(), torch.ones((2, 2), dtype=torch.long))
    audit.disk_reload_n = 1
    failed = audit.result("a" * 64, "b" * 64, Path("cache"), 1)
    assert failed["status"] == "FAIL" and failed["argmax_disagreement_max"] == 1.0
    full_audit = PrecisionAudit("cpu")
    full_audit.update(near_tie.argmax(0), near_tie.half().float(),
                      torch.ones((2, 2), dtype=torch.long), reference_logits=near_tie)
    full_audit.disk_reload_n = 1
    full_failed = full_audit.result("a" * 64, "b" * 64, Path("cache"), 1)
    assert full_failed["status"] == "FAIL" and full_failed["max_abs_logit_error"] > 0
    assert set(full_failed["variants"]) == set(BIAS_VARIANTS)
    exact = torch.zeros((8, 2, 2), dtype=torch.float32)
    exact[0] = 20.0
    exact_audit = PrecisionAudit("cpu")
    exact_audit.update(exact.argmax(0), exact.half().float(),
                       torch.zeros((2, 2), dtype=torch.long), reference_logits=exact)
    exact_audit.disk_reload_n = 1
    passed = exact_audit.result("a" * 64, "b" * 64, Path("cache"), 1)
    assert passed["status"] == "PASS" and passed["max_abs_logit_error"] == 0.0
    assert all(v["argmax_disagreement_max"] == 0 for v in passed["variants"].values())
    empty_audit = PrecisionAudit("cpu").result("a" * 64, "b" * 64, None, 0)
    assert empty_audit["status"] == "NOT_EVALUATED"
    json.dumps([failed, empty_audit], allow_nan=False)
    print("SELF_TEST_OK schemes=11 views=8 batches=1,2,4,8 device=cpu true_fp16_gate=ok")
    return 0


def main(argv=None) -> int:
    global SNAP, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--view-batch-size", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--expected-checkpoint-sha")
    parser.add_argument("--expected-config-sha")
    parser.add_argument("--expected-split-sha")
    parser.add_argument("--expected-val-count", type=int, default=1399)
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--cache-mean-logits", action="store_true")
    parser.add_argument("--precision-gate-n", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--snapshot-root", type=Path, default=SNAP)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--numpy-self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.numpy_self_test:
        return numpy_self_test()
    if not all((args.config, args.checkpoint, args.out_dir)):
        parser.error("--config, --checkpoint and --out-dir are required for inference")
    if args.limit < 0 or args.precision_gate_n < 0:
        parser.error("--limit and --precision-gate-n must be nonnegative")
    if args.device != "cuda:0":
        parser.error("Inference requires --device cuda:0 with CUDA_VISIBLE_DEVICES=2")
    assert_cuda_mapping(args.device)
    SNAP, DATA = args.snapshot_root.resolve(), args.data_root.resolve()
    checkpoint_sha = _verified_sha(args.checkpoint, args.expected_checkpoint_sha, "checkpoint")
    config_sha = _verified_sha(args.config, args.expected_config_sha, "config")
    split_sha = (_verified_sha(SNAP / "splits" / "val.txt", args.expected_split_sha, "split")
                 if args.split == "val" else None)
    imgs, masks = image_list(args.split)
    if args.split == "val" and not args.limit and len(imgs) != args.expected_val_count:
        raise ValueError(f"Expected {args.expected_val_count} validation images, found {len(imgs)}")
    if args.assignments is not None and args.split != "val":
        parser.error("--assignments is only supported with --split val")
    assignments = load_assignments(args.assignments, [p.stem for p in imgs]) if args.assignments else None
    assignments_sha = sha256_file(args.assignments) if args.assignments else None
    if args.limit:
        imgs = imgs[:args.limit]
        masks = masks[:args.limit] if masks is not None else None
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"Output directory must be new or empty; refusing to overwrite: {out_dir}")

    import torch
    from PIL import Image
    _snapshot_imports()
    from src.transforms import normalize_to_tensor
    if masks is not None:
        from src.label_mapping import official_to_train

    device = torch.device(args.device)
    model = build_model(args.config, args.checkpoint, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "mean_logits_fp16" if args.cache_mean_logits else None
    if cache_dir is not None:
        cache_dir.mkdir(exist_ok=False)
    stats = RunningStatistics(device, domains=assignments is not None)
    audit = PrecisionAudit(device)
    required_disk_n = min(args.precision_gate_n, len(imgs)) if cache_dir is not None else 0
    cache_shape = None
    cache_file_sha256 = {}
    timings = {"decode_and_preprocess": 0.0, "forward_views": 0.0,
               "aggregation": 0.0, "cache_and_precision": 0.0}
    started = time.perf_counter()
    with torch.inference_mode():
        for index, image_path in enumerate(imgs):
            stage_started = time.perf_counter()
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("RGB")).copy()
            if tuple(pixels.shape) != (CROP, CROP, 3):
                raise ValueError(f"Production images must be {CROP}x{CROP} RGB: {image_path}")
            gt = None
            if masks is not None:
                with Image.open(masks[index]) as mask:
                    mapped = official_to_train(np.asarray(mask).copy())
                gt = torch.from_numpy(np.asarray(mapped, dtype=np.int64).copy()).to(device)
            px = normalize_to_tensor(pixels).unsqueeze(0).to(device)
            timings["decode_and_preprocess"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            views = forward_views(model, px, args.view_batch_size)
            timings["forward_views"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            mean_logits = views.mean(dim=0)  # exactly once, reused by scores/cache/gate
            reference_pred = stats.update(views, mean_logits, gt,
                                          domain=assignments[image_path.stem] if assignments else None)
            timings["aggregation"] += time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            if cache_dir is not None:
                shape = [int(mean_logits.shape[1]), int(mean_logits.shape[2]), NUM_CLASSES]
                if cache_shape is not None and cache_shape != shape:
                    raise ValueError("Mixed image dimensions cannot share this cache manifest")
                cache_shape = shape
                mean_fp16 = mean_logits.to(torch.float16)
                if not bool(torch.isfinite(mean_fp16).all()):
                    raise ValueError("Mean logits overflow float16; cache certification blocked")
                # The sole full-image device->host transfer; HWC matches v1 cache.
                cache_array = mean_fp16.permute(1, 2, 0).contiguous().cpu().numpy()
                cache_path = cache_dir / f"{image_path.stem}.npy"
                with cache_path.open("xb") as handle:
                    np.save(handle, cache_array, allow_pickle=False)
                cache_file_sha256[cache_path.name] = sha256_file(cache_path)
                if index < required_disk_n:
                    disk_array = np.load(cache_path, allow_pickle=False)
                    if (disk_array.dtype != np.float16 or disk_array.shape != tuple(shape)
                            or not np.isfinite(disk_array).all()
                            or not np.array_equal(cache_array, disk_array)):
                        raise ValueError(f"Saved fp16 cache differs from GPU quantization: {cache_path}")
                    audit.disk_reload_n += 1
                    del disk_array
                audit.update(reference_pred, mean_fp16.float(), gt, reference_logits=mean_logits)
                del cache_array, mean_fp16
            timings["cache_and_precision"] += time.perf_counter() - stage_started
            del px, views, mean_logits, reference_pred, gt
            if (index + 1) % 100 == 0 or index + 1 == len(imgs):
                # Only scalar progress synchronization; no retained image tensors.
                torch.cuda.synchronize(device)
                print(f"[{index + 1}/{len(imgs)}] {time.perf_counter() - started:.1f}s", flush=True)
    report = stats.result()
    precision = audit.result(checkpoint_sha, config_sha, cache_dir, required_disk_n)
    if args.split == "val":
        _verified_sha(SNAP / "splits" / "val.txt", split_sha, "split_after_run")
    _verified_sha(args.checkpoint, checkpoint_sha, "checkpoint_after_run")
    _verified_sha(args.config, config_sha, "config_after_run")
    if args.assignments is not None:
        _verified_sha(args.assignments, assignments_sha, "assignments_after_run")
    precision["split_sha256"] = split_sha
    report.update({
        "engine": "tta_anatomy_v2", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha, "config": str(args.config.resolve()),
        "config_sha256": config_sha, "split": args.split, "split_sha256": split_sha,
        "n_images": len(imgs),
        "assignments": str(args.assignments.resolve()) if args.assignments else None,
        "assignments_sha256": assignments_sha,
        "image_list_sha256": sha256_lines([path.stem for path in imgs]),
        "class_order": "train 0..7 = Background,Building,Road,Water,Barren,Vegetation,Agricultural,Vehicle",
        "views": VIEW_NAMES, "view_batch_size": args.view_batch_size,
        "scheme_definitions": {name: {"members": list(members), "mode": mode}
                               for name, (members, mode) in SCHEMES.items()},
        "tta_definition": "D4: 4 rotations x 2 mirrors, inverse-transformed, mean logits/probs",
        "normalization": "ImageNet mean/std via src.transforms.normalize_to_tensor",
        "device": str(device), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_diagnostics": {
            "logical_device": str(device), "physical_device": 2,
            "model_parameter_devices": sorted({str(p.device) for p in model.parameters()}),
            "model_buffer_devices": sorted({str(b.device) for b in model.buffers()}),
            "model_eval": not model.training, "autocast_enabled": False,
            "torch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        },
        "precision_audit": precision, "fp16_precision_gate": precision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "cache_file_sha256": cache_file_sha256,
        "host_stage_seconds": timings,
        "timing_note": "Host enqueue/wait wall times, not isolated GPU kernel timings; "
                       "finite/cache transfers and final synchronization absorb outstanding work",
        "mean_logits_cache": {
            "dir": str(cache_dir) if cache_dir else None, "dtype": "float16", "shape": cache_shape,
            "note": "standard TTA8 mean logits only; no per-view logits persisted",
            "certified": precision["status"] == "PASS"},
        "elapsed_seconds": time.perf_counter() - started,
    })
    _write_json_exclusive(out_dir / "TTA_ANATOMY.json", report)
    lines = ["# TTA Anatomy v2", "", f"- checkpoint SHA256: `{checkpoint_sha}`",
             f"- config SHA256: `{config_sha}`", f"- split SHA256: `{split_sha}`",
             f"- images: {len(imgs)}", f"- view batch size: {args.view_batch_size}",
             f"- fp16 certification: **{precision['status']}**", "",
             "| Scheme | mIoU | Pixel accuracy |", "|---|---:|---:|"]
    for name, metrics in report["schemes"].items():
        values = ["N/A" if metrics[key] is None else f"{metrics[key]:.8f}"
                  for key in ("mIoU", "pixel_accuracy")]
        lines.append(f"| {name} | {values[0]} | {values[1]} |")
    lines += ["", "Detailed confusion, pairwise disagreement, entropy/confidence, three-bias",
              "precision audit and cache-file hashes are in TTA_ANATOMY.json.", "",
              "Cache must not be consumed unless precision_audit.status is PASS."]
    with (out_dir / "TTA_ANATOMY.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("FP16_GATE", json.dumps(precision, allow_nan=False), flush=True)
    print("TTA_ANATOMY_DONE", out_dir, flush=True)
    if cache_dir is not None and masks is not None and precision["status"] != "PASS":
        print("CACHE_NOT_CERTIFIED: block downstream cache use; no exact fallback was produced",
              file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
