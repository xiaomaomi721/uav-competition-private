#!/usr/bin/env python3
"""Shared contracts for Target Complement Anchor (TCA).

TCA keeps Arm B's A05 teacher, confidence mask, augmentation, trainable
scope, optimizer, and endpoint.  It removes source CE and uses a weak KL on
the complement of the original mask as a target-domain anti-drift anchor.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

EXPECTED_A05_SHA = "023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134"
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
VIEWS = tuple((rotation, horizontal_flip) for rotation in range(4) for horizontal_flip in (False, True))
BLIND_CLASSES = (3, 4, 6)
CONF_NORMAL = 0.98
CONF_BLIND = 0.50
SEED = 20260922
BASE_LR = 1e-5
HEAD_MULT = 0.05
WEIGHT_DECAY = 0.01
RAMP_STEPS = 500
LR_DENOMINATOR = 10000
FORMULA_VERSION = "tca-v4-xlogy-zero-safe-all-pixels-selected-plus-gradient-calibrated-complement"
TRAIN_PREFIXES = ("segformer.stages.2.", "segformer.stages.3.")
HEAD_PREFIX = "decode_head."


@dataclass(frozen=True)
class ServerPaths:
    root: Path
    src: Path
    config: Path
    model_source: Path
    a05: Path
    test2: Path
    teacher_cache: Path

    @classmethod
    def from_root(cls, root: Path) -> "ServerPaths":
        root = root.resolve()
        return cls(
            root=root,
            src=root / "code" / "src",
            config=root / "code" / "config.yaml",
            model_source=root / "models" / "segformer-b3-ade512",
            a05=root / "checkpoints" / "A05_WEIGHT_alpha05.pt",
            test2=root / "datasets" / "test_2" / "images",
            teacher_cache=root / "teacher_cache" / "scores",
        )


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exact_test2_names(count: int = 1300) -> list[str]:
    return [f"test2_{index}.png" for index in range(1, count + 1)]


def exact_cache_names(count: int = 1300) -> list[str]:
    return [f"test2_{index}.npy" for index in range(1, count + 1)]


def validate_exact_test2_images(directory: Path, count: int = 1300) -> list[str]:
    expected = exact_test2_names(count)
    actual = sorted(path.name for path in directory.iterdir() if path.is_file())
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"test2 identity mismatch: missing={missing[:5]} extra={extra[:5]}")
    return expected


def load_state(path: Path) -> Mapping[str, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model", "student_model", "model_state_dict", "state_dict"):
            value = obj.get(key)
            if isinstance(value, dict) and value and all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
                return value
        if obj and all(isinstance(value, torch.Tensor) for value in obj.values()):
            return obj
    raise RuntimeError(f"no model state dict found in {path}")


def normalize_uuid(value: object) -> str:
    text = str(value)
    return text if text.startswith("GPU-") else f"GPU-{text}"


def verify_gpu(physical_gpu: int, expected_uuid: str) -> dict:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must be exactly {physical_gpu}, got {visible!r}; "
            "bind the physical GPU before launching Python"
        )
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    matched = [row for row in rows if row.split(",", 1)[0].strip() == str(physical_gpu)]
    if len(matched) != 1 or expected_uuid not in matched[0]:
        raise RuntimeError(f"physical GPU identity mismatch: expected index={physical_gpu} uuid={expected_uuid}; rows={rows}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")
    torch_uuid = normalize_uuid(torch.cuda.get_device_properties(0).uuid)
    if torch_uuid != expected_uuid:
        raise RuntimeError(f"PyTorch GPU UUID mismatch: expected {expected_uuid}, got {torch_uuid}")
    return {
        "physical_gpu": physical_gpu,
        "expected_uuid": expected_uuid,
        "torch_uuid": torch_uuid,
        "nvidia_smi_row": matched[0],
        "logical_device": "cuda:0",
    }


def validate_server_paths(paths: ServerPaths) -> None:
    required_files = (paths.config, paths.a05)
    required_dirs = (paths.src, paths.model_source, paths.test2, paths.teacher_cache)
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in required_dirs if not path.is_dir())
    if missing:
        raise RuntimeError(f"missing required server paths: {missing}")
    actual_sha = sha256_file(paths.a05)
    if actual_sha != EXPECTED_A05_SHA:
        raise RuntimeError(f"A05 SHA mismatch: expected {EXPECTED_A05_SHA}, got {actual_sha}")
    validate_exact_test2_images(paths.test2)


def validate_teacher_cache(cache_dir: Path, full_scan: bool = True) -> dict:
    expected = exact_cache_names()
    actual = sorted(path.name for path in cache_dir.iterdir() if path.is_file())
    if actual != sorted(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"teacher cache identity mismatch: missing={missing[:5]} extra={extra[:5]}")

    chain = hashlib.sha256()
    checked = 0
    for name in expected:
        path = cache_dir / name
        payload_sha = sha256_file(path)
        chain.update(f"{name}\t{path.stat().st_size}\t{payload_sha}\n".encode("utf-8"))
        if full_scan:
            array = np.load(path, allow_pickle=False)
            if array.shape != (8, 256, 256) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise RuntimeError(
                    f"invalid teacher cache {name}: shape={array.shape} dtype={array.dtype} finite={np.isfinite(array).all()}"
                )
            checked += 1
    return {
        "count": len(expected),
        "full_scan": full_scan,
        "arrays_checked": checked,
        "content_manifest_sha256": chain.hexdigest(),
    }


def d4_forward(tensor: torch.Tensor, view: int) -> torch.Tensor:
    rotation, horizontal_flip = divmod(view, 2)
    if rotation:
        tensor = torch.rot90(tensor, rotation, (-2, -1))
    if horizontal_flip:
        tensor = tensor.flip((-1,))
    return tensor


def d4_inverse(tensor: torch.Tensor, view: int) -> torch.Tensor:
    rotation, horizontal_flip = divmod(view, 2)
    if horizontal_flip:
        tensor = tensor.flip((-1,))
    if rotation:
        tensor = torch.rot90(tensor, -rotation, (-2, -1))
    return tensor.contiguous()


def strong_photo(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device, dtype=tensor.dtype)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device, dtype=tensor.dtype)[None, :, None, None]
    tensor = (tensor * std + mean).clamp(0, 1)
    brightness = 1 + random.uniform(-0.15, 0.15)
    contrast = 1 + random.uniform(-0.15, 0.15)
    saturation = 1 + random.uniform(-0.12, 0.12)
    gamma = random.uniform(0.9, 1.1)
    tensor = tensor * brightness
    spatial_mean = tensor.mean((-2, -1), keepdim=True)
    tensor = (tensor - spatial_mean) * contrast + spatial_mean
    gray = tensor.mean(1, keepdim=True)
    tensor = (tensor - gray) * saturation + gray
    tensor = tensor.clamp(0, 1).pow(gamma)
    tensor = tensor + torch.randn_like(tensor) * random.uniform(0, 0.015)
    if random.random() < 0.25:
        tensor = F.avg_pool2d(tensor, 3, 1, 1)
    return (tensor.clamp(0, 1) - mean) / std


def is_trainable_parameter(name: str) -> bool:
    return name.startswith(TRAIN_PREFIXES) or name.startswith(HEAD_PREFIX)


def configure_trainable_scope(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    model.eval()
    model.get_submodule("segformer.stages.2").train()
    model.get_submodule("segformer.stages.3").train()
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = is_trainable_parameter(name)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("no trainable parameters matched TCA scope")
    return trainable


def split_parameter_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    base = [parameter for name, parameter in model.named_parameters() if name.startswith(TRAIN_PREFIXES)]
    head = [parameter for name, parameter in model.named_parameters() if name.startswith(HEAD_PREFIX)]
    if not base or not head:
        raise RuntimeError(f"invalid optimizer groups: base={len(base)} head={len(head)}")
    return base, head


def teacher_masks(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if probabilities.ndim != 4 or probabilities.shape[1] != 8:
        raise ValueError(f"teacher probabilities must be [N,8,H,W], got {tuple(probabilities.shape)}")
    confidence, predicted = probabilities.max(1)
    blind = torch.zeros_like(predicted, dtype=torch.bool)
    for class_id in BLIND_CLASSES:
        blind |= predicted == class_id
    selected = (confidence >= CONF_NORMAL) | (blind & (confidence >= CONF_BLIND))
    complement = ~selected
    return confidence, predicted, selected, complement


def per_pixel_kl(teacher_probabilities: torch.Tensor, student_log_probabilities: torch.Tensor) -> torch.Tensor:
    if teacher_probabilities.shape != student_log_probabilities.shape:
        raise ValueError(
            f"KL shape mismatch: teacher={tuple(teacher_probabilities.shape)} student={tuple(student_log_probabilities.shape)}"
        )
    values = (torch.xlogy(teacher_probabilities, teacher_probabilities) - teacher_probabilities * student_log_probabilities).sum(1)
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite per-pixel KL")
    return values


def masked_mean(values: torch.Tensor, mask: torch.Tensor, label: str) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(f"{label} shape mismatch: values={tuple(values.shape)} mask={tuple(mask.shape)}")
    count = int(mask.sum().item())
    if count == 0:
        raise RuntimeError(f"{label} mask is empty")
    result = values[mask].mean()
    if not torch.isfinite(result):
        raise FloatingPointError(f"{label} mean is non-finite")
    return result


def global_grad_norm(grads: Iterable[torch.Tensor | None]) -> float:
    total = 0.0
    seen = 0
    for grad in grads:
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            raise FloatingPointError("non-finite calibration gradient")
        total += float(grad.detach().double().pow(2).sum().item())
        seen += grad.numel()
    if seen == 0 or total <= 0:
        raise RuntimeError("zero or missing calibration gradients")
    return math.sqrt(total)


def freeze_mu(
    main_norms: Sequence[float],
    rest_norms: Sequence[float],
    target_grad_ratio: float = 0.10,
    mu_min: float = 0.02,
    mu_max: float = 0.15,
) -> dict:
    if len(main_norms) != len(rest_norms) or not main_norms:
        raise ValueError("main/rest norm sequences must be non-empty and equal length")
    ratios = []
    for main, rest in zip(main_norms, rest_norms):
        if not (math.isfinite(main) and math.isfinite(rest) and main > 0 and rest > 0):
            raise ValueError(f"invalid gradient norms: main={main} rest={rest}")
        ratios.append(main / rest)
    median_ratio = float(np.median(np.asarray(ratios, dtype=np.float64)))
    raw = target_grad_ratio * median_ratio
    frozen = min(mu_max, max(mu_min, raw))
    return {
        "target_grad_ratio": target_grad_ratio,
        "mu_min": mu_min,
        "mu_max": mu_max,
        "median_main_over_rest": median_ratio,
        "mu_raw": raw,
        "mu": frozen,
        "clamped": frozen != raw,
        "ratios": ratios,
    }


def reset_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(step: int, base: float) -> float:
    return base * (1 - (step - 1) / LR_DENOMINATOR)


def ramp_coefficient(step: int) -> float:
    return min(1.0, step / RAMP_STEPS)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_calibration_receipt(receipt: Mapping, a05_sha: str, cache_manifest_sha: str, common_sha: str) -> float:
    provided_payload_sha = receipt.get("receipt_payload_sha256")
    unsigned_payload = dict(receipt)
    unsigned_payload.pop("receipt_payload_sha256", None)
    calculated_payload_sha = canonical_json_sha256(unsigned_payload)
    if provided_payload_sha != calculated_payload_sha:
        raise RuntimeError(
            f"calibration receipt payload SHA mismatch: provided={provided_payload_sha} calculated={calculated_payload_sha}"
        )
    required = {
        "formula_version": FORMULA_VERSION,
        "black_region_training_exclusion": False,
        "full_teacher_grid_partition": True,
        "a05_sha256": a05_sha,
        "teacher_cache_content_manifest_sha256": cache_manifest_sha,
        "tca_common_sha256": common_sha,
    }
    mismatches = {key: (receipt.get(key), expected) for key, expected in required.items() if receipt.get(key) != expected}
    if mismatches:
        raise RuntimeError(f"calibration receipt mismatch: {mismatches}")
    mu = float(receipt["mu"])
    if not (0.02 <= mu <= 0.15):
        raise RuntimeError(f"calibration mu outside frozen bounds: {mu}")
    return mu
