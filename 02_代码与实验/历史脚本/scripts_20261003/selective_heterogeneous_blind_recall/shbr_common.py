#!/usr/bin/env python3
"""Shared fail-closed contracts for Selective Heterogeneous Blind Recall (SHBR)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

NUM_CLASSES = 8
IGNORE_INDEX = 255
CLASS_NAMES = (
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Vegetation",
    "Agricultural",
    "Vehicle",
)
BLIND_CLASSES = (3, 4, 6)
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
EXPECTED_A05_SHA = "023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134"
EXPECTED_ARMB_SHA = "f186e06ddd531724f647757587e5afec1415fb672410534de9658be60e3c58b9"
EXPECTED_ARMB_ZIP_SHA = "9b426a5bda7fcc8b6b96af9a7a6c2a8e86467dbb4d8bb61441717538f29f1885"
EXPECTED_CONTROL_CHECKPOINT_SHA = "5bc7a35303859dea5e0091bd35ac855ccdf4a74795db17897e079b11a7755548"
EXPECTED_CONTROL_ZIP_SHA = "1700504fbc9659bbb00c386a63c5624902fbf0f45ba1cce09bd8f2727be648b4"
CONF_NORMAL = 0.98
CONF_BLIND = 0.50
BETA = 0.25
SEED = 20260922
BASE_LR = 1e-5
HEAD_MULT = 0.05
WEIGHT_DECAY = 0.01
RAMP_STEPS = 500
LR_DENOMINATOR = 10000
MAX_STEPS = 2000
TRAIN_PREFIXES = ("segformer.stages.2.", "segformer.stages.3.")
HEAD_PREFIX = "decode_head."
FORMULA_VERSION = "shbr-v1-armb-plus-selective-heterogeneous-blind-recall-beta025"


@dataclass(frozen=True)
class ServerPaths:
    root: Path
    code: Path
    src: Path
    config: Path
    model_source: Path
    a05: Path
    armb: Path
    source_images: Path
    source_masks: Path
    source_train_split: Path
    test2_images: Path
    a05_target_cache: Path
    donor_checkpoint: Path
    donor_config: Path
    source_audit_split: Path
    control_checkpoint: Path
    control_zip: Path
    control_reference: Path
    armb_zip: Path

    @classmethod
    def from_config(cls, config: Mapping) -> "ServerPaths":
        root = Path(config["root"]).resolve()

        def path(key: str, default: str) -> Path:
            return Path(config.get(key, root / default)).resolve()

        return cls(
            root=root,
            code=path("code", "code"),
            src=path("src", "code/src"),
            config=path("b3_config", "code/config.yaml"),
            model_source=path("b3_model_source", "models/segformer-b3-ade512"),
            a05=path("a05_checkpoint", "checkpoints/A05_WEIGHT_alpha05.pt"),
            armb=path("armb_checkpoint", "runs/formal_armB/checkpoint_step2000.pt"),
            source_images=path("source_images", "datasets/训练集/train/images"),
            source_masks=path("source_masks", "datasets/训练集/train/masks"),
            source_train_split=path("source_train_split", "code/splits/train.txt"),
            test2_images=path("test2_images", "datasets/test_2/images"),
            a05_target_cache=path("a05_target_cache", "teacher_cache/scores"),
            donor_checkpoint=path(
                "donor_checkpoint",
                "m2f_minimaladapt/work_dirs/exp00r_m2f_swinl_10h_b8_30k/best_mIoU_iter_15000.pth",
            ),
            donor_config=path(
                "donor_config",
                "m2f_minimaladapt/work_dirs/exp00r_m2f_swinl_10h_b8_30k/config.py",
            ),
            source_audit_split=path("source_audit_split", "shbr_bundle/SOURCE_AUDIT_SPLIT.json"),
            control_checkpoint=path("control_checkpoint", "controls/RAS_KD_CONTROL_checkpoint_step2000.pt"),
            control_zip=path("control_zip", "controls/RAS_KD_CONTROL_test2.zip"),
            control_reference=path("control_reference", "controls/RAS_KD_CONTROL_RUN_SUMMARY.json"),
            armb_zip=path("armb_zip", "controls/REVERSE_ARMB_test2.zip"),
        )


@dataclass(frozen=True)
class GateRule:
    origin_class: int
    donor_class: int
    min_agreement: int
    margin_threshold: float
    min_component_area: int

    @classmethod
    def from_mapping(cls, value: Mapping) -> "GateRule":
        return cls(
            origin_class=int(value["origin_class"]),
            donor_class=int(value["donor_class"]),
            min_agreement=int(value["min_agreement"]),
            margin_threshold=float(value["margin_threshold"]),
            min_component_area=int(value["min_component_area"]),
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["origin_name"] = CLASS_NAMES[self.origin_class]
        result["donor_name"] = CLASS_NAMES[self.donor_class]
        return result


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8") + b"\n")
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state(path: Path) -> Mapping[str, torch.Tensor]:
    value = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        for key in ("model", "student_model", "model_state_dict", "state_dict"):
            state = value.get(key)
            if isinstance(state, dict) and state and all(isinstance(item, torch.Tensor) for item in state.values()):
                return state
        if value and all(isinstance(item, torch.Tensor) for item in value.values()):
            return value
    raise RuntimeError(f"no model state dictionary in {path}")


def exact_test2_names(count: int = 1300) -> list[str]:
    return [f"test2_{index}.png" for index in range(1, count + 1)]


def validate_exact_names(directory: Path, expected: Sequence[str], label: str) -> None:
    if not directory.is_dir():
        raise RuntimeError(f"missing {label} directory: {directory}")
    actual = sorted(item.name for item in directory.iterdir() if item.is_file())
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"{label} identity mismatch: missing={missing[:5]} extra={extra[:5]}")


def canonical_gt_to_train(mask: np.ndarray, encoding: str) -> np.ndarray:
    """Convert an explicitly declared GT encoding while preserving ignore=255."""
    if mask.ndim != 2:
        raise ValueError(f"GT must be 2-D, got {mask.shape}")
    values = set(int(item) for item in np.unique(mask))
    if encoding == "train_ids_0_7_ignore255":
        allowed = set(range(8)) | {IGNORE_INDEX}
        if not values <= allowed:
            raise ValueError(f"train-id GT contains unexpected labels: {sorted(values - allowed)}")
        return mask.astype(np.uint8, copy=True)
    if encoding == "official_ids_1_8_ignore255":
        allowed = set(range(1, 9)) | {IGNORE_INDEX}
        if not values <= allowed:
            raise ValueError(f"official-id GT contains unexpected labels: {sorted(values - allowed)}")
        result = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
        valid = mask != IGNORE_INDEX
        result[valid] = mask[valid].astype(np.int16).astype(np.uint8) - 1
        return result
    raise ValueError(f"unknown explicit GT encoding: {encoding}")


def d4_forward(tensor: torch.Tensor, view: int) -> torch.Tensor:
    rotation, flip = divmod(int(view), 2)
    if not 0 <= view < 8:
        raise ValueError(f"D4 view must be 0..7, got {view}")
    if rotation:
        tensor = torch.rot90(tensor, rotation, (-2, -1))
    if flip:
        tensor = tensor.flip((-1,))
    return tensor


def d4_inverse(tensor: torch.Tensor, view: int) -> torch.Tensor:
    rotation, flip = divmod(int(view), 2)
    if not 0 <= view < 8:
        raise ValueError(f"D4 view must be 0..7, got {view}")
    if flip:
        tensor = tensor.flip((-1,))
    if rotation:
        tensor = torch.rot90(tensor, -rotation, (-2, -1))
    return tensor.contiguous()


def teacher_selected(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probabilities.ndim != 4 or probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(f"teacher probabilities must be [N,8,H,W], got {tuple(probabilities.shape)}")
    confidence, predicted = probabilities.max(1)
    blind = torch.zeros_like(predicted, dtype=torch.bool)
    for class_id in BLIND_CLASSES:
        blind |= predicted == class_id
    selected = (confidence >= CONF_NORMAL) | (blind & (confidence >= CONF_BLIND))
    return confidence, predicted, selected


def per_pixel_kl(teacher_probabilities: torch.Tensor, student_log_probabilities: torch.Tensor) -> torch.Tensor:
    if teacher_probabilities.shape != student_log_probabilities.shape:
        raise ValueError("teacher/student KL shapes differ")
    value = (
        torch.xlogy(teacher_probabilities, teacher_probabilities)
        - teacher_probabilities * student_log_probabilities
    ).sum(1)
    if not torch.isfinite(value).all():
        raise FloatingPointError("non-finite per-pixel KL")
    return value


def shbr_losses(
    source_loss: torch.Tensor,
    a05_kl: torch.Tensor,
    selected: torch.Tensor,
    donor_kl: torch.Tensor,
    gate: torch.Tensor,
    ramp: float,
    role: str,
) -> dict[str, torch.Tensor]:
    if a05_kl.shape != selected.shape or donor_kl.shape != gate.shape or selected.shape != gate.shape:
        raise ValueError("SHBR loss mask/value shapes differ")
    if torch.any(selected & gate):
        raise RuntimeError("G must be disjoint from A05 selected M")
    count = selected.sum()
    if int(count.item()) == 0:
        raise RuntimeError("A05 selected M is empty")
    main = a05_kl[selected].sum() / count
    donor = donor_kl[gate].sum() / count if bool(gate.any()) else main.new_zeros(())
    coefficient = 0.0 if role == "control" else BETA
    if role not in ("control", "treat"):
        raise ValueError(f"unknown role: {role}")
    total = 0.1 * source_loss + float(ramp) * (main + coefficient * donor)
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite SHBR total loss")
    return {"total": total, "main": main, "donor": donor, "donor_coefficient": main.new_tensor(coefficient)}


def reset_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strong_photo(tensor: torch.Tensor) -> torch.Tensor:
    """Arm B photometric augmentation; call only after reset_seeds."""
    import torch.nn.functional as F

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


def learning_rate(step: int, base: float) -> float:
    return base * (1.0 - (step - 1) / LR_DENOMINATOR)


def ramp_coefficient(step: int) -> float:
    return min(1.0, step / RAMP_STEPS)


def is_trainable_parameter(name: str) -> bool:
    return name.startswith(TRAIN_PREFIXES) or name.startswith(HEAD_PREFIX)


def configure_trainable_scope(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    model.eval()
    model.get_submodule("segformer.stages.2").train()
    model.get_submodule("segformer.stages.3").train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_trainable_parameter(name))
    base = [parameter for name, parameter in model.named_parameters() if name.startswith(TRAIN_PREFIXES)]
    head = [parameter for name, parameter in model.named_parameters() if name.startswith(HEAD_PREFIX)]
    if not base or not head:
        raise RuntimeError(f"invalid trainable scope: base={len(base)} head={len(head)}")
    return base, head


def component_area_map(mask: np.ndarray) -> np.ndarray:
    """Return four-connected component area at every true pixel."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("component mask must be 2-D")
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
        areas = stats[:, cv2.CC_STAT_AREA]
        result = areas[labels]
        result[~mask] = 0
        return result.astype(np.int32, copy=False)
    except ImportError:
        pass

    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = np.zeros(mask.shape, dtype=np.int32)
    for row in range(height):
        for column in range(width):
            if not mask[row, column] or seen[row, column]:
                continue
            queue = deque([(row, column)])
            seen[row, column] = True
            component = []
            while queue:
                current_row, current_column = queue.popleft()
                component.append((current_row, current_column))
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and mask[next_row, next_column]
                        and not seen[next_row, next_column]
                    ):
                        seen[next_row, next_column] = True
                        queue.append((next_row, next_column))
            area = len(component)
            for current_row, current_column in component:
                result[current_row, current_column] = area
    return result


def apply_gate_rule(
    rule: GateRule,
    a05_prediction: np.ndarray,
    armb_prediction: np.ndarray,
    donor_prediction: np.ndarray,
    a05_selected: np.ndarray,
    donor_margin: np.ndarray,
    donor_agreement: np.ndarray,
) -> np.ndarray:
    arrays = (a05_prediction, armb_prediction, donor_prediction, a05_selected, donor_margin, donor_agreement)
    if any(np.asarray(item).shape != np.asarray(arrays[0]).shape for item in arrays[1:]):
        raise ValueError("gate arrays have different shapes")
    base = (
        (~a05_selected.astype(bool))
        & (armb_prediction == rule.origin_class)
        & (donor_prediction == rule.donor_class)
        & (a05_prediction != rule.donor_class)
        & (donor_agreement >= rule.min_agreement)
        & (donor_margin >= rule.margin_threshold)
    )
    if not base.any():
        return base
    return base & (component_area_map(base) >= rule.min_component_area)


def apply_gate_rules(rules: Sequence[GateRule], **arrays: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    if not rules:
        raise ValueError("at least one frozen gate rule is required")
    masks = [apply_gate_rule(rule, **arrays) for rule in rules]
    combined = np.logical_or.reduce(masks)
    if np.any(combined & np.asarray(arrays["a05_selected"], dtype=bool)):
        raise RuntimeError("combined G overlaps M")
    return combined, masks


def confusion_matrix(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if gt.shape != prediction.shape:
        raise ValueError("GT and prediction shapes differ")
    valid = gt != IGNORE_INDEX
    encoded = gt[valid].astype(np.int64) * NUM_CLASSES + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_confusion(confusion: np.ndarray) -> tuple[np.ndarray, float]:
    confusion = np.asarray(confusion, dtype=np.float64)
    denominator = confusion.sum(0) + confusion.sum(1) - np.diag(confusion)
    iou = np.divide(np.diag(confusion), denominator, out=np.full(NUM_CLASSES, np.nan), where=denominator > 0)
    return iou, float(np.nanmean(iou))


def replacement_metrics(
    records: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict:
    """Evaluate frozen donor replacement. Records are (gt, base, donor, gate)."""
    base_confusions = []
    candidate_confusions = []
    gains = losses = selected = donor_correct = both_wrong = 0
    support_images = 0
    for gt, base, donor, gate in records:
        candidate = np.asarray(base).copy()
        candidate[gate] = donor[gate]
        base_confusions.append(confusion_matrix(gt, base))
        candidate_confusions.append(confusion_matrix(gt, candidate))
        valid_gate = gate & (gt != IGNORE_INDEX)
        count = int(valid_gate.sum())
        if count:
            support_images += 1
            selected += count
            base_ok = base == gt
            donor_ok = donor == gt
            gains += int((valid_gate & donor_ok & ~base_ok).sum())
            losses += int((valid_gate & base_ok & ~donor_ok).sum())
            donor_correct += int((valid_gate & donor_ok).sum())
            both_wrong += int((valid_gate & ~base_ok & ~donor_ok).sum())
    base_confusion = np.sum(base_confusions, axis=0)
    candidate_confusion = np.sum(candidate_confusions, axis=0)
    base_iou, base_miou = iou_from_confusion(base_confusion)
    candidate_iou, candidate_miou = iou_from_confusion(candidate_confusion)
    return {
        "images": len(records),
        "support_images": support_images,
        "selected_pixels": selected,
        "gain_pixels": gains,
        "damage_pixels": losses,
        "both_wrong_pixels": both_wrong,
        "selected_pixel_donor_accuracy": donor_correct / selected if selected else 0.0,
        "correction_damage_ratio": gains / max(losses, 1),
        "base_miou": base_miou,
        "candidate_miou": candidate_miou,
        "delta_miou": candidate_miou - base_miou,
        "base_iou": base_iou.tolist(),
        "candidate_iou": candidate_iou.tolist(),
        "per_class_delta_iou": (candidate_iou - base_iou).tolist(),
        "base_confusion": base_confusion.tolist(),
        "candidate_confusion": candidate_confusion.tolist(),
    }


def bootstrap_delta_miou(
    records: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    repetitions: int = 2000,
    seed: int = SEED,
) -> dict:
    if not records:
        raise ValueError("bootstrap needs records")
    base = np.stack([confusion_matrix(gt, prediction) for gt, prediction, _, _ in records])
    candidate = []
    for gt, prediction, donor, gate in records:
        replaced = prediction.copy()
        replaced[gate] = donor[gate]
        candidate.append(confusion_matrix(gt, replaced))
    candidate_array = np.stack(candidate)
    generator = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    image_count = len(records)
    for index in range(repetitions):
        sampled = generator.integers(0, image_count, image_count)
        _, base_miou = iou_from_confusion(base[sampled].sum(0))
        _, candidate_miou = iou_from_confusion(candidate_array[sampled].sum(0))
        values[index] = candidate_miou - base_miou
    return {
        "repetitions": repetitions,
        "seed": seed,
        "mean": float(values.mean()),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def global_grad_norm(grads: Iterable[torch.Tensor | None]) -> float:
    total = 0.0
    count = 0
    for gradient in grads:
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("non-finite gradient")
        total += float(gradient.detach().double().pow(2).sum().item())
        count += gradient.numel()
    if count == 0 or not math.isfinite(total) or total <= 0:
        raise RuntimeError("missing or zero gradient")
    return math.sqrt(total)


def normalize_gpu_uuid(value: object) -> str:
    text = str(value)
    return text if text.startswith("GPU-") else f"GPU-{text}"


def query_gpus() -> list[dict]:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
        text=True,
    ).splitlines()
    busy_uuids = {row.split(",", 1)[0].strip() for row in processes if row.strip()}
    result = []
    for row in rows:
        fields = [part.strip() for part in row.split(",")]
        result.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "pci_bus_id": fields[2],
                "name": fields[3],
                "memory_used_mib": int(fields[4]),
                "memory_total_mib": int(fields[5]),
                "has_compute_process": fields[1] in busy_uuids,
            }
        )
    return result


def choose_idle_gpu(allowed: Sequence[int]) -> dict:
    choices = [row for row in query_gpus() if row["index"] in set(int(item) for item in allowed)]
    choices.sort(key=lambda row: row["index"])
    for row in choices:
        if not row["has_compute_process"]:
            return row
    raise RuntimeError(f"STOP_GPU_BUSY: no idle authorized GPU among {list(allowed)}; inventory={choices}")


def verify_visible_gpu(expected_physical: int, expected_uuid: str) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(expected_physical):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match frozen physical GPU")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")
    actual = normalize_gpu_uuid(torch.cuda.get_device_properties(0).uuid)
    if actual != expected_uuid:
        raise RuntimeError(f"GPU UUID mismatch: expected {expected_uuid}, got {actual}")
    return {"physical_index": expected_physical, "uuid": actual, "logical_device": "cuda:0"}


def content_manifest(directory: Path, pattern: str = "*.npz") -> dict:
    files = sorted(directory.glob(pattern), key=lambda item: item.name)
    digest = hashlib.sha256()
    for path in files:
        file_sha = sha256_file(path)
        digest.update(f"{path.name}\t{path.stat().st_size}\t{file_sha}\n".encode("utf-8"))
    return {"count": len(files), "sha256": digest.hexdigest()}
