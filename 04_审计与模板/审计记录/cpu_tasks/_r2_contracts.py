#!/usr/bin/env python3
"""Frozen contracts shared by the Round2 multiprobe campaign.

No platform submission helpers live in this module.  The only deployment
operator implemented here is the already-scored native-logit pipeline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

NUM_CLASSES = 8
CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren",
               "Vegetation", "Agricultural", "Vehicle"]
IGNORE_INDEX = 255
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
NATIVE_GRID = 256
OUTPUT_GRID = 1024
SEED = 20260922

ROOT = Path("/home/jhy/rare_diag_20260915")
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
TEST2 = Path("/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images")
OUT = ROOT / "round2_multiprobe_20260922"
ROUND2 = ROOT / "round2_20260921"
ATDA = ROOT / "atda_20260918"
CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
THETA261 = ROOT / "phase0_candidates_20260918/A_cleanSEM_M_k100_th209.pt"
ATDA8K = ATDA / "run_s2s3_base261/checkpoint_step8000.pt"
THETA278 = ATDA / "export/composed_atda_8k_a1p25.pt"
THETA278_FOLDED = ATDA / "finalize/composed_atda_8k_a1p25/composed_atda_8k_a1p25_folded.pt"

EXPECTED_FILE_SHA = {
    "theta261": "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786",
    "atda8k": "c3e87f6c51bd1fb32e6249783618d9bec7f1fc431a4a36d4ef3066d54e480eac",
    "theta278": "5f8a092b92f1e5c68e7173131c946f50b1663ade1fd43fd71a2ce2d394f001dc",
    "theta278_folded": "5be48fa5048b97c67aa7fc486fdb6619661d2ff2f08af9f92998ef4ac811d2ba",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(str(line).encode("utf-8") + b"\n")
    return h.hexdigest()


def load_state(path: Path):
    import torch
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            state = obj.get(key)
            if isinstance(state, dict) and state and all(isinstance(v, torch.Tensor) for v in state.values()):
                break
        else:
            state = obj if obj and all(isinstance(v, torch.Tensor) for v in obj.values()) else None
    else:
        state = None
    if state is None:
        raise RuntimeError(f"no tensor state dict in {path}")
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state, obj


def tensor_bytes(tensor) -> bytes:
    import torch
    value = tensor.detach().cpu().contiguous()
    if value.ndim == 0:
        value = value.reshape(1)
    if value.dtype == torch.bfloat16:
        return value.view(torch.int16).numpy().tobytes()
    return value.numpy().tobytes()


def canonical_state_sha(state) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        h.update(name.encode("utf-8") + b"\0")
        h.update(str(value.dtype).encode("ascii") + b"\0")
        h.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        h.update(tensor_bytes(value))
    return h.hexdigest()


def compare_states(a, b) -> dict:
    import torch
    if set(a) != set(b):
        return {"same_keys": False, "only_a": sorted(set(a) - set(b)), "only_b": sorted(set(b) - set(a))}
    changed, max_abs = [], 0.0
    for name in sorted(a):
        if not torch.equal(a[name], b[name]):
            changed.append(name)
            if a[name].is_floating_point() and a[name].shape == b[name].shape:
                max_abs = max(max_abs, float((a[name].float() - b[name].float()).abs().max()))
    return {"same_keys": True, "changed_tensors": len(changed),
            "changed_names": changed, "max_abs_diff": max_abs,
            "tensor_identical": not changed}


def d4_forward(tensor, index: int):
    rotation, mirror = VIEWS[index]
    if rotation:
        tensor = tensor.rot90(rotation, (-2, -1))
    if mirror:
        tensor = tensor.flip((-1,))
    return tensor


def d4_inverse(tensor, index: int):
    rotation, mirror = VIEWS[index]
    if mirror:
        tensor = tensor.flip((-1,))
    if rotation:
        tensor = tensor.rot90(-rotation, (-2, -1))
    return tensor.contiguous()


def deployed_logits(native_logits, size: int = OUTPUT_GRID):
    """Raw native logits -> Bstar -> exact bilinear logit upsample."""
    import torch
    import torch.nn.functional as F
    value = torch.as_tensor(native_logits, dtype=torch.float32)
    if tuple(value.shape) != (NUM_CLASSES, NATIVE_GRID, NATIVE_GRID):
        raise ValueError(f"expected [8,256,256], got {tuple(value.shape)}")
    value = value + torch.from_numpy(BSTAR)[:, None, None]
    return F.interpolate(value[None], size=(size, size), mode="bilinear",
                         align_corners=False)[0]


def exact_names(prefix: str, count: int, suffix: str) -> list[str]:
    return [f"{prefix}_{i}.{suffix}" for i in range(1, count + 1)]


def assert_no_submit_metadata(payload: dict) -> None:
    if payload.get("platform_submission_performed") is not False:
        raise RuntimeError("metadata must explicitly record no platform submission")
    if payload.get("test2_ground_truth_used") is not False:
        raise RuntimeError("metadata must explicitly record no test2 GT")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
