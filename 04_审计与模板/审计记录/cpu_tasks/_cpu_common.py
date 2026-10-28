"""Shared CPU-only loaders for the ATDA CPU task queue. No CUDA, no forward."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ATDA = Path("/home/jhy/rare_diag_20260915/atda_20260918")
RARE = Path("/home/jhy/rare_diag_20260915")
THETA261 = RARE / "phase0_candidates_20260918/A_cleanSEM_M_k100_th209.pt"
THETA261_SHA = "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786"
THETA261_SCORE = 0.702836
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)

# official class ids 1..8 -> name
CLASS_NAMES = ["Background", "Building", "Road", "Water",
               "Barren", "Vegetation", "Agricultural", "Vehicle"]

TRAINABLE_PREFIXES = ("segformer.stages.2.", "segformer.stages.3.")


def is_trainable(name: str) -> bool:
    return any(name.startswith(p) for p in TRAINABLE_PREFIXES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_state(path: Path):
    """Return the pure tensor state dict of a checkpoint / model file."""
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            c = ck.get(key)
            if isinstance(c, dict) and c and all(isinstance(v, torch.Tensor) for v in c.values()):
                return c, ck
        if ck and all(isinstance(v, torch.Tensor) for v in ck.values()):
            return ck, {}
    raise SystemExit(f"no tensor state dict in {path}")


def load_student(path: Path):
    """Load the 'model' (student) state from an ATDA gate checkpoint."""
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return ck["model"], ck


def load_ema(path: Path):
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return ck["ema"], ck


def tensor_sha(state) -> str:
    """Deterministic SHA over a state dict's tensors, in sorted key order."""
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(t.shape), str(t.dtype))).encode())
        h.update(t.numpy().tobytes() if t.dtype in (
            torch.float32, torch.float16, torch.float64, torch.bfloat16,
            torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool
        ) else t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path: Path, state, meta: dict):
    import torch
    torch.save({"model": state, "cpu_candidate_meta": meta}, path)
    return {"path": str(path), "sha256": sha256_file(path),
            "state_sha256": tensor_sha(state)}


def rms(t) -> float:
    import torch
    return float(torch.sqrt(torch.mean(t.detach().float() ** 2)).item())


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
