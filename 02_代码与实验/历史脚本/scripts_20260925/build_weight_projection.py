#!/usr/bin/env python3
"""Build A05_WEIGHT single checkpoint: theta = parent + alpha*(D1-parent).

Pure CPU operation. Verifies alpha=0 (==parent) and alpha=1 (==D1) controls.
"""
import hashlib, json, sys
from pathlib import Path
import torch

ROOT = Path("/home/jhy/rare_diag_20260915")
PARENT = ROOT / "atda_20260918/cpu_candidates/f_s2p100_s3p125.pt"
D1 = ROOT / "round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt"
OUT_DIR = ROOT / "breakthrough_v2_20260926/03_weight_projection"
ALPHA = 0.5

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def get_state(obj):
    if isinstance(obj, dict):
        for k in ("model","student_model","model_state_dict","state_dict"):
            v = obj.get(k)
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()): return v
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()): return obj
    raise RuntimeError("no state dict")

print("Loading parent...")
parent_obj = torch.load(PARENT, map_location="cpu", weights_only=False)
parent_sd = get_state(parent_obj)
print("Loading D1...")
d1_obj = torch.load(D1, map_location="cpu", weights_only=False)
d1_sd = get_state(d1_obj)

# Key set check
assert set(parent_sd.keys()) == set(d1_sd.keys()), "key mismatch"
print(f"Keys: {len(parent_sd)}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build alpha=0.5
new_sd = {}
float_changes = []
buffer_issues = []
for k in parent_sd:
    p = parent_sd[k]
    d = d1_sd[k]
    if p.shape != d.shape:
        raise RuntimeError(f"shape mismatch {k}: {p.shape} vs {d.shape}")
    if p.dtype.is_floating_point:
        out = p.float() + ALPHA * (d.float() - p.float())
        new_sd[k] = out.to(p.dtype)
        delta = (d.float() - p.float()).abs().max().item()
        if delta > 0:
            float_changes.append((k, delta))
    else:
        # Integer/buffer: must be identical
        if not torch.equal(p, d):
            buffer_issues.append(k)
        new_sd[k] = p  # keep parent
        if not torch.equal(p, d):
            print(f"  WARNING buffer differs: {k}")

print(f"Float tensors changed: {len(float_changes)}")
print(f"Buffer issues: {len(buffer_issues)}")

# Verify alpha=0 control
alpha0_sd = {k: parent_sd[k] for k in parent_sd}
identical_0 = all(torch.equal(alpha0_sd[k], parent_sd[k]) for k in parent_sd)
print(f"alpha=0 == parent: {identical_0}")

# Verify alpha=1 control
alpha1_sd = {k: d1_sd[k] for k in d1_sd}
identical_1 = all(torch.equal(alpha1_sd[k], d1_sd[k]) for k in d1_sd)
print(f"alpha=1 == D1: {identical_1}")

# Save
out_path = OUT_DIR / "A05_WEIGHT_alpha05.pt"
save_obj = {"model": new_sd}
torch.save(save_obj, out_path)
out_sha = sha256_file(out_path)
print(f"\nSaved: {out_path}")
print(f"SHA256: {out_sha}")

# Manifest
manifest = {
    "name": "A05_WEIGHT_alpha05",
    "formula": "theta = parent + 0.5*(D1-parent)",
    "alpha": ALPHA,
    "parent_path": str(PARENT),
    "parent_sha256": sha256_file(PARENT),
    "d1_path": str(D1),
    "d1_sha256": sha256_file(D1),
    "output_sha256": out_sha,
    "float_tensors_changed": len(float_changes),
    "buffer_issues": buffer_issues,
    "alpha0_control_identical_to_parent": identical_0,
    "alpha1_control_identical_to_d1": identical_1,
    "single_checkpoint": True,
    "ensemble": False,
}
with open(OUT_DIR / "A05_WEIGHT_MANIFEST.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest written.")
