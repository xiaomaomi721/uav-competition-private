#!/usr/bin/env python3
"""Direction audit: theta261 -> theta278 -> D1 parameter vectors."""
import hashlib, json, math
from pathlib import Path
import torch

THETA261 = Path("/home/jhy/rare_diag_20260915/phase0_candidates_20260918/A_cleanSEM_M_k100_th209.pt")
THETA278 = Path("/home/jhy/rare_diag_20260915/atda_20260918/cpu_candidates/f_s2p100_s3p125.pt")
D1 = Path("/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt")
OUT = Path("/home/jhy/rare_diag_20260915/direction_audit_20260924")
OUT.mkdir(exist_ok=True)

def load_state(path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model", "student_model", "state_dict", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def stage_of(name):
    n = name.lower()
    if "stages.2" in n or "block.2" in n: return "stage2"
    if "stages.3" in n or "block.3" in n: return "stage3"
    if "decode_head" in n or "classifier" in n: return "head"
    if "stages.0" in n or "stages.1" in n or "block.0" in n or "block.1" in n: return "early"
    return "other"

def vec(states_a, states_b, keys_filter=None):
    """Compute flattened diff a-b, optionally filtered by stage."""
    parts = []
    for k in states_a:
        if k not in states_b: continue
        if not torch.is_floating_point(states_a[k]): continue
        if keys_filter and stage_of(k) not in keys_filter: continue
        parts.append((states_a[k].float() - states_b[k].float()).reshape(-1))
    return torch.cat(parts)

def stats(name, v):
    return {
        "norm": float(v.norm()),
        "rms": float(v.norm() / math.sqrt(v.numel())),
        "numel": v.numel(),
    }

print("Loading weights...")
s261 = load_state(THETA261)
s278 = load_state(THETA278)
sD1 = load_state(D1)
print(f"  theta261 keys: {len(s261)}")
print(f"  theta278 keys: {len(s278)}")
print(f"  D1 keys: {len(sD1)}")

# Verify same keys
common = set(s261.keys()) & set(s278.keys()) & set(sD1.keys())
print(f"  common keys: {len(common)}")

results = {"files": {
    "theta261": str(THETA261), "theta278": str(THETA278), "D1": str(D1)
}}

# Compute deltas
stages = ["all", "stage2", "stage3", "head", "early"]
results["deltas"] = {}

for s in stages:
    filt = None if s == "all" else {s}
    d_atda = vec(s278, s261, filt)
    d_d1 = vec(sD1, s278, filt)
    
    cos = float(torch.dot(d_atda, d_d1) / (d_atda.norm() * d_d1.norm())) if d_atda.norm() > 0 and d_d1.norm() > 0 else None
    atda_stats = stats("atda", d_atda)
    d1_stats = stats("d1", d_d1)
    
    # Parallel/orthogonal decomposition of d_d1 along d_atda
    if cos is not None and d_atda.norm() > 0:
        alpha = float(torch.dot(d_d1, d_atda) / (d_atda.norm()**2))
        parallel_energy = float((alpha * d_atda).norm()**2)
        total_energy = float(d_d1.norm()**2)
        results["deltas"][s] = {
            "cos_d1_vs_atda": cos,
            "atda": atda_stats,
            "d1": d1_stats,
            "d1_parallel_to_atda_fraction": parallel_energy / total_energy if total_energy > 0 else None,
            "d1_orthogonal_fraction": 1 - parallel_energy / total_energy if total_energy > 0 else None,
            "equivalent_alpha_along_atda": alpha,
        }
    else:
        results["deltas"][s] = {"atda": atda_stats, "d1": d1_stats}
    
    print(f"\n{s}:")
    print(f"  ||ΔATDA|| = {atda_stats['norm']:.4f} (rms={atda_stats['rms']:.6f})")
    print(f"  ||ΔD1||   = {d1_stats['norm']:.4f} (rms={d1_stats['rms']:.6f})")
    if cos is not None:
        print(f"  cos(ΔATDA, ΔD1) = {cos:.6f}")
        print(f"  D1 parallel energy fraction: {results['deltas'][s].get('d1_parallel_to_atda_fraction', '?'):.4f}")

# Stage ratio
s2_norm = results["deltas"]["stage2"]["atda"]["norm"]
s3_norm = results["deltas"]["stage3"]["atda"]["norm"]
results["atda_stage_ratio"] = float(s2_norm / s3_norm) if s3_norm > 0 else None
print(f"\nATDA stage2/stage3 norm ratio: {results['atda_stage_ratio']:.4f}")

# Also compare D1's stage2/stage3 movement ratio
d1_s2 = results["deltas"]["stage2"]["d1"]["norm"]
d1_s3 = results["deltas"]["stage3"]["d1"]["norm"]
results["d1_stage_ratio"] = float(d1_s2 / d1_s3) if d1_s3 > 0 else None
print(f"D1 stage2/stage3 norm ratio: {results['d1_stage_ratio']:.4f}")

# Save
(OUT / "direction_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nSaved to {OUT / 'direction_report.json'}")
