#!/usr/bin/env python3
"""Module change analysis: theta278 -> D1_alpha050 parameter delta norms."""
import torch, numpy as np, json
from pathlib import Path

THETA278 = "/home/jhy/rare_diag_20260915/atda_20260918/cpu_candidates/f_s2p100_s3p125.pt"
D1 = "/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt"
ALPHA = 0.5

def load_state(p):
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for k in ("model","student_model","state_dict","model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt

def categorize(name):
    n = name.lower()
    if "stages.0" in n: return "stage0"
    if "stages.1" in n: return "stage1"
    if "stages.2" in n: return "stage2"
    if "stages.3" in n: return "stage3"
    if "decode_head" in n or "classifier" in n: return "head"
    if "norm" in n or "layernorm" in n: return "norm/other"
    return "other"

print("Loading...")
s278 = load_state(THETA278)
sd1 = load_state(D1)

# Build delta for alpha=0.5
groups = {}
for k in s278:
    if k not in sd1: continue
    if "num_batches_tracked" in k: continue
    if not torch.is_floating_point(s278[k]): continue
    delta = (sd1[k].float() - s278[k].float()) * ALPHA
    g = categorize(k)
    if g not in groups: groups[g] = {"sq":0.0, "n":0, "max":0.0}
    d = delta.reshape(-1)
    groups[g]["sq"] += float(torch.dot(d, d))
    groups[g]["n"] += d.numel()
    groups[g]["max"] = max(groups[g]["max"], float(d.abs().max()))

total_sq = sum(g["sq"] for g in groups.values())
print(f"\n{'Module':<15} {'Delta norm':>12} {'% total':>10} {'numel':>12} {'rms':>10} {'max_abs':>10}")
results = {}
for g in ["stage0","stage1","stage2","stage3","head","norm/other","other"]:
    if g not in groups: continue
    d = groups[g]
    norm = d["sq"]**0.5
    pct = norm**2/total_sq*100
    rms = (d["sq"]/d["n"])**0.5
    print(f"{g:<15} {norm:>12.4f} {pct:>9.1f}% {d['n']:>12,} {rms:>10.6f} {d['max']:>10.6f}")
    results[g] = {"norm":norm, "pct":pct, "numel":d["n"], "rms":rms, "max_abs":d["max"]}

# Also head sub-breakdown
print(f"\n--- Head breakdown ---")
head_groups = {}
for k in s278:
    if k not in sd1 or "classifier" not in k.lower(): continue
    if "num_batches_tracked" in k: continue
    if not torch.is_floating_point(s278[k]): continue
    delta = (sd1[k].float() - s278[k].float()) * ALPHA
    d = delta.reshape(-1)
    head_groups[k] = {"norm": float(torch.dot(d,d))**0.5, "shape": list(s278[k].shape)}
for k, v in sorted(head_groups.items(), key=lambda x:-x[1]["norm"]):
    print(f"  {k:<50} {v['norm']:>10.4f}  shape={v['shape']}")

# Stage2 vs stage3 ratio
print(f"\n--- Stage comparison ---")
for g in ["stage0","stage1","stage2","stage3","head"]:
    if g in groups:
        print(f"  {g}: norm={groups[g]['sq']**0.5:.4f} ({groups[g]['sq']/total_sq*100:.1f}%)")
