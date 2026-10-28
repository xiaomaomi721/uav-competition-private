#!/usr/bin/env python3
"""Layer-wise round 3: exploit the measured additivity. CPU only.

Round 2 (all @b*, baseline 0.699835) found a clean additive structure:

    early 0.3 -> 0.4  (deepest held at 0.1, head 0)     0.700054   +0.0219
    deepest 0.1 -> 0.0 (early held at 0.3, head 0)      0.700368   +0.0533
    BOTH                                                0.700587   +0.0752

    0.0219 + 0.0533 = 0.0752  -- exactly additive

so "how hard the shallow stages move" and "whether the deepest stage moves at
all" are INDEPENDENT knobs, not the geometric depth profile k^depth that the
joint grid assumed. Also, mid-stage movement (204, s2 0.2->0.3) was neutral
(-0.0002), so stage2 is best left where it is.

This round pushes the two independent knobs:
  * early stages upward, 0.4 -> 0.6 -> 0.8, with the deepest already at 0
  * stage1 vs stage0 decoupled, to see whether the two shallow stages agree
  * the head stays at 0 throughout -- every nonzero head alpha measured today
    was negative (0.2 -> -0.0509, 0.4 -> -0.0509 vs its own baseline)

CPU only. Writes only under --outdir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

B = "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1"
S58 = f"{B}/stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt"
S64 = f"{B}/stage64_runs/stage64_w1_exp64_b3_cwfa_wd0005_seed42/best_miou.pt"
S55B = f"{B}/stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt"

ROUND2 = {"197 s0/s1=0.3 deep0.1": 0.699835, "198 s0/s1=0.4": 0.700054,
          "199 s0/s1=0.5": 0.700229, "200 s0/s1=0.6": 0.700361,
          "201 deep=0.0": 0.700368, "202 both": 0.700587,
          "203 monotone": 0.699963, "204 mid0.3": 0.699819}

GRID = {
    "N0_ctrl_202":      {"stage0": 0.4, "stage1": 0.4, "stage2": 0.2, "stage3": 0.0, "head": 0.0},
    "N1_e060_d000":     {"stage0": 0.6, "stage1": 0.6, "stage2": 0.2, "stage3": 0.0, "head": 0.0},
    "N2_e080_d000":     {"stage0": 0.8, "stage1": 0.8, "stage2": 0.2, "stage3": 0.0, "head": 0.0},
    "N3_e100_d000":     {"stage0": 1.0, "stage1": 1.0, "stage2": 0.2, "stage3": 0.0, "head": 0.0},
    # decouple the two shallow stages: s0 harder than s1
    "N4_s0_0.6_s1_0.4": {"stage0": 0.6, "stage1": 0.4, "stage2": 0.2, "stage3": 0.0, "head": 0.0},
    "N5_s0_0.8_s1_0.4": {"stage0": 0.8, "stage1": 0.4, "stage2": 0.2, "stage3": 0.0, "head": 0.0},
    # and the reverse, to test the direction
    "N6_s0_0.4_s1_0.8": {"stage0": 0.4, "stage1": 0.8, "stage2": 0.2, "stage3": 0.0, "head": 0.0},
    # mid stage down instead of up
    "N7_e050_s2_0.1":   {"stage0": 0.5, "stage1": 0.5, "stage2": 0.1, "stage3": 0.0, "head": 0.0},
    # mid stage up, on top of the additive optimum
    "N8_e050_s2_0.3":   {"stage0": 0.5, "stage1": 0.5, "stage2": 0.3, "stage3": 0.0, "head": 0.0},
    # small negative alphas: extrapolate past "zero" on the deepest stage
    "N9_e050_dm010":    {"stage0": 0.5, "stage1": 0.5, "stage2": 0.2, "stage3": -0.1, "head": 0.0},
}


def group_of(key: str) -> str:
    if key.startswith("segformer.stages."):
        return f"stage{key.split('.')[2]}"
    if key.startswith("segformer."):
        return "stage0"
    return "head"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model", "model_state_dict", "state_dict"):
        c = ck.get(k)
        if isinstance(c, dict) and c and all(torch.is_tensor(v) for v in c.values()):
            return c
    raise SystemExit(f"no model state in {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    a, b, c = load(S58), load(S64), load(S55B)
    keys = sorted(a)
    base, d = {}, {}
    for k in keys:
        ref = a[k]
        if ref.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8,
                         torch.bool):
            base[k] = ref.clone()
            continue
        base[k] = ((a[k].float() + b[k].float()) * 0.5).to(ref.dtype)
        d[k] = c[k].float() - base[k].float()

    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "scope": "LAYERWISE_ROUND3", "round2": ROUND2, "grid": {}}
    for name, al in GRID.items():
        out = {}
        for key in keys:
            if key not in d:
                out[key] = base[key]
                continue
            out[key] = (base[key].float() + al[group_of(key)] * d[key]).to(base[key].dtype)
        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["grid"][name] = {"alphas": al, "path": str(dst), "sha256": sha(dst)}
        print(f"{name:20s} s0=%.2f s1=%.2f s2=%.2f s3=%.2f head=%.1f"
              % (al["stage0"], al["stage1"], al["stage2"], al["stage3"], al["head"]),
              flush=True)

    (outdir / "LAYERWISE3_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'LAYERWISE3_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
