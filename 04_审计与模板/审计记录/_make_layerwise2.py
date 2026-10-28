#!/usr/bin/env python3
"""Layer-wise merge round 2: push the measured depth gradient.

Round 1 (all @b*, baseline L0 = 0.698641) found a clean monotone structure:

    L1  head frozen, rest 0.2             0.698940  +0.0299
    L2  late 0.3, head 0                  0.697713  -0.0928
    L4  0.3/0.3/0.2/0.1/0.0               0.699835  +0.1194   <- champion
    L5  head 0.4                          0.698132  -0.0509
    L6  0.1/0.1/0.3/0.3/0.1               0.697735  -0.0906

So alpha wants to DECREASE with depth: shallow encoder moves more, the deepest
encoder stage moves less, the head does not move at all. A single global alpha
was averaging those opposing demands together.

Round 2 walks that gradient further and in two directions at once:
  * push the early stages harder (0.4, 0.5, 0.6)
  * pull the late stage to zero
  * one monotone-in-stage variant (0.4/0.3/0.2/0.1/0.0)

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

# round-1 measured, @b*
ROUND1 = {"L0": 0.698641, "L1": 0.698940, "L2": 0.697713,
          "L3": 0.697547, "L4": 0.699835, "L5": 0.698132, "L6": 0.697735}

GRID = {
    # control reproducing round-1 champion
    "M0_ctrl_L4":        {"stage0": 0.3, "stage1": 0.3, "stage2": 0.2,
                          "stage3": 0.1, "head": 0.0},
    # push the early stages while everything else is held
    "M1_early040":       {"stage0": 0.4, "stage1": 0.4, "stage2": 0.2,
                          "stage3": 0.1, "head": 0.0},
    "M2_early050":       {"stage0": 0.5, "stage1": 0.5, "stage2": 0.2,
                          "stage3": 0.1, "head": 0.0},
    "M3_early060":       {"stage0": 0.6, "stage1": 0.6, "stage2": 0.2,
                          "stage3": 0.1, "head": 0.0},
    # pull the deepest encoder stage to zero as well
    "M4_late000":        {"stage0": 0.3, "stage1": 0.3, "stage2": 0.2,
                          "stage3": 0.0, "head": 0.0},
    "M5_early040_late000": {"stage0": 0.4, "stage1": 0.4, "stage2": 0.2,
                            "stage3": 0.0, "head": 0.0},
    # strictly monotone in depth rather than a two-level step
    "M6_monotone":       {"stage0": 0.4, "stage1": 0.3, "stage2": 0.2,
                          "stage3": 0.1, "head": 0.0},
    # mid stage also lifted
    "M7_mid030":         {"stage0": 0.4, "stage1": 0.4, "stage2": 0.3,
                          "stage3": 0.1, "head": 0.0},
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
           "scope": "LAYERWISE_MERGE_ROUND2", "round1": ROUND1,
           "base": "soupC = (s58+s64)/2", "direction": "s55b - soupC",
           "grid": {}}
    for name, alphas in GRID.items():
        out = {}
        for k in keys:
            if k not in d:
                out[k] = base[k]
                continue
            out[k] = (base[k].float() + alphas[group_of(k)] * d[k]).to(base[k].dtype)
        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["grid"][name] = {"alphas": alphas, "path": str(dst), "sha256": sha(dst)}
        print(f"{name:24s} {alphas}", flush=True)

    (outdir / "LAYERWISE2_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'LAYERWISE2_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
