#!/usr/bin/env python3
"""Layer-wise round 4: joint early-strength x negative-deepest. CPU only.

Round 3 measured a THIRD independent knob, and one that runs against intuition:

    207  0.8/0.8/0.2/ 0.0   0.701042      deepest at zero
    209  0.5/0.5/0.2/-0.1   0.701091      deepest NEGATIVE

209 wins despite moving its shallow stages much LESS than 207 (0.5 vs 0.8), so
the gain from pushing the deepest encoder stage to a negative alpha more than
covers the shallow-stage loss. The deepest stage wants to move AWAY from s55b.

Round 3's other readings so far, all @b*:
    s0/s1 0.3  ->  0.700368
    s0/s1 0.4  ->  0.700587     +0.0219
    s0/s1 0.6  ->  0.700870     +0.0283
    s0/s1 0.8  ->  0.701042     +0.0172   (decelerating)
    s0/s1 decoupled 0.6/0.4 -> 0.700774    (worse -- the two shallow stages agree)
    head != 0 always hurt; stage2 (mid) neutral at +0.1

So this round searches the (shallow, deepest) plane directly, with head pinned
at 0 and mid held at the neutral 0.2.

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

# name -> (shallow alpha for stage0&1, deepest alpha for stage3)
GRID = {
    "P0_ctrl_207":   (0.80, 0.0),     # control: should reproduce 0.701042
    "P1_080_m010":   (0.80, -0.1),    # 207 + the N9 win
    "P2_080_m020":   (0.80, -0.2),
    "P3_100_m010":   (1.00, -0.1),
    "P4_060_m010":   (0.60, -0.1),
    "P5_060_m020":   (0.60, -0.2),
    "P6_050_m020":   (0.50, -0.2),
    "P7_080_m030":   (0.80, -0.3),
    "P8_100_m020":   (1.00, -0.2),
    "P9_040_m010":   (0.40, -0.1),
}
MID = 0.2


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
           "scope": "LAYERWISE_ROUND4", "mid": MID, "head": 0.0, "grid": {}}
    for name, (sh, deep) in GRID.items():
        al = {"stage0": sh, "stage1": sh, "stage2": MID, "stage3": deep, "head": 0.0}
        out = {}
        for key in keys:
            if key not in d:
                out[key] = base[key]
                continue
            out[key] = (base[key].float() + al[group_of(key)] * d[key]).to(base[key].dtype)
        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["grid"][name] = {"shallow": sh, "deepest": deep, "alphas": al,
                             "path": str(dst), "sha256": sha(dst)}
        print(f"{name:16s} shallow=%.2f  mid=%.2f  deepest=%+.2f  head=0"
              % (sh, MID, deep), flush=True)

    (outdir / "LAYERWISE4_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'LAYERWISE4_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
