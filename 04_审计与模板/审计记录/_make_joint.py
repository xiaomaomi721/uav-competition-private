#!/usr/bin/env python3
"""Joint search over direction strength AND depth profile. CPU only.

Two independent effects were measured today and have never been searched
together:

  (A) direction strength -- along d = s55b - soupC, a single global alpha peaks
      at 0.20 (0.696865 -> 0.698641), and the response is a narrow peak:
      0.30 costs 0.075pp, 0.40 costs 0.159pp.

  (B) depth profile -- with the global alpha fixed at 0.20, giving each B3 group
      its own alpha moved the score further: shallow stages want MORE, the
      deepest stage wants LESS, and the head wants NONE (0.699835).

The two knobs act on the same direction d, so the natural joint parameterisation
is a strength a times a geometric depth profile k:

    stage0 : a
    stage1 : a * k
    stage2 : a * k^2
    stage3 : a * k^3
    head   : 0            (round 1 showed every nonzero head alpha hurt)

a = 0.2 with k = 1 recovers the global champion; a = 0.3, k = 0.693 is close to
the current best (0.3/0.3/0.2/0.1/0.0 for stages 0..3).

Round 1 evidence, @b*, baseline 0.698641:
    head frozen                     0.698940
    0.3/0.3/0.2/0.1/0.0             0.699835   <- current champion
    0.3/0.3/0.2/0.3/0.2            0.697547
    0.2/0.2/0.2/0.2/0.4            0.698132

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

# name -> (a, k);  alphas by depth: a, a*k, a*k^2, a*k^3, head 0
GRID = {
    "J0_ctrl_a020_k100": (0.20, 1.000),   # global champion, should be 0.698641
    "J1_ctrl_a030_k069": (0.30, 0.693),   # ~ current champion
    "J2_a030_k050":      (0.30, 0.500),
    "J3_a030_k030":      (0.30, 0.300),
    "J4_a040_k050":      (0.40, 0.500),
    "J5_a040_k030":      (0.40, 0.300),
    "J6_a050_k040":      (0.50, 0.400),
    "J7_a050_k020":      (0.50, 0.200),
    "J8_a025_k040":      (0.25, 0.400),
    "J9_a035_k020":      (0.35, 0.200),
}


def group_of(key: str) -> str:
    if key.startswith("segformer.stages."):
        return f"stage{key.split('.')[2]}"
    if key.startswith("segformer."):
        return "stage0"
    return "head"


def alphas_for(a: float, k: float) -> dict:
    return {"stage0": a, "stage1": a * k, "stage2": a * k * k,
            "stage3": a * k ** 3, "head": 0.0}


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
           "scope": "JOINT_STRENGTH_X_DEPTH", "base": "soupC",
           "direction": "s55b - soupC", "grid": {}}
    for name, (aa, kk) in GRID.items():
        al = alphas_for(aa, kk)
        out = {}
        for key in keys:
            if key not in d:
                out[key] = base[key]
                continue
            out[key] = (base[key].float() + al[group_of(key)] * d[key]).to(base[key].dtype)
        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["grid"][name] = {"a": aa, "k": kk, "alphas": al,
                             "path": str(dst), "sha256": sha(dst)}
        print(f"{name:22s} a=%.2f k=%.3f  s0..s3 = %.3f/%.3f/%.3f/%.3f head=0"
              % (aa, kk, al["stage0"], al["stage1"], al["stage2"], al["stage3"]),
              flush=True)

    (outdir / "JOINT_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'JOINT_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
