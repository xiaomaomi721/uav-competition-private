#!/usr/bin/env python3
"""Layer-wise optimisation of the s62 direction. CPU only.

Today's champion (0.701091) applied layer-wise alphas to the s55b direction:

    theta = soupC + sum_g alpha_g * d_g ,  d = s55b - soupC
    alpha = (0.5, 0.5, 0.2, -0.1, 0.0) for stages 0..3 + head

The s62 and Stage23B directions were only ever tested with a SINGLE global
alpha (0.20 and 0.33), and layer-wise was worth +0.12pp on s55b. So the same
treatment has never been tried on the other two live directions.

s62's direction is the interesting one: s62 has the WEAKEST response of any soup
member (R_C 0.5637pp, vs 1.49-1.50 for s58/s64) yet at alpha=0.10 it contributed
+0.1453pp -- second only to s55b. That is a direction whose value is invisible in
the member's own score.

This grid applies the layer-wise profile that won on s55b to d = s62 - soupC,
plus a couple of variants so the two directions are not forced to share a shape.

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
S62 = f"{B}/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/best_miou.pt"
S55B = f"{B}/stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt"

# (name, direction member, alphas)  alphas for stage0,1,2,3,head
GRID = {
    # s62 direction, several depth profiles (global-0.20 was 0.698370)
    "Q0_s62_ctrl_g020": ("s62", (0.20, 0.20, 0.20, 0.20, 0.0)),
    "Q1_s62_lw_50_50_20_m10": ("s62", (0.50, 0.50, 0.20, -0.10, 0.0)),
    "Q2_s62_lw_80_80_20_00": ("s62", (0.80, 0.80, 0.20, 0.00, 0.0)),
    "Q3_s62_lw_80_80_20_m10": ("s62", (0.80, 0.80, 0.20, -0.10, 0.0)),
    "Q4_s62_monotone": ("s62", (0.60, 0.40, 0.20, -0.10, 0.0)),
    # Stage23B direction with the same treatment (global-0.33 was 0.698103)
    "Q5_23b_ctrl_g033": ("stage23b", (0.33, 0.33, 0.33, 0.33, 0.0)),
    "Q6_23b_lw_50_50_20_00": ("stage23b", (0.50, 0.50, 0.20, 0.00, 0.0)),
    "Q7_23b_lw_60_60_20_m10": ("stage23b", (0.60, 0.60, 0.20, -0.10, 0.0)),
}
MEMBERS = {"s62": S62,
           "stage23b": f"{B}/stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/best_miou.pt"}


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

    a, b = load(S58), load(S64)
    keys = sorted(a)
    base = {}
    for k in keys:
        ref = a[k]
        if ref.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8,
                         torch.bool):
            base[k] = ref.clone()
        else:
            base[k] = ((a[k].float() + b[k].float()) * 0.5).to(ref.dtype)

    dirs = {m: load(p) for m, p in MEMBERS.items()}
    d = {m: {k: dirs[m][k].float() - base[k].float()
             for k in keys if k in base and base[k].dtype not in
             (torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool)}
         for m in MEMBERS}

    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "scope": "S62_23B_LAYERWISE", "grid": {}}
    for name, (mem, al) in GRID.items():
        als = dict(zip(("stage0", "stage1", "stage2", "stage3", "head"), al))
        out = {}
        for k in keys:
            if k not in d[mem]:
                out[k] = base[k]
                continue
            out[k] = (base[k].float() + als[group_of(k)] * d[mem][k]).to(base[k].dtype)
        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["grid"][name] = {"direction": mem, "alphas": als,
                             "path": str(dst), "sha256": sha(dst)}
        print(f"{name:26s} d={mem:9s} s0..s3={al[:4]} head={al[4]}", flush=True)

    (outdir / "S62_23B_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'S62_23B_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
