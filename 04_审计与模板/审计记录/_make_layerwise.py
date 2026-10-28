#!/usr/bin/env python3
"""Layer-wise merge grid. CPU only.

The global soup ceiling is not the parameter-merging ceiling: the best global
blend put EVERY one of the 644 tensors at the same alpha (0.20 for the s55b
direction), yet the B3 groups differ by two orders of magnitude in size and
plainly do different jobs:

    stage0   954K   2.0%    low-level texture
    stage1   1.9M   4.1%
    stage2   30.2M 63.9%    bulk of the encoder
    stage3   11.0M 23.3%    highest-level semantics
    head     3.2M   6.7%    class decision  <- where a bias operates

Parameterisation: base is soupC (s58/s64 equal), direction is
d = theta_s55b - theta_soupC, and each group gets its OWN alpha:

    theta_g = theta_C,g + alpha_g * d_g

so the measured global champion (0.698641) is simply alpha = 0.2 for every
group. L0 below reproduces exactly that as a pipeline control.

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

# group -> alpha applied to d = s55b - soupC
GRID = {
    "L0_all020_ctrl":   {"stage0": 0.2, "stage1": 0.2, "stage2": 0.2,
                         "stage3": 0.2, "head": 0.2},
    # head frozen at champion: does the class-decision layer want no movement?
    "L1_head000":       {"stage0": 0.2, "stage1": 0.2, "stage2": 0.2,
                         "stage3": 0.2, "head": 0.0},
    # deepen late, freeze head
    "L2_late030_head000": {"stage0": 0.1, "stage1": 0.1, "stage2": 0.2,
                           "stage3": 0.3, "head": 0.0},
    # deepen late, keep head at global
    "L3_late030_head020": {"stage0": 0.1, "stage1": 0.1, "stage2": 0.2,
                           "stage3": 0.3, "head": 0.2},
    # mirror: deepen early instead of late
    "L4_early030_head000": {"stage0": 0.3, "stage1": 0.3, "stage2": 0.2,
                            "stage3": 0.1, "head": 0.0},
    # head moves more than the rest
    "L5_head040":       {"stage0": 0.2, "stage1": 0.2, "stage2": 0.2,
                         "stage3": 0.2, "head": 0.4},
    # bulk encoder gets more, decision layers less
    "L6_mid030_head010": {"stage0": 0.1, "stage1": 0.1, "stage2": 0.3,
                          "stage3": 0.3, "head": 0.1},
}


def group_of(key: str) -> str:
    if key.startswith("segformer.stages."):
        return f"stage{key.split('.')[2]}"
    if key.startswith("segformer."):
        return "stage0"          # encoder.other is tiny; fold into stage0 bucket
    return "head"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = None
    for k in ("model", "model_state_dict", "state_dict"):
        c = ck.get(k)
        if isinstance(c, dict) and c and all(torch.is_tensor(v) for v in c.values()):
            st = c
            break
    if st is None:
        raise SystemExit(f"no model state in {path}")
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    a = load(S58)
    b = load(S64)
    c = load(S55B)
    keys = sorted(a)
    assert set(b) == set(keys) and set(c) == set(keys), "key mismatch"

    # soupC base and the direction d
    base, d = {}, {}
    for k in keys:
        ref = a[k]
        if ref.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8,
                         torch.bool):
            base[k] = ref.clone()
            continue
        base[k] = ((a[k].float() + b[k].float()) * 0.5).to(ref.dtype)
        d[k] = (c[k].float() - base[k].float())

    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "scope": "LAYERWISE_MERGE_GRID",
           "base": "soupC = (s58+s64)/2", "direction": "s55b - soupC",
           "members": {"s58": sha(Path(S58)), "s64": sha(Path(S64)),
                       "s55b": sha(Path(S55B))},
           "grid": {}}

    for name, alphas in GRID.items():
        out = {}
        for k in keys:
            if k not in d:
                out[k] = base[k]
                continue
            al = alphas[group_of(k)]
            out[k] = (base[k].float() + al * d[k]).to(base[k].dtype)
        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["grid"][name] = {"alphas": alphas, "path": str(dst),
                             "sha256": sha(dst)}
        print(f"{name:22s} {alphas} -> {dst.name}", flush=True)

    (outdir / "LAYERWISE_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'LAYERWISE_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
