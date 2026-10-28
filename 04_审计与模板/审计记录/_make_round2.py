#!/usr/bin/env python3
"""Coefficient search round 2: push the three live parameter rays further.

Round 1 (all at the champion operator b* = farmland 2.1215 / water 0.4,
anchor = soupC 0.696865):

    ->Stage23b   a=0.10  0.697276  +0.0411
                 a=0.20  0.697493  +0.0628
                 a=0.33  0.698103  +0.1238   still rising, and accelerating
    ->s55b       a=0.10  0.698480  +0.1615
                 a=0.20  0.698641  +0.1776   current champion
    ->s62        a=0.10  0.698318  +0.1453   only one point so far

Every ray is still above its last measured point, so none of them is a settled
vertex. This round extends each one at the same 0.1-0.2 granularity, and adds a
two-member extension for the leader: s55b and Stage23b together, since both are
high-response bodies and the centre (s58/s64) is held fixed at 50:50.

Directions are parameter-space rays from the soupC centre, so the mixtures stay
in the same basin by construction -- nothing here is a new checkpoint.

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
BODIES = {
    "s58": f"{B}/stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt",
    "s64": f"{B}/stage64_runs/stage64_w1_exp64_b3_cwfa_wd0005_seed42/best_miou.pt",
    "s55b": f"{B}/stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt",
    "stage23b": f"{B}/stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/best_miou.pt",
    "s62": f"{B}/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/best_miou.pt",
}

# round-1 measured, @b*, anchor 0.696865
ROUND1 = {
    "->23B":  [(0.10, 0.697276), (0.20, 0.697493), (0.33, 0.698103)],
    "->s55b": [(0.10, 0.698480), (0.20, 0.698641)],
    "->s62":  [(0.10, 0.698318)],
}

# name -> weights
GRID = {
    # ->s55b, continue past the current champion
    "r2_55b_a30":   {"s58": 0.35, "s64": 0.35, "s55b": 0.30},
    "r2_55b_a40":   {"s58": 0.30, "s64": 0.30, "s55b": 0.40},
    "r2_55b_a50":   {"s58": 0.25, "s64": 0.25, "s55b": 0.50},
    # ->s62, expand the single point
    "r2_62_a20":    {"s58": 0.40, "s64": 0.40, "s62": 0.20},
    "r2_62_a30":    {"s58": 0.35, "s64": 0.35, "s62": 0.30},
    # ->23B, continues to accelerate; push further
    "r2_23b_a50":   {"s58": 0.25, "s64": 0.25, "stage23b": 0.50},
    # two-member extension: the two strongest single directions together
    "r2_55b_23b_25_25": {"s58": 0.25, "s64": 0.25, "s55b": 0.25, "stage23b": 0.25},
    "r2_55b_23b_35_15": {"s58": 0.25, "s64": 0.25, "s55b": 0.35, "stage23b": 0.15},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_state(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = None
    if isinstance(ck, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            cand = ck.get(key)
            if (isinstance(cand, dict) and cand
                    and all(isinstance(v, torch.Tensor) for v in cand.values())):
                state = cand
                break
    if state is None:
        raise ValueError(f"no model state in {path}")
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    states, src = {}, {}
    for name, path in BODIES.items():
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"missing member: {p}")
        src[name] = {"path": str(p), "sha256": sha256_file(p)}
        states[name] = load_state(path)

    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "scope": "COEFFICIENT_SEARCH_ROUND2", "round1": ROUND1,
           "anchor_bstar": 0.696865, "members": src, "grid": {}}

    for name, weights in GRID.items():
        tot = sum(weights.values())
        w = {k: v / tot for k, v in weights.items()}
        keys = None
        for m in w:
            if keys is None:
                keys = set(states[m].keys())
            elif set(states[m].keys()) != keys:
                raise SystemExit(f"{name}: key sets differ at {m}")
        out = {}
        for k in sorted(keys):
            ref = states[list(w)[0]][k]
            if ref.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8,
                             torch.bool):
                out[k] = ref.clone()
                continue
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for m, wt in w.items():
                acc += states[m][k].to(torch.float32) * wt
            out[k] = acc.to(ref.dtype)
        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["grid"][name] = {"weights": weights, "path": str(dst),
                             "sha256": sha256_file(dst)}
        print(f"{name:22s} {weights} -> {dst.name}", flush=True)

    (outdir / "ROUND2_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'ROUND2_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
