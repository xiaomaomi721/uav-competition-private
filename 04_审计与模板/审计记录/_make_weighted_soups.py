#!/usr/bin/env python3
"""Weighted soups around soupC, CPU only.

Why weighted, not new members
-----------------------------
soupC = equal average of s58 and s64 scored 0.696818, beating the illegal
logit-ensemble reference (0.695637) and the old single-model record (0.693096).
Only two member sets have ever been tried, both with EQUAL weights. The
parameter-averaging map is strongly non-linear (soupB kept 68% of its members'
mean response, soupC kept 93%), so the weights are the unexplored axis, not the
member list -- adding members is what diluted soupB and soupD.

Members and their platform facts (ZERO / +CHAMPION):
    s58      0.676383 / 0.691242   R 1.4859pp
    s64      0.675162 / 0.690209   R 1.5047pp
    s55b     0.679846 / 0.692777   R 1.2931pp
    Stage23B 0.678023 / 0.692812   R 1.4789pp
    s62      0.680973 / 0.686610   R 0.5637pp   <- high ZERO, weak response

Every candidate keeps the mass on the two high-response bodies (s58, s64) and
spends a small budget elsewhere, so the grid tests "can a little ZERO be bought
without paying response" rather than re-rolling the member list.

CPU only -- no GPU. Weights-only checkpoints in the build_model contract.
Writes only under --outdir.
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
PLATFORM = {
    "s58": (0.676383, 1.4859), "s64": (0.675162, 1.5047),
    "s55b": (0.679846, 1.2931), "stage23b": (0.678023, 1.4789),
    "s62": (0.680973, 0.5637),
}

# name -> {member: weight}   (weights are normalised)
GRID = {
    # control: reproduces soupC
    "wA_58_64_50_50":        {"s58": 0.50, "s64": 0.50},
    # asymmetric between the two high-response bodies
    "wB_58_64_40_60":        {"s58": 0.40, "s64": 0.60},
    "wC_58_64_60_40":        {"s58": 0.60, "s64": 0.40},
    "wD_58_64_30_70":        {"s58": 0.30, "s64": 0.70},
    # small budget spent on Stage23B (high response, mid ZERO)
    "wE_58_64_23b_45_45_10": {"s58": 0.45, "s64": 0.45, "stage23b": 0.10},
    "wF_58_64_23b_40_40_20": {"s58": 0.40, "s64": 0.40, "stage23b": 0.20},
    # small budget on s55b (response 1.2931, ZERO 0.679846 the highest of the pool)
    "wG_58_64_55b_45_45_10": {"s58": 0.45, "s64": 0.45, "s55b": 0.10},
    "wH_58_64_55b_40_40_20": {"s58": 0.40, "s64": 0.40, "s55b": 0.20},
    # small budget on s62: the high-ZERO / weak-response body, deliberately tiny
    "wI_58_64_62_45_45_10":  {"s58": 0.45, "s64": 0.45, "s62": 0.10},
    # three-way spread over the three strongest responses
    "wJ_58_64_23b_33each":   {"s58": 1/3, "s64": 1/3, "stage23b": 1/3},
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
           "scope": "WEIGHTED_SOUP_GRID", "members": src,
           "platform_zero_rc": {k: {"zero": v[0], "R_C_pp": v[1]}
                                for k, v in PLATFORM.items()},
           "grid": {}}

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
            for m in w:
                if states[m][k].shape != ref.shape:
                    raise SystemExit(f"{name}/{k}: shape mismatch")
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
        mz = sum(PLATFORM[m][0] * wt for m, wt in w.items())
        mr = sum(PLATFORM[m][1] * wt for m, wt in w.items())
        rec["grid"][name] = {"weights": weights, "normalised": w,
                             "path": str(dst), "sha256": sha256_file(dst),
                             "member_mean_zero": round(mz, 6),
                             "member_mean_R_C_pp": round(mr, 4)}
        print(f"{name:24s} memberZ {mz:.6f}  memberR {mr:.4f}pp  -> {dst.name}",
              flush=True)

    (outdir / "WEIGHTED_GRID_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'WEIGHTED_GRID_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
