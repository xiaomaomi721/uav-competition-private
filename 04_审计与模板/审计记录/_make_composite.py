#!/usr/bin/env python3
"""Composite donor packages (phase B of the overnight chain). CPU only.

The isolation probes answer "which group of which donor helps". A composite is
the natural follow-up only if some donor has real positive groups -- but the
platform scores will not be read until morning, so this round cannot be
*conditional* on them. Instead it covers both plausible shapes:

  A: donor injected with the champion's OWN measured profile
     (early +, mid +, deep NEGATIVE, head 0) -- the shape that won on s55b
  B: donor injected with a uniform small alpha on every group including head
     -- the shape that would win IF the new donor's head/geometry differs, which
     is exactly what the H probes test

Both are relative-RMS matched so the total perturbation magnitude matches the
known-good s55b perturbation, and BN running stats are pinned to the champion
(stage 1 probes isolate trainable parameters; buffers were tested separately).

Also includes the double-direction merges: champion + donor direction + the
known-good s55b direction, since the s55b direction is worth +0.1776pp on its
own and a new donor only has to ADD to it, not replace it.

CPU only. Writes only under --outdir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

B = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
R = Path("/home/jhy/rare_diag_20260915")
CHAMP = R / "layerwise3_20260917/N9_e050_dm010.pt"      # 0.701091
S58 = B / "stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt"
S64 = B / "stage64_runs/stage64_w1_exp64_b3_cwfa_wd0005_seed42/best_miou.pt"
S55B = B / "stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt"

DONORS = {
    "semeda": B / "stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42/best_miou.pt",
    "maxent": B / "stage52_runs/stage52_w1_exp52_b3_cwfa_maxent_l100_full6796_seed42/best_miou.pt",
    "paramwise": Path("/home/jhy/gpu3_paramwise01/runs/"
                      "stage_paramwise01_b3_1024_ce_rarecrop_cwfa_paramwise_seed42/best_miou.pt"),
}
# the champion's measured profile along the s55b direction
CHAMP_PROFILE = {"stage0": 0.5, "stage1": 0.5, "stage2": 0.2, "stage3": -0.1, "head": 0.0}
UNIFORM_PROFILE = {"stage0": 0.15, "stage1": 0.15, "stage2": 0.15, "stage3": 0.15, "head": 0.05}
EPS = 1e-12


def group_of(key: str) -> str:
    if key.startswith("segformer.stages."):
        return f"stage{key.split('.')[2]}"
    if key.startswith("segformer."):
        return "stage0"
    return "head"


def is_param(k):
    return "num_batches_tracked" not in k and ".running_" not in k


def load(p: Path) -> dict:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("model", "model_state_dict", "state_dict"):
        c = ck.get(k)
        if isinstance(c, dict) and c and all(torch.is_tensor(v) for v in c.values()):
            return c
    raise SystemExit(f"no model state in {p}")


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(champ, soupC, donor, profile, beta_scale=1.0):
    """theta = champ + sum_g beta_g * (donor - soupC)_g, beta from relative RMS."""
    pkeys = sorted(k for k in champ if is_param(k) and champ[k].is_floating_point())
    out = dict(champ)
    betas = {}
    for g in ("stage0", "stage1", "stage2", "stage3", "head"):
        al = profile.get(g, 0.0)
        gk = [k for k in pkeys if group_of(k) == g]
        if not gk or al == 0.0:
            continue
        d = torch.cat([(donor[k].float() - soupC[k].float()).ravel() for k in gk])
        # reference amplitude: what the s55b direction would give at |al| on THIS group
        ref = torch.cat([(S55_STATE[k].float() - soupC[k].float()).ravel() for k in gk])
        c = torch.cat([champ[k].float().ravel() for k in gk])
        target = abs(al) * rms(ref) / (rms(c) + EPS)
        beta = (1.0 if al > 0 else -1.0) * beta_scale * target * (rms(c) + EPS) / (rms(d) + EPS)
        for k in gk:
            out[k] = (champ[k].float() + beta * (donor[k].float() - soupC[k].float())
                      ).to(champ[k].dtype)
        betas[g] = beta
    return out, betas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    global S55_STATE
    champ = load(CHAMP)
    a, b = load(S58), load(S64)
    soupC = {k: (a[k].float() + b[k].float()) * 0.5 for k in a}
    S55_STATE = load(S55B)

    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "scope": "DONOR_COMPOSITE", "champion": str(CHAMP),
           "champion_score": 0.701091, "probes": {}}

    for dname, dpath in DONORS.items():
        if not dpath.is_file():
            print(f"SKIP {dname}: missing")
            continue
        don = load(dpath)
        for pname, prof in (("P1champ", CHAMP_PROFILE), ("P2unif", UNIFORM_PROFILE)):
            out, betas = build(champ, soupC, don, prof)
            name = f"C_{dname}_{pname}"
            dst = outdir / f"{name}.pt"
            torch.save({"model": out}, dst)
            rec["probes"][name] = {"donor": dname, "profile": pname,
                                   "betas": {k: float(v) for k, v in betas.items()},
                                   "path": str(dst), "sha256": sha(dst)}
            print(f"{name:26s} betas={ {k: round(v,4) for k,v in betas.items()} }", flush=True)
        # double-direction: champion already carries s55b; add donor on top
        out, betas = build(champ, soupC, don, CHAMP_PROFILE, beta_scale=0.5)
        name = f"C_{dname}_P3half"
        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["probes"][name] = {"donor": dname, "profile": "P1champ@0.5",
                               "betas": {k: float(v) for k, v in betas.items()},
                               "path": str(dst), "sha256": sha(dst)}
        print(f"{name:26s} betas={ {k: round(v,4) for k,v in betas.items()} } (half)", flush=True)

    (outdir / "COMPOSITE_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'COMPOSITE_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
