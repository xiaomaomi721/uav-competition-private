#!/usr/bin/env python3
"""Donor gate round 1 (GPT-corrected): 3 mechanisms x functional groups.

Three donors, each a genuinely different training mechanism:
    SEMEDA      structure/embedding objective   (stage47)
    MaxEnt      entropy objective               (stage52, full6796)
    Paramwise   optimizer geometry              (gpu3_paramwise01)

Isolation probes -- only one functional group moves, everything else stays at
the champion (score 0.701091).

Probe set (15)
--------------
    E : stage0 + stage1, alpha>0                     3 donors
    M : stage2, alpha>0                              3 donors
    H : head, alpha>0                                3 donors
    Hbn: head params + BN running stats, alpha>0     3 donors  (rescue)
    D+ : stage3, alpha>0                             3 donors
    D- : stage3, alpha<0                             3 donors

Corrections applied versus the first draft
------------------------------------------
1. E uses ONE beta over the JOINT stage0+stage1 set, computed from the combined
   relative RMS of that union. Summing two per-stage references would let the
   union overshoot by up to 2x.
2. stage3's known-good reference is alpha = -0.1, not +0.1. |alpha| fixes
   magnitude but must not erase the sign, so D is probed on BOTH sides.
3. Probes pin the champion's BN buffers, which only answers "is the trainable
   parameter direction useful". If a donor's head probe fails while its running
   stats differ a lot from the champion's (MaxEnt especially), that is not
   grounds to close the head -- hence the Hbn rescue probe, where the buffers
   are interpolated by the same alpha. num_batches_tracked never participates.

Step size is relative-RMS matching against the known-good s55b perturbation:
    r_g(d) = RMS(d_g) / (RMS(champ_g) + eps)
    beta chosen so the donor's r matches the reference r.

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
CHAMP = R / "layerwise3_20260917/N9_e050_dm010.pt"        # 0.701091
S58 = B / "stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt"
S64 = B / "stage64_runs/stage64_w1_exp64_b3_cwfa_wd0005_seed42/best_miou.pt"
S55B = B / "stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt"

DONORS = {
    "S": ("semeda", B / "stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42/best_miou.pt"),
    "X": ("maxent", B / "stage52_runs/stage52_w1_exp52_b3_cwfa_maxent_l100_full6796_seed42/best_miou.pt"),
    "W": ("paramwise", Path("/home/jhy/gpu3_paramwise01/runs/"
                            "stage_paramwise01_b3_1024_ce_rarecrop_cwfa_paramwise_seed42/best_miou.pt")),
}
# the s55b alphas that produced the champion; note stage3 is NEGATIVE
ALPHA_REF = {"stage0": 0.5, "stage1": 0.5, "stage2": 0.2, "stage3": -0.1, "head": 0.0}
HEAD_FALLBACK_ALPHA = 0.10
EPS = 1e-12


def group_of(key: str) -> str:
    if key.startswith("segformer.stages."):
        return f"stage{key.split('.')[2]}"
    if key.startswith("segformer."):
        return "stage0"
    return "head"


def is_param(key: str) -> bool:
    return "num_batches_tracked" not in key and ".running_" not in key


def is_buffer(key: str) -> bool:
    return "num_batches_tracked" not in key and ".running_" in key


def load(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model", "model_state_dict", "state_dict"):
        c = ck.get(k)
        if isinstance(c, dict) and c and all(torch.is_tensor(v) for v in c.values()):
            return c
    raise SystemExit(f"no model state in {path}")


def rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


def cat(state, keys, other, sub=False):
    if sub:
        return torch.cat([(state[k].float() - other[k].float()).ravel() for k in keys])
    return torch.cat([state[k].float().ravel() for k in keys])


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    champ = load(CHAMP)
    a, b = load(S58), load(S64)
    soupC = {k: (a[k].float() + b[k].float()) * 0.5 for k in a}
    s55 = load(S55B)

    pkeys = sorted(k for k in champ if is_param(k) and champ[k].is_floating_point())
    bkeys = sorted(k for k in champ if is_buffer(k) and champ[k].is_floating_point())

    def ref_for(gset):
        gk = [k for k in pkeys if group_of(k) in gset]
        d = cat(s55, gk, soupC, sub=True)
        c = cat(champ, gk, None)
        al = max(abs(ALPHA_REF.get(g, 0.0)) for g in gset)
        if al == 0.0:
            al = HEAD_FALLBACK_ALPHA
        return al * rms(d) / (rms(c) + EPS), gk

    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "scope": "DONOR_GATE_ROUND1", "champion": str(CHAMP),
           "champion_sha256": sha(CHAMP), "champion_score": 0.701091,
           "alpha_ref": ALPHA_REF, "probes": {}}

    # probe table: (suffix, group set, sign)
    PROBES = [
        ("E", ("stage0", "stage1"), +1),
        ("M", ("stage2",), +1),
        ("H", ("head",), +1),
        ("Dp", ("stage3",), +1),
        ("Dm", ("stage3",), -1),
    ]

    for dtag, (dname, dpath) in DONORS.items():
        if not dpath.is_file():
            print(f"SKIP {dname}: missing")
            continue
        don = load(dpath)
        for suffix, gset, sign in PROBES:
            target, gk = ref_for(gset)
            d = cat(don, gk, soupC, sub=True)
            c = cat(champ, gk, None)
            beta = sign * target * (rms(c) + EPS) / (rms(d) + EPS)
            name = f"{dtag}_{dname}_{suffix}"
            out = dict(champ)
            for k in gk:
                out[k] = (champ[k].float() + beta * (don[k].float() - soupC[k].float())
                          ).to(champ[k].dtype)
            dst = outdir / f"{name}.pt"
            torch.save({"model": out}, dst)
            rec["probes"][name] = {"donor": dname, "groups": list(gset), "sign": sign,
                                   "beta": beta, "path": str(dst), "sha256": sha(dst)}
            print(f"{name:24s} {list(gset)} sign={sign:+d} beta={beta:+.6e}", flush=True)

        # ---- Hbn rescue: head params AND BN running stats interpolated ----
        hk = [k for k in pkeys if group_of(k) == "head"]
        hbk = [k for k in bkeys if group_of(k) == "head"]
        target, _ = ref_for(("head",))
        d = cat(don, hk, soupC, sub=True)
        c = cat(champ, hk, None)
        beta = target * (rms(c) + EPS) / (rms(d) + EPS)
        dbuf_gap = None
        if hbk:
            dbuf_gap = float(torch.cat([(don[k].float() - champ[k].float()).ravel()
                                        for k in hbk]).norm())
        name = f"{dtag}_{dname}_Hbn"
        out = dict(champ)
        for k in hk:
            out[k] = (champ[k].float() + beta * (don[k].float() - soupC[k].float())
                      ).to(champ[k].dtype)
        for k in hbk:      # buffers interpolate champion -> donor by the same beta
            out[k] = (champ[k].float() + beta * (don[k].float() - champ[k].float())
                      ).to(champ[k].dtype)
        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["probes"][name] = {"donor": dname, "groups": ["head"], "sign": +1,
                               "buffers": True, "beta": beta,
                               "head_buffer_gap_norm": dbuf_gap,
                               "path": str(dst), "sha256": sha(dst)}
        print(f"{name:24s} head+BNbuffers beta={beta:+.6e}  buffer_gap={dbuf_gap:.3e}",
              flush=True)

    (outdir / "DONOR_GATE_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'DONOR_GATE_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
