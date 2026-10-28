#!/usr/bin/env python3
"""Donor gate round 1: 3 mechanisms x 4 functional groups, one group at a time.

Design (GPT-reviewed)
---------------------
Three donors representing genuinely different training mechanisms, not three
geometrically-similar checkpoints:

    SEMEDA      structure/embedding objective   (stage47)
    MaxEnt      entropy objective               (stage52, full6796)
    Paramwise   optimizer geometry              (gpu3_paramwise01)

For each donor, four ISOLATION probes -- only one functional group moves, every
other group stays exactly at the champion:

    E : stage0 + stage1        (early encoder)
    M : stage2                 (bulk encoder, 64% of params)
    D : stage3                 (deepest encoder)
    H : head                   (decode_head, 16 tensors)

Why isolation rather than one composite shot: a composite that loses tells us
nothing about which group cancelled which. Today's evidence is that along the
s55b direction the groups want very different, even opposite, alphas, so a
single composite alpha mixes four different signs together.

Step size: RELATIVE RMS, not raw parameter norm --
    r_g(d) = RMS(d_g) / (RMS(theta_champ,g) + eps)
The reference is the known-good s55b perturbation that produced the current
0.701091 champion, i.e. r_g^ref = |alpha_g*| * RMS(d_s55b,g) / RMS(champ_g).
beta is then chosen so the new donor's relative perturbation matches that. Raw
norm matching would be dominated by parameter scale and by layers whose
parameterisation differs.

Head has no s55b reference (its alpha is 0), so it is given a conservative step
sized from the historical global alpha of 0.10.

BN running statistics are functional inference state (eval uses
(x - running_mean)/sqrt(running_var + eps)), so they are NOT part of the
direction and are pinned to the champion's values in every probe.

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
CHAMP = R / "layerwise3_20260917/N9_e050_dm010.pt"       # 0.701091
SOUPC_PARTS = (B / "stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt",
               B / "stage64_runs/stage64_w1_exp64_b3_cwfa_wd0005_seed42/best_miou.pt")
S55B = B / "stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt"

DONORS = {
    "S": ("semeda", B / "stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42/best_miou.pt"),
    "X": ("maxent", B / "stage52_runs/stage52_w1_exp52_b3_cwfa_maxent_l100_full6796_seed42/best_miou.pt"),
    "W": ("paramwise", Path("/home/jhy/gpu3_paramwise01/runs/"
                            "stage_paramwise01_b3_1024_ce_rarecrop_cwfa_paramwise_seed42/best_miou.pt")),
}
GROUPS = {
    "E": ("stage0", "stage1"),
    "M": ("stage2",),
    "D": ("stage3",),
    "H": ("head",),
}
# the s55b alphas that produced the champion (stage0,1,2,3,head)
ALPHA_REF = {"stage0": 0.5, "stage1": 0.5, "stage2": 0.2, "stage3": -0.1, "head": 0.0}
HEAD_FALLBACK_ALPHA = 0.10       # head had no s55b reference; use global .10
EPS = 1e-12


def group_of(key: str) -> str:
    if key.startswith("segformer.stages."):
        return f"stage{key.split('.')[2]}"
    if key.startswith("segformer."):
        return "stage0"
    return "head"


def is_dir_key(key: str) -> bool:
    """Parameters only. BN running stats are functional inference state, not a
    trainable direction."""
    if "num_batches_tracked" in key:
        return False
    if ".running_" in key:
        return False
    return True


def load(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model", "model_state_dict", "state_dict"):
        c = ck.get(k)
        if isinstance(c, dict) and c and all(torch.is_tensor(v) for v in c.values()):
            return c
    raise SystemExit(f"no model state in {path}")


def rms(t: torch.Tensor) -> float:
    return float(t.float().pow(2).mean().sqrt())


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
    a, b = load(SOUPC_PARTS[0]), load(SOUPC_PARTS[1])
    soupC = {k: (a[k].float() + b[k].float()) * 0.5 for k in a}
    s55 = load(S55B)

    keys = sorted(k for k in champ if is_dir_key(k) and champ[k].is_floating_point())
    # reference relative-RMS for each group, from the known-good s55b perturb.
    # head has alpha=0 along s55b, so it has no reference; per GPT it still needs
    # a probe -- a donor whose head movement helps would be a genuinely new
    # degree of freedom -- so it gets a conservative step sized from the
    # historical global alpha of 0.10 applied to its OWN direction.
    ref = {}
    for g in ("stage0", "stage1", "stage2", "stage3", "head"):
        gk = [k for k in keys if group_of(k) == g]
        if not gk:
            continue
        d = torch.cat([(s55[k].float() - soupC[k].float()).ravel() for k in gk])
        c = torch.cat([champ[k].float().ravel() for k in gk])
        al = ALPHA_REF.get(g, 0.0)
        ref[g] = abs(al) * rms(d) / (rms(c) + EPS)
    # head fallback: relative RMS that a global alpha of 0.10 would produce
    head_keys = [k for k in keys if group_of(k) == "head"]
    if head_keys:
        ch = torch.cat([champ[k].float().ravel() for k in head_keys])
        s5 = torch.cat([(s55[k].float() - soupC[k].float()).ravel() for k in head_keys])
        ref["head"] = HEAD_FALLBACK_ALPHA * rms(s5) / (rms(ch) + EPS)

    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "scope": "DONOR_GATE_ISOLATION", "champion": str(CHAMP),
           "champion_sha256": sha(CHAMP), "champion_score": 0.701091,
           "alpha_ref": ALPHA_REF, "relative_rms_ref": ref,
           "note": "buffers pinned to champion; step from relative-RMS matching",
           "probes": {}}

    print("=== reference relative-RMS from the s55b champion perturbation ===")
    for g, v in ref.items():
        print(f"  {g:7s} r_ref = {v:.6e}")
    print()

    for dtag, (dname, dpath) in DONORS.items():
        if not dpath.is_file():
            print(f"SKIP donor {dname}: missing {dpath}")
            continue
        don = load(dpath)
        for gtag, glist in GROUPS.items():
            gset = set(glist)
            gk = [k for k in keys if group_of(k) in gset]
            if not gk:
                continue
            d = torch.cat([(don[k].float() - soupC[k].float()).ravel() for k in gk])
            c = torch.cat([champ[k].float().ravel() for k in gk])
            target = sum(ref.get(g, 0.0) for g in gset)
            beta = target * (rms(c) + EPS) / (rms(d) + EPS) if target > 0 else 0.0
            name = f"{dtag}_{dname}_{gtag}"
            out = dict(champ)                       # start from champion verbatim
            for k in gk:
                out[k] = (champ[k].float() + beta * (don[k].float() - soupC[k].float())
                          ).to(champ[k].dtype)
            dst = outdir / f"{name}.pt"
            torch.save({"model": out}, dst)
            rec["probes"][name] = {"donor": dname, "group": gtag, "groups": glist,
                                   "beta": beta, "target_rel_rms": target,
                                   "donor_rms": rms(d), "champ_rms": rms(c),
                                   "path": str(dst), "sha256": sha(dst)}
            print(f"{name:22s} groups={glist} beta={beta:.6e}", flush=True)

    (outdir / "DONOR_GATE_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'DONOR_GATE_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
