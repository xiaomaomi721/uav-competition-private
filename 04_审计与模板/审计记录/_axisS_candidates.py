#!/usr/bin/env python3
"""Axis S candidate builder (CPU only).

From a P1 stage2-SEMEDA training checkpoint, build candidates

    theta = base + scale * delta_stepN           delta = theta_t - theta209, stage2 only
    scale chosen so that RMS(scale*delta | stage2) = k * R_old

so every candidate is compared at a MATCHED PARAMETER DISPLACEMENT rather than at
a meaningless coefficient. R_old is the displacement the known-good s55b stage2
perturbation produces at alpha=0.2:

    R_old = 0.2 * RMS(s55b - soupC | stage2) = 0.0016307737678289413

Bases:
    th209  -> tests the SEMEDA stage2 DIRECTION against the pre-SEMEDA base
    th237  -> tests the INCREMENT on the current champion (0.702574), which is
              the candidate that actually has to beat the champion

Only the stage2 keys move. Every other key -- including decode_head BN running
stats and num_batches_tracked -- is copied verbatim from the base and verified
byte-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

R = Path("/home/jhy/rare_diag_20260915")
BASES = {
    "th209": R / "layerwise3_20260917/N9_e050_dm010.pt",
    "th237": R / "donor_gate2_20260917/S_semeda_M.pt",
}
R_OLD = 0.0016307737678289413
SCOPE = "segformer.stages.2."


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("model", "model_state_dict", "state_dict"):
        c = ck.get(k)
        if isinstance(c, dict) and c and all(torch.is_tensor(v) for v in c.values()):
            return c
    raise SystemExit(f"no model state in {p}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--base-checkpoint", type=Path, required=True,
                    help="the theta209 the run started from (for delta)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--steps", type=str, default="",
                    help="comma-separated checkpoint steps; default = all found")
    ap.add_argument("--k", type=str, default="0.5,1.0,1.5")
    ap.add_argument("--bases", type=str, default="th209,th237")
    args = ap.parse_args()

    out = args.outdir
    if out.exists():
        raise SystemExit(f"outdir exists: {out}")
    out.mkdir(parents=True)

    init = load(args.base_checkpoint)
    ks = sorted(k for k in init
                if k.startswith(SCOPE) and init[k].is_floating_point()
                and "num_batches_tracked" not in k and ".running_" not in k)

    cks = sorted(args.run_dir.glob("checkpoint_step*.pt"),
                 key=lambda p: int(p.stem.replace("checkpoint_step", "")))
    if args.steps:
        want = {int(s) for s in args.steps.split(",")}
        cks = [c for c in cks if int(c.stem.replace("checkpoint_step", "")) in want]
    if not cks:
        raise SystemExit("no checkpoints matched")

    S = {n: load(p) for n, p in BASES.items()}
    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "scope": "AXIS_S_CANDIDATES", "R_old": R_OLD,
           "delta_source": str(args.base_checkpoint),
           "run_dir": str(args.run_dir), "scope_prefix": SCOPE,
           "convention": "theta = base + scale*delta, scale set so RMS(scale*delta|stage2) = k*R_old",
           "bases": {n: {"path": str(p), "sha256": sha(p)} for n, p in BASES.items()},
           "candidates": {}}

    for ck in cks:
        step = int(ck.stem.replace("checkpoint_step", ""))
        t = load(ck)
        d = torch.cat([(t[k].float() - init[k].float()).ravel() for k in ks])
        r_d = float(d.pow(2).mean().sqrt())
        print(f"\n=== step {step}  RMS(delta|stage2)={r_d:.6f}  /R_old={r_d/R_OLD:.4f} ===",
              flush=True)
        for kstr in [x for x in args.k.split(",") if x]:
            k = float(kstr)
            scale = k * R_OLD / (r_d + 1e-12)
            for bn in [x for x in args.bases.split(",") if x]:
                base = S[bn]
                o = dict(base)
                for kk in ks:
                    o[kk] = (base[kk].float() + scale * (t[kk].float() - init[kk].float())
                             ).to(base[kk].dtype)
                name = f"S_s{step}_k{k:g}_{bn}"
                dst = out / f"{name}.pt"
                torch.save({"model": o}, dst)
                kset = set(ks)
                unt = [kk for kk in base if kk not in kset]
                same = all(torch.equal(o[kk], base[kk]) for kk in unt)
                meas = float(torch.cat([(o[kk].float() - base[kk].float()).ravel()
                                        for kk in ks]).pow(2).mean().sqrt())
                ok = abs(meas - k * R_OLD) < 1e-9
                rec["candidates"][name] = {
                    "step": step, "k": k, "base": bn, "scale": scale,
                    "base_delta_rms": r_d, "target_rms": k * R_OLD,
                    "measured_rms": meas, "rms_ok": bool(ok),
                    "untouched_byte_identical": bool(same),
                    "path": str(dst), "sha256": sha(dst)}
                print(f"  {name:24s} scale={scale:+.6f} target={k*R_OLD:.6f} "
                      f"meas={meas:.6f} rms_ok={ok} untouched_ok={same}", flush=True)

    (out / "AXIS_S_CANDIDATES.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out/'AXIS_S_CANDIDATES.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
