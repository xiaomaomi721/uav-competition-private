#!/usr/bin/env python3
"""Phase 0 candidates: matched-delta capability harvest (CPU only).

Displacement convention is IDENTICAL to _make_donor_gate2.py so results are
directly comparable to the existing gate table (S_semeda_M=+0.1483 etc.):

    unit(gset) = max(|ALPHA_REF[g]| for g in gset) * RMS(s55b - soupC | gset)
    scale  s   = k * unit(gset) / RMS(v | gset)
    out        = base + s * v   (ONLY the keys of gset; everything else is
                                 copied verbatim from base, including BN
                                 running_* and num_batches_tracked)

v is a MATCHED delta (single-variable ablation), not donor-soupC.
CPU only. Writes only under --outdir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

E18 = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
E25 = Path("/home/jhy/deploy_exp25/exp25_b3_1024_norecay_20260827T080025Z_36540dd96743")
RE = Path("/home/jhy/gpu2_day_queue/next_experiment/runs/stage23_rareexposure_v2_seed42_56000")
R = Path("/home/jhy/rare_diag_20260915")

CK = {
    "Exp18": E18 / "stage18_runs/stage18_w1_exp18_b3_1024_full_ce_formal_seed42/best_miou.pt",
    "Exp23": E18 / "stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/best_miou.pt",
    "Exp47": E18 / "stage47_runs/stage47_w1_exp47_b3_semeda_formal_seed42/best_miou.pt",
    "Exp25": E25 / "stage25_runs/stage25_w1_exp25_b3_1024_norecay_ce_formal_seed42/best_miou.pt",
    "RareExp": RE / "best_miou.pt",
    "s55b": E18 / "stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt",
    "s58": E18 / "stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt",
    "s64": E18 / "stage64_runs/stage64_w1_exp64_b3_cwfa_wd0005_seed42/best_miou.pt",
}
BASE = {
    "th209": R / "layerwise3_20260917/N9_e050_dm010.pt",
    "th237": R / "donor_gate2_20260917/S_semeda_M.pt",
}
ALPHA_REF = {"stage0": 0.5, "stage1": 0.5, "stage2": 0.2, "stage3": -0.1, "head": 0.0}
HEAD_FALLBACK_ALPHA = 0.10
EPS = 1e-12

# (name, base, A, B, group-set, k)
PLAN = [
    ("A_cleanSEM_M_k050_th209", "th209", "Exp47", "Exp18", ("stage2",), 0.5),
    ("A_cleanSEM_M_k100_th209", "th209", "Exp47", "Exp18", ("stage2",), 1.0),
    ("Z_k000_selfcheck_th209", "th209", "Exp47", "Exp18", ("stage2",), 0.0),
    ("B_nodK_D_k100_th237", "th237", "Exp25", "Exp18", ("stage3",), 1.0),
    ("B_nodK_M_k100_th237", "th237", "Exp25", "Exp18", ("stage2",), 1.0),
    ("B_nodK_E_k100_th237", "th237", "Exp25", "Exp18", ("stage0", "stage1"), 1.0),
    ("B_nodK_H_k100_th237", "th237", "Exp25", "Exp18", ("head",), 1.0),
    ("C_rareExp_E_k100_th237", "th237", "RareExp", "Exp23", ("stage0", "stage1"), 1.0),
    ("C_rareExp_M_k100_th237", "th237", "RareExp", "Exp23", ("stage2",), 1.0),
    ("C_rareExp_D_k100_th237", "th237", "RareExp", "Exp23", ("stage3",), 1.0),
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("model", "model_state_dict", "state_dict"):
        c = ck.get(k)
        if isinstance(c, dict) and c and all(torch.is_tensor(v) for v in c.values()):
            return c
    raise SystemExit("no model state in %s" % p)


def group_of(k):
    if k.startswith("segformer.stages."):
        return "stage" + k.split(".")[2]
    if k.startswith("segformer."):
        return "stage0"
    return "head"


def is_param(k):
    return "num_batches_tracked" not in k and ".running_" not in k


def rms(t):
    return float(t.float().pow(2).mean().sqrt())


def cat_diff(a, b, keys):
    return torch.cat([(a[k].float() - b[k].float()).ravel() for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit("outdir exists: %s" % outdir)
    outdir.mkdir(parents=True)

    need = sorted(set([p[1] for p in PLAN] + [p[2] for p in PLAN] +
                      [p[3] for p in PLAN] + ["s55b", "s58", "s64"]))
    S = {}
    for n in need:
        src = BASE.get(n) or CK.get(n)
        if src is None or not Path(src).is_file():
            raise SystemExit("missing %s: %s" % (n, src))
        S[n] = load(src)
        print("loaded %s" % n, flush=True)

    ref_keys = sorted(S["Exp18"])
    soupC = {k: (S["s58"][k].float() + S["s64"][k].float()) * 0.5 for k in S["s58"]}

    def pkeys(gset):
        return sorted(k for k in ref_keys
                      if group_of(k) in gset and is_param(k)
                      and S["Exp18"][k].is_floating_point())

    def unit(gset):
        ks = pkeys(gset)
        al = max(abs(ALPHA_REF.get(g, 0.0)) for g in gset) or HEAD_FALLBACK_ALPHA
        return al * rms(cat_diff(S["s55b"], soupC, ks)), ks

    units = {}
    print("\n== displacement units (same convention as donor gate) ==", flush=True)
    for gs in [("stage0", "stage1"), ("stage2",), ("stage3",), ("head",)]:
        u, _ = unit(gs)
        units[gs] = u
        print("  %-18s unit = %.6f" % ("+".join(gs), u), flush=True)

    rec = {
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "PHASE0_MATCHED_DELTA",
        "convention": ("unit(gset)=max|ALPHA_REF[g]|*RMS(s55b-soupC|gset); "
                       "scale=k*unit/RMS(v|gset); out=base+scale*v on gset keys only; "
                       "all other keys verbatim from base"),
        "alpha_ref": ALPHA_REF,
        "head_fallback": HEAD_FALLBACK_ALPHA,
        "units": {"+".join(k): v for k, v in units.items()},
        "base_sha": {n: sha(p) for n, p in BASE.items()},
        "source_sha": {n: sha(p) for n, p in CK.items()},
        "candidates": {},
    }

    for name, bname, An, Bn, gset, k in PLAN:
        base, A, B = S[bname], S[An], S[Bn]
        ks = pkeys(gset)
        u = units[gset]
        v = cat_diff(A, B, ks)
        rv = rms(v)
        s = k * u / (rv + EPS) if rv > 0 else 0.0
        out = dict(base)
        for kk in ks:
            out[kk] = (base[kk].float() + s * (A[kk].float() - B[kk].float())).to(base[kk].dtype)
        dst = outdir / (name + ".pt")
        torch.save({"model": out}, dst)
        kset = set(ks)
        untouched = [kk for kk in ref_keys if kk not in kset]
        same = all(torch.equal(out[kk], base[kk]) for kk in untouched)
        meas = rms(cat_diff(out, base, ks)) if ks else 0.0
        ok = abs(meas - k * u) <= max(1e-9, 1e-6 * max(1.0, k * u))
        rec["candidates"][name] = {
            "base": bname, "A": An, "B": Bn, "groups": list(gset), "k": k,
            "scale": s, "unit": u, "target_rms": k * u, "measured_rms": meas,
            "rms_ok": bool(ok), "untouched_byte_identical": bool(same),
            "n_keys": len(ks), "path": str(dst), "sha256": sha(dst),
        }
        print("%-26s g=%-16s k=%.2f scale=%+.6f target=%.6f meas=%.6f untouched_ok=%s rms_ok=%s"
              % (name, "+".join(gset), k, s, k * u, meas, same, ok), flush=True)

    z = torch.load(outdir / "Z_k000_selfcheck_th209.pt", map_location="cpu",
                   weights_only=False)["model"]
    ident = all(torch.equal(z[kk], S["th209"][kk]) for kk in ref_keys)
    rec["selfcheck_k0_identical_to_base"] = bool(ident)
    print("\nSELFCHECK k=0 identical to theta209: %s" % ident, flush=True)

    (outdir / "PHASE0_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nwrote %s" % (outdir / "PHASE0_RECORD.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
