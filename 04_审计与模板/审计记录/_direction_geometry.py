#!/usr/bin/env python3
"""Direction-geometry screening across the whole B3 checkpoint pool. CPU only.

Why
---
Today's +0.80pp came entirely from combining parameter directions between soup
members, and the marginal return is saturating within the s55b direction. The
error-overlap audit says why: the champion and a Barren-targeted model miss the
SAME pixels 83-91% of the time, and all ten historical models land within
+/-0.02 IoU on every class. The pool is highly correlated.

So the useful question is not "which checkpoint is better" but "which checkpoint
points somewhere NEW". That is a geometry question about the pool, answerable on
CPU in seconds, and it has never been asked -- only 5 of ~40 checkpoints have
ever been used as soup members.

For each checkpoint i, with the soup centre theta_C = (s58+s64)/2 and
d_i = theta_i - theta_C, this measures:

  * cosine similarity of d_i against each known direction (s55b, s62, Stage23B)
  * per-group magnitude of d_i, to see WHERE the movement is
  * the fraction of d_i that is orthogonal to the span of the known directions

A checkpoint whose direction is largely orthogonal to the three known ones is a
candidate for a genuinely new direction, and is worth one TTA8.

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
S58 = B / "stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt"
S64 = B / "stage64_runs/stage64_w1_exp64_b3_cwfa_wd0005_seed42/best_miou.pt"
KNOWN = {
    "s55b": B / "stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt",
    "s62": B / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/best_miou.pt",
    "stage23b": B / "stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/best_miou.pt",
}
# already used as soup members -> excluded from the candidate pool
USED = {"s58", "s64", "s55b", "s62", "stage23b"}


def group_of(key: str) -> str:
    if key.startswith("segformer.stages."):
        return f"stage{key.split('.')[2]}"
    if key.startswith("segformer."):
        return "stage0"
    return "head"


def load(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("model", "model_state_dict", "state_dict"):
        c = ck.get(k)
        if isinstance(c, dict) and c and all(torch.is_tensor(v) for v in c.values()):
            return c
    return None


def flat(state, keys):
    return torch.cat([state[k].float().ravel() for k in keys])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    a, b = load(S58), load(S64)
    keys = sorted(k for k in a if a[k].is_floating_point())
    base = {k: (a[k].float() + b[k].float()) * 0.5 for k in keys}

    known = {}
    for name, p in KNOWN.items():
        st = load(p)
        if st is None:
            raise SystemExit(f"cannot load {p}")
        known[name] = {k: st[k].float() - base[k] for k in keys}

    # candidate pool: every best_miou.pt under stage*_runs
    pool = []
    for p in sorted(B.glob("stage*_runs/*/best_miou.pt")):
        run = p.parent.name
        pool.append((run, p))
    print(f"pool size: {len(pool)}", flush=True)

    rec = {"created": datetime.now(timezone.utc).isoformat(),
           "soup_centre": "(s58+s64)/2", "known_directions": list(KNOWN),
           "n_keys": len(keys), "candidates": {}}

    rows = []
    for run, p in pool:
        short = run.replace("_seed42", "").replace("stage", "s").split("_w1_exp")
        tag = short[0] + ":" + (short[1] if len(short) > 1 else run)
        if any(u in run for u in ("stage58_w1", "stage64_w1", "stage56_w1",
                                  "stage62_w1", "stage23_w1")):
            continue
        st = load(p)
        if st is None:
            continue
        if any(k not in st for k in keys):
            continue
        d = {k: st[k].float() - base[k] for k in keys}

        dn = torch.cat([d[k].ravel() for k in keys])
        nd = float(dn.norm())
        if nd == 0:
            continue
        sims = {}
        for name, dk in known.items():
            v = torch.cat([dk[k].ravel() for k in keys])
            sims[name] = float(torch.dot(dn, v) / (nd * float(v.norm())))

        # per-group norm share
        grp = {}
        for k in keys:
            grp.setdefault(group_of(k), 0.0)
            grp[group_of(k)] += float(d[k].float().pow(2).sum())
        tot = sum(grp.values())
        grp = {g: (v / tot if tot else 0.0) for g, v in grp.items()}

        # residual after removing the span of the known directions
        M = torch.stack([torch.cat([known[n][k].ravel() for k in keys])
                         for n in known])              # 3 x D
        # least squares: coefficients = (M M^T)^-1 M dn
        G = M @ M.t()
        rhs = M @ dn
        try:
            coef = torch.linalg.solve(G.double(), rhs.double()).float()
        except Exception:
            coef = torch.zeros(len(known))
        proj = coef @ M
        resid = dn - proj
        orth_frac = float(resid.norm()) / nd

        rows.append({"tag": tag, "run": run, "norm": nd,
                     "sim_s55b": sims.get("s55b"), "sim_s62": sims.get("s62"),
                     "sim_23b": sims.get("stage23b"),
                     "orthogonal_fraction": orth_frac, "group_share": grp})
        rec["candidates"][tag] = rows[-1]

    rows.sort(key=lambda r: -r["orthogonal_fraction"])
    print("\n=== candidates ranked by ORTHOGONALITY to the three known directions ===")
    print("%-34s %8s %8s %8s %8s %10s" %
          ("checkpoint", "|d|", "cos55b", "cos62", "cos23b", "orth_frac"))
    for r in rows[:20]:
        print("%-34s %8.3f %8.3f %8.3f %8.3f %10.4f"
              % (r["tag"][:34], r["norm"], r["sim_s55b"], r["sim_s62"],
                 r["sim_23b"], r["orthogonal_fraction"]))

    (outdir / "DIRECTION_GEOMETRY.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'DIRECTION_GEOMETRY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
