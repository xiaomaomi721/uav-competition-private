#!/usr/bin/env python3
"""High-calibration-response soups, CPU only.

WHY THIS EXISTS
---------------
soupB averaged {s62, s55b, s58, s64} with EQUAL weights, but those members are
anything but equal where it matters. Their per-model platform responses are:

    s62    ZERO 0.680973  +C 0.686610   R_C 0.5637pp   <- weakest
    s55b   ZERO 0.679846  +C 0.692777   R_C 1.2931pp
    s58    ZERO 0.676383  +C 0.691242   R_C 1.4859pp
    s64    ZERO 0.675162  +C 0.690209   R_C 1.5047pp   <- strongest

s58 and s64 carry Stage23-grade calibration geometry (1.49 / 1.50pp, versus
Stage23's 1.4789pp). soupB diluted them with s62, whose response is 0.5637pp,
and landed at R_C = 0.8288pp.

The transferable invariant across members of the SAME family is that ZERO
barely differs (0.6752 .. 0.6810, a 0.6pp spread) while R_C differs by ~1pp.
s55b beats s62 on final score by 0.0062 while being 0.0011 LOWER on ZERO --
i.e. the selection criterion for a soup member should be its response, not its
zero-bias score. soupB picked by ZERO and got the weaker half.

This script makes the alternative soups. Weights only, float32 average, integer
buffers copied, key sets and shapes checked exactly -- same contract and same
fail-closed behaviour as make_soup.py.

CPU only. No GPU. Writes only under --outdir.
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
    "s62": f"{B}/stage62_runs/stage62_w1_resume55b_pseudo500_seed42/best_miou.pt",
    "s55b": f"{B}/stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt",
    "s58": f"{B}/stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt",
    "s64": f"{B}/stage64_runs/stage64_w1_exp64_b3_cwfa_wd0005_seed42/best_miou.pt",
}
# platform facts, for the record only
PLATFORM = {
    "s62": {"zero": 0.680973, "champ": 0.686610},
    "s55b": {"zero": 0.679846, "champ": 0.692777},
    "s58": {"zero": 0.676383, "champ": 0.691242},
    "s64": {"zero": 0.675162, "champ": 0.690209},
}

SOUPS = {
    "soupC": ["s58", "s64"],                 # both Stage23-grade response
    "soupD": ["s58", "s64", "s55b"],         # add the second-strongest
    "soupE": ["s58", "s64", "s55b", "s62"],  # = soupB, kept as the control
    "soupF": ["s62", "s58", "s64"],          # drop s55b only
    "soupG": ["s64", "s62"],                 # best single response + lineage partner
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

    states = {}
    src = {}
    for name, path in BODIES.items():
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"missing member: {p}")
        src[name] = {"path": str(p), "sha256": sha256_file(p),
                     "bytes": p.stat().st_size}
        states[name] = load_state(path)

    record = {"created": datetime.now(timezone.utc).isoformat(),
              "members": src, "platform": PLATFORM, "soups": {}}

    for soup_name, members in SOUPS.items():
        keys = None
        for m in members:
            if keys is None:
                keys = set(states[m].keys())
            elif set(states[m].keys()) != keys:
                raise SystemExit(f"{soup_name}: key sets differ at {m}")
        out = {}
        for k in sorted(keys):
            tensors = [states[m][k] for m in members]
            ref = tensors[0]
            for t in tensors[1:]:
                if t.shape != ref.shape:
                    raise SystemExit(f"{soup_name}/{k}: shape {t.shape} != {ref.shape}")
            if ref.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8,
                             torch.bool):
                out[k] = ref.clone()          # integer buffers are copied, never averaged
                continue
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for t in tensors:
                acc += t.to(torch.float32)
            acc /= len(tensors)
            out[k] = acc.to(ref.dtype)

        dst = outdir / f"{soup_name}.pt"
        torch.save({"model": out}, dst)
        shas = {m: src[m]["sha256"] for m in members}
        mean_zero = sum(PLATFORM[m]["zero"] for m in members) / len(members)
        mean_rc = sum(PLATFORM[m]["champ"] - PLATFORM[m]["zero"] for m in members) / len(members)
        record["soups"][soup_name] = {
            "members": members, "member_sha256": shas,
            "path": str(dst), "sha256": sha256_file(dst),
            "bytes": dst.stat().st_size, "tensors": len(out),
            "member_mean_zero": round(mean_zero, 6),
            "member_mean_R_C_pp": round(100 * mean_rc, 4),
        }
        print(f"{soup_name:7s} {len(members)} members -> {dst.name}  "
              f"({dst.stat().st_size} B)  mean ZERO {mean_zero:.6f}  "
              f"mean R_C {100*mean_rc:.4f}pp", flush=True)

    (outdir / "SOUP_FAMILY_RECORD.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'SOUP_FAMILY_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
