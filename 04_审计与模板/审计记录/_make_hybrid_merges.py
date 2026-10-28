#!/usr/bin/env python3
"""Hybrid merges: average the encoder, take the head from a chosen member.

Hypothesis under test
---------------------
The calibration geometry -- what a class-bias vector can actually buy -- is
carried by a small part of the network. The checkpoint splits as

    encoder  628 tensors   44.1M params   ("segformer.*")
    head      16 tensors    3.2M params   ("decode_head.*")

and the class-logit scale/relative structure that a bias operates on is produced
at the very end, by decode_head.classifier (768->8) and the fuse/projection
stack feeding it. If the geometry lives there, then a merge that averages the
encoders (for the ZERO score) while taking the head from a high-response member
should hold both halves -- which is the combination no plain average has managed.

Evidence that motivates it -- today's soups:

    soupC  Z 0.681892  R 1.3900pp  ->  0.696818   (both halves decent)
    soupD  Z 0.684004  R 0.9979pp  ->  0.693983   (Z at M2F level, R lost)

soupD already reaches Z = 0.684004, essentially the best single-model ZERO ever
measured. Its problem is purely the lost response. If the head carries that
response, a hybrid recovers it.

This is the same question the abandoned Bridge-transplant proposal asked, but it
costs a CPU merge instead of a GPU training run.

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
P = "/home/jhy/gpu3_paramwise01/runs/stage_paramwise01_b3_1024_ce_rarecrop_cwfa_paramwise_seed42"
BODIES = {
    "s58": f"{B}/stage58_runs/stage58_w1_exp58_b3_cwfa_ce_regionC_seed42/best_miou.pt",
    "s64": f"{B}/stage64_runs/stage64_w1_exp64_b3_cwfa_wd0005_seed42/best_miou.pt",
    "s55b": f"{B}/stage56_runs/stage56_w1_exp56_b3_cwfa_5597_pseudo500_seed42/best_miou.pt",
    "stage23b": f"{B}/stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/best_miou.pt",
    "paramwise01": f"{P}/best_miou.pt",
}
PLATFORM = {  # (ZERO, R_C pp)
    "s58": (0.676383, 1.4859), "s64": (0.675162, 1.5047),
    "s55b": (0.679846, 1.2931), "stage23b": (0.678023, 1.4789),
    "paramwise01": (0.674232, 1.5398),
}

# name -> (encoder member -> weight, head source)
MERGES = {
    "hA_encAvg58_64__head_s64":     ({"s58": 0.5, "s64": 0.5}, "s64"),
    "hB_encAvg58_64__head_s58":     ({"s58": 0.5, "s64": 0.5}, "s58"),
    "hC_encAvg58_64__head_pw01":    ({"s58": 0.5, "s64": 0.5}, "paramwise01"),
    "hD_encAvg58_64_23b__head_s64": ({"s58": 1/3, "s64": 1/3, "stage23b": 1/3}, "s64"),
    "hE_encAvg58_64_55b__head_s64": ({"s58": 1/3, "s64": 1/3, "s55b": 1/3}, "s64"),
    # controls: full average (should reproduce soupC) and head-only swap
    "hZ_encAvg58_64__head_avg":     ({"s58": 0.5, "s64": 0.5}, None),
    "hF_enc_s64__head_avg58_64":    ({"s64": 1.0}, "AVG"),
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


def is_head(key: str) -> bool:
    return not key.startswith("segformer.")


def avg_tensors(tensors, weights, ref):
    if ref.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool):
        return ref.clone()
    acc = torch.zeros_like(ref, dtype=torch.float32)
    for t, w in zip(tensors, weights):
        acc += t.to(torch.float32) * w
    return acc.to(ref.dtype)


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
           "scope": "HYBRID_ENCODER_AVG_HEAD_SWAP", "members": src,
           "platform_zero_rc": {k: {"zero": v[0], "R_C_pp": v[1]}
                                for k, v in PLATFORM.items()},
           "merges": {}}

    for name, (ew, head_src) in MERGES.items():
        tot = sum(ew.values())
        w = {k: v / tot for k, v in ew.items()}
        keys = sorted(states[list(w)[0]].keys())
        out = {}
        for k in keys:
            if not is_head(k):
                out[k] = avg_tensors([states[m][k] for m in w],
                                     list(w.values()), states[list(w)[0]][k])
                continue
            # head tensor
            if head_src is None or head_src == "AVG":
                members = list(w) if head_src == "AVG" else list(w)
                weights = list(w.values()) if head_src == "AVG" else list(w.values())
                out[k] = avg_tensors([states[m][k] for m in members], weights,
                                     states[members[0]][k])
            else:
                out[k] = states[head_src][k].clone()

        dst = outdir / f"{name}.pt"
        torch.save({"model": out}, dst)
        rec["merges"][name] = {"encoder_weights": ew, "head_source": head_src or "avg",
                               "path": str(dst), "sha256": sha256_file(dst)}
        print(f"{name:32s} enc={list(ew)} head={head_src or 'avg'}  "
              f"{dst.stat().st_size} B", flush=True)

    (outdir / "HYBRID_RECORD.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outdir/'HYBRID_RECORD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
