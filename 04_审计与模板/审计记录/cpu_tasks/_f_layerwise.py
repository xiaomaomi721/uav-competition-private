"""Module F: ATDA layerwise probes (stage2 vs stage3 alpha split).

Splits the 8k ATDA delta into its stage2 and stage3 parts and recombines them
with independent alpha, so we can measure whether the gain lives in the mid
(stage2) or deep (stage3) block. Other scope byte-identical to theta261.

  f_s2p125_s3p100  : stage2=1.25, stage3=1.00
  f_s2p100_s3p125  : stage2=1.00, stage3=1.25
  f_s2p125_s3p125  : stage2=1.25, stage3=1.25  (control == d_s8k_a1p25)
  f_s2p110_s3p135  : stage2=1.10, stage3=1.35
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from _cpu_common import (THETA261, THETA261_SHA, is_trainable, load_state,
                         load_student, save_checkpoint, save_json)

ATDA = Path("/home/jhy/rare_diag_20260915/atda_20260918")
RUN = ATDA / "run_s2s3_base261"
OUT = ATDA / "cpu_candidates"


def stage_of(name: str):
    if name.startswith("segformer.stages.2."):
        return 2
    if name.startswith("segformer.stages.3."):
        return 3
    return 0


def main():
    base_state, _ = load_state(THETA261)
    stu8k, _ = load_student(RUN / "checkpoint_step8000.pt")

    manifest = []
    specs = [
        ("f_s2p125_s3p100", 1.25, 1.00),
        ("f_s2p100_s3p125", 1.00, 1.25),
        ("f_s2p125_s3p125", 1.25, 1.25),
        ("f_s2p110_s3p135", 1.10, 1.35),
    ]
    for tag, a2, a3 in specs:
        out = {}
        for n in base_state:
            if base_state[n].is_floating_point():
                base = base_state[n].float()
                delta = stu8k[n].float() - base
                s = stage_of(n)
                alpha = a2 if s == 2 else (a3 if s == 3 else 0.0)
                out[n] = (base + alpha * delta).to(base_state[n].dtype)
            else:
                out[n] = base_state[n].clone()  # frozen scope exact
        # assert out-of-scope
        bad = [n for n in base_state if not is_trainable(n)
               and not torch.equal(base_state[n], out[n])]
        assert not bad, (tag, bad[:5])
        rec = save_checkpoint(OUT / f"{tag}.pt", out, {
            "formula": f"stage2*{a2}, stage3*{a3} of (theta8k-theta261)",
            "out_of_scope_byte_identical": True,
            "theta261_sha": THETA261_SHA,
        })
        manifest.append({"tag": tag, "stage2_alpha": a2, "stage3_alpha": a3,
                         "path": rec["path"], "sha256": rec["sha256"],
                         "state_sha256": rec["state_sha256"]})

    save_json(OUT / "manifest_layerwise.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
