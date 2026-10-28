"""Modules C + D + E: construct-only candidates from the ATDA checkpoints.

No inference. All checkpoint construction is exact tensor arithmetic with
byte-identical out-of-scope assertion against theta261.

  C: EMA candidates (from 8k gate checkpoint's EMA state)
     c_ema8k_direct, c_05s_05e, c_075s_025e, c_261p125ema
  D: alpha narrow (student8k): alpha = 1.10, 1.15, 1.20, 1.30, 1.35
  E: late extrapolation: vlate = theta8k - theta6k; X025, X050
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from _cpu_common import (THETA261, THETA261_SHA, is_trainable, load_state,
                         load_student, load_ema, save_checkpoint, save_json,
                         rms, tensor_sha)

ATDA = Path("/home/jhy/rare_diag_20260915/atda_20260918")
RUN = ATDA / "run_s2s3_base261"
OUT = ATDA / "cpu_candidates"
OUT.mkdir(parents=True, exist_ok=True)


def assert_out_of_scope(base_state, cand_state):
    """Every non-trainable param must be byte-identical to theta261."""
    bad = [n for n in base_state if not is_trainable(n)
           and not torch.equal(base_state[n], cand_state[n])]
    return {"out_of_scope_byte_identical": not bad, "changed": bad[:10]}


def snap_frozen_to_base(base_state, state):
    """Overwrite every non-trainable key (params AND buffers) with theta261.

    The frozen scope is semantically identity to theta261; the EMA loop averaged
    it too (0.999*a+0.001*b) which is a trainer bug that accumulates float
    round-off over steps. Snapping it back to the exact base removes that noise.
    """
    out = {k: v.clone() for k, v in state.items()}
    for n in base_state:
        if not is_trainable(n):
            out[n] = base_state[n].clone()
    return out


def rms_delta(base_state, cand_state):
    sq = 0.0
    cnt = 0
    for n in base_state:
        if is_trainable(n):
            d = cand_state[n].float() - base_state[n].float()
            sq += float((d ** 2).sum())
            cnt += d.numel()
    return float((sq / cnt) ** 0.5)


def main():
    base_state, _ = load_state(THETA261)
    stu8k, _ = load_student(RUN / "checkpoint_step8000.pt")
    ema8k, _ = load_ema(RUN / "checkpoint_step8000.pt")
    stu6k, _ = load_student(RUN / "checkpoint_step6000.pt")

    manifest = []

    def emit(tag, state, formula, src_sha):
        out_scope = assert_out_of_scope(base_state, state)
        assert out_scope["out_of_scope_byte_identical"], (tag, out_scope)
        rms_v = rms_delta(base_state, state)
        rec = save_checkpoint(OUT / f"{tag}.pt", state, {
            "formula": formula, "source_sha256": src_sha,
            "out_of_scope_byte_identical": True,
            "rms_vs_theta261": round(rms_v, 8),
            "theta261_sha": THETA261_SHA,
        })
        manifest.append({"tag": tag, "path": rec["path"], "sha256": rec["sha256"],
                         "state_sha256": rec["state_sha256"], "formula": formula,
                         "rms_vs_theta261": round(rms_v, 8)})

    # ---- C: EMA candidates (frozen scope snapped back to theta261) ----
    ema_direct = snap_frozen_to_base(base_state, ema8k)
    emit("c_ema8k_direct", ema_direct, "EMA8k (frozen snapped to theta261)",
         tensor_sha(ema8k))

    def mix(a_state, b_state, wa, wb):
        out = {}
        for n in base_state:
            a, b = a_state[n].float(), b_state[n].float()
            out[n] = (wa * a + wb * b).to(base_state[n].dtype)
        return snap_frozen_to_base(base_state, out)

    emit("c_05s_05e", mix(stu8k, ema8k, 0.5, 0.5), "0.5*student8k+0.5*EMA8k",
         tensor_sha(stu8k))
    emit("c_075s_025e", mix(stu8k, ema8k, 0.75, 0.25), "0.75*student8k+0.25*EMA8k",
         tensor_sha(stu8k))

    def compose(base, delta_state, alpha):
        out = {}
        for n in base_state:
            if base_state[n].is_floating_point():
                out[n] = (base_state[n].float()
                          + alpha * (delta_state[n].float() - base_state[n].float())
                          ).to(base_state[n].dtype)
            else:
                out[n] = delta_state[n].clone()
        return snap_frozen_to_base(base, out)

    emit("c_261p125ema", compose(base_state, ema8k, 1.25),
         "theta261+1.25*(EMA8k-theta261)", tensor_sha(ema8k))

    # ---- D: alpha narrow on student8k ----
    for alpha in (1.10, 1.15, 1.20, 1.30, 1.35):
        tag = f"d_s8k_a{str(alpha).replace('.', 'p')}"
        emit(tag, compose(base_state, stu8k, alpha),
             f"theta261+{alpha}*(theta8k-theta261)", tensor_sha(stu8k))

    # ---- E: late extrapolation ----
    vlate = {n: (stu8k[n].float() - stu6k[n].float()) for n in base_state}
    for k, tag in ((0.25, "e_X025"), (0.50, "e_X050")):
        out = {}
        for n in base_state:
            if base_state[n].is_floating_point():
                out[n] = (stu8k[n].float() + k * vlate[n]).to(base_state[n].dtype)
            else:
                out[n] = stu8k[n].clone()
        rms_v8 = rms_delta(base_state, out)
        # rms vs theta8k too
        d8 = {n: out[n].float() - stu8k[n].float() for n in base_state
              if is_trainable(n)}
        sq = sum(float((d8[n] ** 2).sum()) for n in d8)
        cnt = sum(d8[n].numel() for n in d8)
        emit(tag, out, f"theta8k+{k}*vlate (vlate=theta8k-theta6k)",
             tensor_sha(stu8k))

    save_json(OUT / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
