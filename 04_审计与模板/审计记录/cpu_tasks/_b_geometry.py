"""Module B: ATDA trajectory geometry (CPU tensor arithmetic only).

Loads theta261 and the 1k/2k/4k/6k/8k student states, computes:
  - RMS(theta_t - theta261) per stage and per sub-component (stage2/3 -> blocks,
    layer_norm, patch_embeddings)
  - cos(d2,d4), cos(d4,d6), cos(d6,d8) where d_t = theta_t - theta261
  - incremental u24/u46/u68 and their cosines
  - maps the 7 real platform scores to alpha*RMS(delta) to separate direction
    maturation from pure amplitude
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from _cpu_common import (THETA261, THETA261_SHA, THETA261_SCORE, is_trainable,
                         load_state, load_student, rms, save_json, tensor_sha)

ATDA = Path("/home/jhy/rare_diag_20260915/atda_20260918")
RUN = ATDA / "run_s2s3_base261"
OUT = ATDA / "cpu_geometry"
STEPS = [1000, 2000, 4000, 6000, 8000]

# 7 real platform scores in submission order 272..278
PLATFORM = [
    (272, "2k a1.0", 0.700504),
    (273, "2k a0.5", 0.703288),
    (274, "4k a1.0", 0.705031),
    (275, "4k a0.5", 0.704711),
    (276, "8k a1.0", 0.706048),
    (277, "8k a0.75", 0.705655),
    (278, "8k a1.25", 0.706154),
]


def component_of(name: str) -> str:
    if name.startswith("segformer.stages.2"):
        stage = "stage2"
        rest = name[len("segformer.stages.2."):]
    elif name.startswith("segformer.stages.3"):
        stage = "stage3"
        rest = name[len("segformer.stages.3."):]
    else:
        return "other"
    if rest.startswith("blocks."):
        return f"{stage}.blocks"
    if rest.startswith("layer_norm"):
        return f"{stage}.layer_norm"
    if rest.startswith("patch_embeddings"):
        return f"{stage}.patch_embeddings"
    return f"{stage}.other"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import torch

    base_state, _ = load_state(THETA261)
    states = {}
    for s in STEPS:
        st, _ = load_student(RUN / f"checkpoint_step{s}.pt")
        states[s] = st

    # ---- RMS per stage/component ----
    rms_by_component = {}
    for s in STEPS:
        acc = {}
        for name in base_state:
            if not is_trainable(name):
                continue
            delta = states[s][name].float() - base_state[name].float()
            c = component_of(name)
            d = acc.setdefault(c, [0.0, 0.0])  # [sum sq, count]
            d[0] += float((delta ** 2).sum())
            d[1] += delta.numel()
        rms_by_component[s] = {
            c: round(float(np.sqrt(sq / cnt)), 8) for c, (sq, cnt) in acc.items()
        }

    # overall RMS of full trainable delta
    def full_rms(state):
        sq = 0.0
        cnt = 0
        for name in base_state:
            if not is_trainable(name):
                continue
            delta = state[name].float() - base_state[name].float()
            sq += float((delta ** 2).sum())
            cnt += delta.numel()
        return float(np.sqrt(sq / cnt))

    full_rms_by_step = {s: full_rms(states[s]) for s in STEPS}

    # ---- direction cosines ----
    def cos_of(delta_a, delta_b):
        da, db = [], []
        for name in base_state:
            if not is_trainable(name):
                continue
            a = (states[delta_a][name].float() - base_state[name].float()).reshape(-1)
            b = (states[delta_b][name].float() - base_state[name].float()).reshape(-1)
            da.append(a)
            db.append(b)
        da = torch.cat(da)
        db = torch.cat(db)
        return float((da @ db) / (da.norm() * db.norm() + 1e-30))

    d_cos = {
        "cos_d2_d4": cos_of(2000, 4000),
        "cos_d4_d6": cos_of(4000, 6000),
        "cos_d6_d8": cos_of(6000, 8000),
    }

    # incremental u24 = theta4 - theta2 etc.
    def incremental(a, b):
        return {n: (states[b][n].float() - states[a][n].float()) for n in base_state
                if is_trainable(n)}

    u24 = incremental(2000, 4000)
    u46 = incremental(4000, 6000)
    u68 = incremental(6000, 8000)

    def cos_inc(ua, ub):
        ca, cb = [], []
        for name in base_state:
            if not is_trainable(name):
                continue
            ca.append(ua[name].reshape(-1))
            cb.append(ub[name].reshape(-1))
        ca = torch.cat(ca)
        cb = torch.cat(cb)
        return float((ca @ cb) / (ca.norm() * cb.norm() + 1e-30))

    u_cos = {"cos_u24_u46": cos_inc(u24, u46), "cos_u46_u68": cos_inc(u46, u68)}

    # ---- amplitude mapping: alpha * RMS(delta_t) vs platform score ----
    # For each platform point, we know its alpha and which delta it used.
    alpha_map = {
        "2k": 2000, "4k": 4000, "8k": 8000,
    }
    amplitude_rows = []
    for pid, tag, score in PLATFORM:
        key, a = tag.split(" a")
        alpha = float(a)
        step = alpha_map[key]
        amp = alpha * full_rms_by_step[step]
        amplitude_rows.append({
            "id": pid, "tag": tag, "score": score,
            "delta_step": step, "alpha": alpha,
            "rms_delta": round(full_rms_by_step[step], 8),
            "alpha_times_rms": round(amp, 8),
            "gain_pp_vs_261": round((score - THETA261_SCORE) * 100, 4),
        })

    report = {
        "scope": "atda_trajectory_geometry",
        "theta261_sha": THETA261_SHA,
        "theta261_score": THETA261_SCORE,
        "full_trainable_rms_by_step": full_rms_by_step,
        "rms_by_component": rms_by_component,
        "direction_cosines": d_cos,
        "incremental_cosines": u_cos,
        "amplitude_vs_score": amplitude_rows,
        "note": "rms is over the trainable scope (stages 2+3) only; "
                "out-of-scope params are byte-identical to theta261",
    }
    save_json(OUT / "geometry.json", report)

    # CSV of per-component RMS
    with open(OUT / "rms_by_component.csv", "w") as f:
        comps = sorted(rms_by_component[1000].keys())
        f.write("component," + ",".join(f"s{step}" for step in STEPS) + "\n")
        for c in comps:
            f.write(c + "," + ",".join(f"{rms_by_component[s][c]}" for s in STEPS) + "\n")

    print(json.dumps({
        "full_rms_by_step": full_rms_by_step,
        "direction_cosines": d_cos,
        "incremental_cosines": u_cos,
        "amplitude_vs_score": amplitude_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
