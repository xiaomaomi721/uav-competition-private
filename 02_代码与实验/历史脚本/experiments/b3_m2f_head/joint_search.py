"""Stage D: a SMALL, interpretable joint search over the 7 class offsets.

Deliberately not a large random sweep. C2 showed the one-dimensional response is
weak (best single class +0.0035 pp; the seven 1D bests sum to ~0.012 pp against a
1.5867 pp gap to the target). A large random search would be unjustified.

WHAT MAKES THIS A JOINT SEARCH AND NOT C2 REPEATED
  When only offset k varies and the other seven are held at their current joint
  values, the winner among those seven is a per-pixel constant, so

      pred = k      if  S_k + b_k > other_v
                    or (S_k + b_k == other_v and k < other_a)
      pred = other_a otherwise

  is exactly equivalent to the full 8-channel argmax. The critical detail is
  that `other_v` / `other_a` MUST be computed from  S_j + b_j, i.e. with the
  other seven offsets applied. Building them from the raw scores would silently
  turn every trial into "only class k has a bias, everything else is zero",
  which is exactly the 1D C2 problem again. `build_ref` therefore takes the full
  current vector, and a runtime cross-check compares the fast score against the
  full 8-channel score on random trials and aborts on any disagreement.

Sequence:
  1. zero vector, asserted to reproduce the frozen TTA8 ZERO baseline
  2. joint_1d_best: the seven biases C2 froze, used verbatim, never re-estimated
  3. `rounds` passes of coordinate descent; within a pass each of the 7 offsets
     is swept over a small neighbourhood of the CURRENT joint vector, with the
     span shrinking twice per class
  4. every reported point re-scored with the full 8-channel exact argmax

Reported set: zero, joint_1d_best, joint_local_best, plus each accepted move,
each carrying exact mIoU, 8 per-class IoUs, the 9x9 confusion matrix and the four
baseline->biased transitions.

1399 validation images only. test1 is never consulted to choose a bias.
No training, no packaging.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

SHM = Path("/dev/shm/tta8_56k_v2")
ROOT = Path("/home/jhy/rare_diag_20260915/m2f_gpu_20260915")
C2 = ROOT / "c2_56k" / "C2_RESULT.json"
OUT = ROOT / "c2_56k"
NAMES = ["Background", "Building", "Road", "Water", "Barren", "Vegetation",
         "Agricultural", "Vehicle"]
EXPECTED_ZERO = 0.8173524135
BACKGROUND = 0          # held at zero: only 7 degrees of freedom are searched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--span-frac", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--crosschecks", type=int, default=6,
                    help="random trials whose fast score is compared against the "
                         "full 8-channel score; a disagreement aborts")
    ap.add_argument("--out", default="JOINT_RESULT.json")
    args = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "48")))

    if not C2.exists():
        raise SystemExit("C2_RESULT.json missing; run C2 first")
    c2 = json.loads(C2.read_text())

    # ---- the seven C2 biases, used verbatim, never re-estimated ----
    best = {}
    for name, d in c2["per_class_best"].items():
        k = NAMES.index(name)
        if k == BACKGROUND:
            raise SystemExit("Background must not appear in per_class_best")
        best[k] = float(d["bias"])
    if set(best) != set(range(1, 8)):
        raise SystemExit(f"need all 7 adjustable classes, got {sorted(best)}")
    base_miou_c2 = float(c2["baseline_zero_miou"])
    print("C2 frozen 1D best biases (used verbatim):", flush=True)
    for k in sorted(best):
        print(f"   {NAMES[k]:13s} {best[k]:+.9f}   hex={best[k].hex()}", flush=True)

    stems = sorted(p.stem for p in SHM.glob("*.npy")
                   if not p.stem.startswith("gt_") and p.stem != "_DONE")
    if args.max_images:
        stems = stems[:args.max_images]
    n = len(stems)
    S = [torch.from_numpy(np.load(SHM / f"{s}.npy")) for s in stems]
    GT = [torch.from_numpy(np.load(SHM / f"gt_{s}.npy")).long().reshape(-1)
          for s in stems]
    print(f"loaded {n} validation images", flush=True)
    H, W = S[0].shape[:2]

    def conf_miou(preds):
        cm = np.zeros((9, 9), dtype=np.int64)
        for i in range(n):
            g = GT[i]
            m = g != 0
            cm += np.bincount((g[m] * 9 + preds[i][m].to(torch.int64)).numpy(),
                              minlength=81).reshape(9, 9)
        sub = cm[1:, 1:].astype(np.float64)
        inter = np.diag(sub)
        union = sub.sum(1) + sub.sum(0) - inter
        ok = union > 0
        iou = np.where(ok, inter / np.maximum(union, 1e-12), np.nan)
        return cm, float(np.nanmean(iou[ok])), iou

    def full_pred(vec):
        """Exact 8-channel argmax with the FULL joint vector applied."""
        b = torch.tensor(vec, dtype=torch.float32).view(1, 1, 8)
        return [((t + b).argmax(-1).reshape(-1) + 1).to(torch.uint8) for t in S]

    def score_full(vec):
        return conf_miou(full_pred(vec))

    # ---- 1. zero baseline ----
    cm0, m_zero, iou0 = score_full([0.0] * 8)
    full_run = (args.max_images == 0)
    print(f"ZERO baseline ({n} images): {m_zero:.10f}", flush=True)
    if full_run:
        # the frozen reference is defined on the full 1399-map validation set, so
        # this assertion only applies when the whole set is used
        if abs(m_zero - EXPECTED_ZERO) > 5e-10 or abs(m_zero - base_miou_c2) > 5e-10:
            raise SystemExit(f"ZERO baseline mismatch: got {m_zero!r}, frozen "
                             f"{EXPECTED_ZERO!r}, C2 {base_miou_c2!r}")
        print(f"ZERO baseline reproduced the frozen value {EXPECTED_ZERO}", flush=True)
    else:
        print("subset run: the frozen-value assertion is skipped by design "
              "(it is defined on the full 1399-map set)", flush=True)

    def build_ref(k, vec):
        """max and argmax over j != k of  S_j + vec[j].

        `vec` is the CURRENT joint vector. Applying it here is what keeps this a
        joint search; using raw scores would degenerate every trial into a 1D
        C2-style scan with the other six offsets set to zero.
        """
        keep = [j for j in range(8) if j != k]
        bvec = torch.tensor([float(vec[j]) for j in keep],
                            dtype=torch.float32).view(1, len(keep))
        kt = torch.tensor(keep, dtype=torch.int64)
        ov = torch.empty(n, H * W, dtype=torch.float32)
        oa = torch.empty(n, H * W, dtype=torch.uint8)
        for i, t in enumerate(S):
            r = t.reshape(-1, 8)[:, keep] + bvec
            a = r.argmax(1)
            ov[i] = r.gather(1, a.unsqueeze(1)).squeeze(1)
            oa[i] = kt[a]
            del r, a
        return ov, oa

    def score_fast(k, c, ov, oa):
        """Joint score of the vector that equals `vec` except at k, which is c."""
        preds = []
        for i, t in enumerate(S):
            sc = t.reshape(-1, 8)[:, k] + float(c)
            win = (sc > ov[i]) | ((sc == ov[i]) & (k < oa[i]))
            preds.append(torch.where(
                win, torch.full_like(oa[i], k + 1, dtype=torch.int64),
                oa[i].to(torch.int64) + 1).to(torch.uint8))
            del sc, win
        out = conf_miou(preds)
        del preds
        return out

    # ---- 2. joint vector assembled from the seven C2 biases ----
    j1 = [0.0] * 8
    for k in best:
        j1[k] = best[k]
    assert j1[BACKGROUND] == 0.0, "Background must stay 0"
    cm1, m_j1, iou1 = score_full(j1)
    print(f"JOINT_1D_BEST {m_j1:.10f} ({(m_j1-m_zero)*100:+.4f} pp) "
          f"{[round(x,6) for x in j1]}", flush=True)

    # ---- 3. coordinate descent on the joint vector ----
    vec = list(j1)
    cur = m_j1
    trace = [{"round": 0, "class": None, "bias": None, "vector": list(vec),
              "accepted": None, "miou": cur, "gain_pp": (cur - m_zero) * 100}]
    t0 = time.time()
    for rnd in range(1, args.rounds + 1):
        accepted_any = False
        for k in range(1, 8):
            span = max(abs(best[k]) * args.span_frac, 0.002)
            for shrink in range(2):
                ov, oa = build_ref(k, vec)          # <-- uses the CURRENT vector
                cands = np.linspace(vec[k] - span, vec[k] + span, args.grid)
                best_c, best_m = vec[k], cur
                for c in cands:
                    _, m, _ = score_fast(k, float(c), ov, oa)
                    if m > best_m + 1e-12:
                        best_m, best_c = m, float(c)
                accepted = best_m > cur + 1e-12
                if accepted:
                    vec[k], cur = best_c, best_m
                    accepted_any = True
                trace.append({"round": rnd, "class": NAMES[k], "bias": best_c,
                              "vector": list(vec), "accepted": bool(accepted),
                              "miou": cur, "gain_pp": (cur - m_zero) * 100})
                span /= 3.0
                del ov, oa
        print(f"  round {rnd}: {cur:.10f} ({(cur-m_zero)*100:+.4f} pp) "
              f"{[round(x,6) for x in vec]}", flush=True)
        if not accepted_any:
            break
    cd_vec, cd_m = list(vec), cur
    print(f"coordinate descent finished in {time.time()-t0:.0f}s", flush=True)

    # ---- runtime guard: fast score must equal full 8-channel score ----
    print(f"cross-checking {args.crosschecks} trials fast-vs-full-8", flush=True)
    rng = np.random.default_rng(20260916)
    for t_i in range(args.crosschecks):
        k = int(rng.integers(1, 8))
        probe = list(cd_vec)
        probe[k] = float(probe[k] + rng.normal(0, 0.05))
        ov, oa = build_ref(k, cd_vec)
        _, m_fast, _ = score_fast(k, probe[k], ov, oa)
        _, m_full, _ = score_full(probe)
        print(f"   trial {t_i}: {NAMES[k]:13s} fast={m_fast:.10f} "
              f"full8={m_full:.10f} agree={abs(m_fast-m_full) < 1e-12}", flush=True)
        if abs(m_fast - m_full) > 1e-12:
            raise SystemExit(
                f"JOINT FAST PATH != FULL 8-CHANNEL: {NAMES[k]} bias={probe[k]!r} "
                f"fast={m_fast!r} full={m_full!r}. This is the degeneracy bug: "
                f"other_v/other_a must be built from S_j + vec[j].")
        del ov, oa

    # ---- 4. final report points, exact full 8-channel ----
    points = [("zero", [0.0] * 8), ("joint_1d_best", j1),
              ("joint_local_best", cd_vec)]
    for t in trace:
        if t["accepted"]:
            points.append((f"accepted_r{t['round']}_{t['class']}", t["vector"]))
    seen, uniq = set(), []
    for tag, v in points:
        key = tuple(round(x, 9) for x in v)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((tag, list(v)))
    print(f"re-scoring {len(uniq)} report points with full 8-channel argmax",
          flush=True)

    p0 = full_pred([0.0] * 8)

    def transitions_of(preds, k):
        agg = {"corrected_FN": 0, "created_FP": 0,
               "removed_FP": 0, "destroyed_TP": 0}
        for i in range(n):
            new_c = preds[i].to(torch.int64) == (k + 1)
            gc = GT[i] == (k + 1)
            bc = p0[i].to(torch.int64) == (k + 1)
            agg["corrected_FN"] += int((gc & ~bc & new_c).sum())
            agg["created_FP"] += int((~gc & ~bc & new_c).sum())
            agg["removed_FP"] += int((~gc & bc & ~new_c).sum())
            agg["destroyed_TP"] += int((gc & bc & ~new_c).sum())
        return agg

    rows = []
    for tag, v in uniq:
        preds = full_pred(v)
        cm, m, iou = conf_miou(preds)
        trans = {NAMES[k]: transitions_of(preds, k) for k in range(1, 8)}
        del preds
        rows.append({
            "tag": tag, "vector": [float(x) for x in v],
            "exact_miou": m, "gain_pp_vs_zero": (m - m_zero) * 100,
            "per_class_iou": {NAMES[j]: float(iou[j]) for j in range(8)},
            "per_class_iou_gain_pp": {NAMES[j]: (float(iou[j]) - float(iou0[j])) * 100
                                      for j in range(8)},
            "confusion_9x9": cm.tolist(),
            "transitions_vs_zero": trans,
        })
        print(f"   {tag:26s} {m:.10f} ({(m-m_zero)*100:+.4f} pp)", flush=True)
    rows.sort(key=lambda r: -r["exact_miou"])

    res = {
        "schema": "M2F_56K_JOINT_OFFSET_SEARCH_V2_JOINTFIXED",
        "baseline_zero_miou_exact": m_zero,
        "baseline_zero_miou_expected": EXPECTED_ZERO,
        "baseline_zero_miou_c2": base_miou_c2,
        "c2_1d_best_biases_used_verbatim": {NAMES[k]: best[k] for k in sorted(best)},
        "c2_1d_best_biases_hex": {NAMES[k]: best[k].hex() for k in sorted(best)},
        "background_fixed_at_zero": True,
        "degrees_of_freedom": 7,
        "rounds": args.rounds, "grid_points": args.grid,
        "span_frac": args.span_frac,
        "joint_1d_best_miou": m_j1,
        "joint_1d_best_gain_pp": (m_j1 - m_zero) * 100,
        "joint_local_best_miou": cd_m,
        "joint_local_best_gain_pp": (cd_m - m_zero) * 100,
        "joint_local_best_vector": [float(x) for x in cd_vec],
        "coordinate_descent_trace": trace,
        "fast_vs_full8_crosschecks_passed": args.crosschecks,
        "n_report_points": len(rows),
        "candidates": rows,
        "constraints": "1399 validation images only; test1 never consulted; "
                       "no training; no packaging",
        "note": "small controlled joint search, NOT a large random sweep.",
    }
    (OUT / args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                           allow_nan=False))
    print("JOINT_DONE", flush=True)
    print(f"  joint_1d_best   {m_j1:.10f} ({(m_j1-m_zero)*100:+.4f} pp)", flush=True)
    print(f"  joint_local_best {cd_m:.10f} ({(cd_m-m_zero)*100:+.4f} pp)", flush=True)
    print(f"  best vector {[round(x,6) for x in cd_vec]}", flush=True)


if __name__ == "__main__":
    main()
