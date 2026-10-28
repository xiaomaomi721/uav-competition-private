"""Joint fast scorer vs full 8-channel, replayed on the REAL coordinate-descent trials.

THE BUG IN THE PREVIOUS VERSION
  The trial list was rebuilt by iterating rounds and classes while never updating
  the running vector from accepted moves. After the first accepted move the
  reconstructed state diverged from what the search actually scored, so the
  "verified" candidates were mostly states the search never visited. A smoke test
  cannot catch this: fast == full holds for ANY base vector, so passing on an
  invented state proves the formula, not the replay.

THE REPLAY
  joint_search.py's trace stores, for every one of its 2*7*2 = 28 decisions, the
  base state and the accepted/rejected outcome. Those records ARE the state
  machine, so the replay is driven by the trace rather than reconstructed:

      cur = joint_1d_best
      for each decision record t:
          k     = class index of t
          base  = cur                      (asserted to equal the search's own
                                            pre-decision state via the helpers)
          span  = max(|c2_1d_best[k]| * 0.5, 0.002) / 3**shrink
          grid  = linspace(base[k] - span, base[k] + span, 9)
          verify(base, k, grid)            # the 9 candidates the search scored
          if t.accepted: cur = t.vector
          else:          assert t.vector == cur

  Asserted at the end: 28 decisions, only coordinate k changed on an accepted
  move, and the final cur equals JOINT_RESULT's joint_local_best_vector.

PERFORMANCE
  other_v / other_a are built once per (base, k) and reused across the 9
  candidates; GT is cached in RAM; the full side stays an INDEPENDENT
  argmax(S + v') and never reuses the other-max machinery.

M_all == 0 implies M_valid == 0, so the gate is TOTAL_ALL_PIXEL_MISMATCH == 0;
GT is loaded only to report M_valid and the three counts separately.

Read-only. No training, no packaging.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

SHM = Path("/dev/shm/tta8_56k_v2")
ROOT = Path("/home/jhy/rare_diag_20260915/m2f_gpu_20260915")
C2 = ROOT / "c2_56k" / "C2_RESULT.json"
JOINT = ROOT / "c2_56k" / "JOINT_RESULT.json"
OUT = ROOT / "c2_56k"
NAMES = ["Background", "Building", "Road", "Water", "Barren", "Vegetation",
         "Agricultural", "Vehicle"]
SPAN_FRAC = 0.5
GRID = 9
SPATIAL_PX = 1466957824          # frozen, from the C1/pixelwise verification
VALID_PX = 1445194394


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--out", default="JOINT_FASTPATH_PIXELWISE.json")
    args = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("THREADS", "48")))

    jr = json.loads(JOINT.read_text())
    c2 = json.loads(C2.read_text())
    best = {NAMES.index(k): float(v["bias"])
            for k, v in c2["per_class_best"].items()}
    trace = jr["coordinate_descent_trace"]
    decisions = [t for t in trace if t["class"] is not None]
    if len(decisions) != 28:
        raise SystemExit(f"expected 28 decision records, got {len(decisions)}")

    stems = sorted(p.stem for p in SHM.glob("*.npy")
                   if not p.stem.startswith("gt_") and p.stem != "_DONE")
    if args.max_images:
        stems = stems[:args.max_images]
    n = len(stems)
    S = [torch.from_numpy(np.load(SHM / f"{s}.npy")) for s in stems]
    GT = [torch.from_numpy(np.load(SHM / f"gt_{s}.npy")).long().reshape(-1)
          for s in stems]
    print(f"loaded {n} images (scores + GT cached in RAM)", flush=True)

    cur = [0.0] * 8
    for k in best:
        cur[k] = best[k]
    start = trace[0]["vector"]
    assert [float(x) for x in start] == cur, (
        "trace start != joint_1d_best assembled from C2 bests")

    # ---- build the REAL trial list by replaying the trace state machine ----
    trials = []
    for di, t in enumerate(decisions):
        k = NAMES.index(t["class"])
        shrink = di % 2                       # two decisions per (round, class)
        span = max(abs(best[k]) * SPAN_FRAC, 0.002) / (3.0 ** shrink)
        base = list(cur)
        grid = [float(np.float32(x))
                for x in np.linspace(base[k] - span, base[k] + span, GRID)]
        trials.append({"decision": di, "round": t["round"], "class": t["class"],
                       "k": k, "base": base, "span": span, "cands": grid,
                       "search_accepted": bool(t["accepted"])})
        # advance the state machine exactly as the search did
        tv = [float(x) for x in t["vector"]]
        if t["accepted"]:
            changed = [j for j in range(8) if tv[j] != base[j]]
            assert changed == [k], (
                f"decision {di} accepted but changed coordinates {changed}, "
                f"expected only {k}")
            cur = tv
        else:
            assert tv == base, (
                f"decision {di} rejected but the recorded vector differs from "
                f"the pre-decision state")
    final_expect = [float(x) for x in jr["joint_local_best_vector"]]
    assert cur == final_expect, (
        f"replayed final vector != joint_local_best_vector\n got {cur}\n exp {final_expect}")
    print("trace replay validated: 28 decisions, single-coordinate accepted moves, "
          "final vector reproduces joint_local_best_vector", flush=True)
    print(f"real trials to verify: {len(trials)} groups x {GRID} = "
          f"{len(trials)*GRID} candidates", flush=True)
    print(f"accepted among them: {sum(1 for t in trials if t['search_accepted'])}",
          flush=True)

    total_all = 0
    total_valid = 0
    comparisons = 0
    bad_first = None
    per_group = []

    for gi, g in enumerate(trials):
        k, base = g["k"], g["base"]
        keep = [j for j in range(8) if j != k]
        kt = torch.tensor(keep, dtype=torch.int64)
        bvec = torch.tensor([float(base[j]) for j in keep],
                            dtype=torch.float32).view(1, len(keep))
        ref = []
        for i, t in enumerate(S):
            tf = t.reshape(-1, 8)
            r = tf[:, keep] + bvec
            a = r.argmax(1)
            ref.append((r.gather(1, a.unsqueeze(1)).squeeze(1), kt[a]))
            del r, a, tf
        gm_all = gm_valid = 0
        for c in g["cands"]:
            vp = list(base)
            vp[k] = c
            bfull = torch.tensor(vp, dtype=torch.float32).view(1, 8)
            c32 = float(np.float32(c))
            for i, t in enumerate(S):
                tf = t.reshape(-1, 8)
                ov, oa = ref[i]
                sc = tf[:, k] + c32
                win = (sc > ov) | ((sc == ov) & (k < oa))
                fast = torch.where(win,
                                   torch.full_like(oa, k + 1, dtype=torch.int64),
                                   oa.to(torch.int64) + 1)
                full = (tf + bfull).argmax(1).to(torch.int64) + 1   # independent
                mism = fast != full
                nb = int(mism.sum())
                if nb:
                    g_ = GT[i]
                    valid = g_ != 0
                    gm_valid += int((mism & valid).sum())
                    if bad_first is None:
                        p = int(torch.nonzero(mism)[0])
                        row = tf[p]
                        biased = (tf + bfull)[p]
                        t2 = torch.topk(biased, 2).values
                        bad_first = {
                            "decision": g["decision"], "round": g["round"],
                            "class": g["class"], "candidate_bias": c,
                            "base_vector": base, "joint_vector": vp,
                            "stem": stems[i], "flat_pixel_index": p,
                            "gt_official": int(g_[p]),
                            "gt_is_ignore": bool(g_[p] == 0),
                            "score_vector_8d_raw": [float(x) for x in row],
                            "score_plus_bias_8d": [float(x) for x in biased],
                            "fast_winner_official": int(fast[p]),
                            "full_winner_official": int(full[p]),
                            "other_value": float(ov[p]),
                            "other_class_official": int(oa[p]) + 1,
                            "target_biased_score": float(sc[p]),
                            "comparison_is_exact_tie": bool(float(sc[p]) == float(ov[p])),
                            "biased_top2_gap": float(t2[0] - t2[1]),
                        }
                    del g_, valid
                gm_all += nb
                comparisons += 1
                del sc, win, fast, full, mism, tf
            if gm_all:
                break
        del ref
        total_all += gm_all
        total_valid += gm_valid
        per_group.append({"decision": g["decision"], "round": g["round"],
                          "class": g["class"], "base": base, "span": g["span"],
                          "cands": g["cands"],
                          "search_accepted": g["search_accepted"],
                          "mismatch_all_px": gm_all,
                          "mismatch_valid_px": gm_valid})
        if gm_all:
            break

    verdict = "PASS" if total_all == 0 else "FAIL"
    # three SEPARATE quantities; never label the loop accumulator as unique pixels
    if args.max_images:
        n_spatial = sum(int(t.reshape(-1, 8).shape[0]) for t in S[:1]) * n
        n_valid = sum(int((g != 0).sum()) for g in GT)
    else:
        n_spatial = SPATIAL_PX
        n_valid = VALID_PX
    print(f"IMAGES_VERIFIED {n}", flush=True)
    print(f"TRIAL_GROUPS {len(per_group)}", flush=True)
    print(f"CANDIDATE_BIASES_PER_GROUP {GRID}", flush=True)
    print(f"SPATIAL_PIXELS {n_spatial}", flush=True)
    print(f"VALID_SPATIAL_PIXELS {n_valid}", flush=True)
    print(f"CANDIDATE_PIXEL_COMPARISONS "
          f"{n_spatial * sum(len(g['cands']) for g in per_group)}", flush=True)
    print(f"TOTAL_ALL_PIXEL_MISMATCH {total_all}", flush=True)
    print(f"TOTAL_VALID_PIXEL_MISMATCH {total_valid}", flush=True)
    print(f"VERDICT {verdict}", flush=True)
    if bad_first:
        print("FIRST_DISAGREEMENT_RAW (no interpretation):", flush=True)
        print(json.dumps(bad_first, indent=2), flush=True)

    res = {
        "schema": "M2F_56K_JOINT_FASTPATH_PIXELWISE_V3_TRACEREPLAY",
        "gate": "TOTAL_ALL_PIXEL_MISMATCH == 0 (implies M_valid == 0)",
        "scope": "the REAL coordinate-descent trials, replayed from the search's "
                 "own 28-decision trace rather than reconstructed",
        "state_machine_assertions": [
            "28 decision records",
            "accepted move changes exactly coordinate k",
            "rejected move leaves the vector unchanged",
            "replayed final vector == joint_local_best_vector",
        ],
        "independence": "full = independent argmax(S + v'); other_v/other_a reused "
                        "on the fast side only",
        "images_verified": n,
        "trial_groups": len(per_group),
        "candidate_biases_per_group": GRID,
        "spatial_pixels": n_spatial,
        "valid_spatial_pixels": n_valid,
        "candidate_pixel_comparisons": n_spatial * sum(
            len(g["cands"]) for g in per_group),
        "total_all_pixel_mismatch": total_all,
        "total_valid_pixel_mismatch": total_valid,
        "verdict": verdict,
        "per_group": per_group,
        "first_disagreement_raw": bad_first,
    }
    (OUT / args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False,
                                           allow_nan=False))
    print("JOINT_PIXELWISE_DONE", flush=True)


if __name__ == "__main__":
    main()
