"""C2 acceptance gates (v2). Runs AFTER the sweep; never touches the sweep.

TRANSITION SEMANTICS -- the correction that matters most
  The four events are baseline->biased TRANSITIONS, not post-bias totals:

      base_c  = (p0 == c)          p0 = prediction at bias 0
      new_c   = (pb == c)          pb = prediction at this bias
      gt_c    = (GT == c)

      corrected_FN =  gt_c & ~base_c & new_c      (was FN, now c)
      created_FP   = ~gt_c & ~base_c & new_c      (was neither-c-right nor c, now c)
      removed_FP   = ~gt_c &  base_c & ~new_c     (was FP, no longer c)
      destroyed_TP =  gt_c &  base_c & ~new_c     (was correct c, no longer c)

  All four are EXACTLY 0 at bias 0 (since pb == p0), and each must be
  non-decreasing as the bias moves in its own direction. Counting post-bias
  totals instead would pass while measuring the wrong thing.

GATES
  G0 semantics   hand-computed micro-case for the four transition counts.
  G1 baseline    two independent paths; valid count, 8x8 confusion, 8 class IoUs
                 and mIoU must all agree; mIoU also matches the frozen 0.8173524135.
  G2 exactness   per class: near-zero, low-rate threshold, medium, and the ACTUAL
                 C2 best bias added unconditionally whether positive or negative.
                 Fast path must equal full 8-channel FP32 argmax, 0 mismatches.
  G3 monotonic   true transition counts on the exact path; +b grows corrected_FN
                 and created_FP; -|b| grows removed_FP and destroyed_TP.
  G4a pred map   force each channel +50 -> exactly that official id, 100%.
  G4b GT map     raw GT label -> evaluator index, validated on the OTHER axis that
                 G4a cannot reach: row0 of the confusion matrix MUST be 0 (it is
                 accumulated over valid pixels only), total_raw - ignore == valid,
                 and per-class raw GT counts == confusion ROW SUMS exactly.
  G5 re-score    exact quantities must match the C2 report. Histogram-approximated
                 counts are compared as TRANSITIONS of the matching sign and their
                 absolute/relative error is recorded, never used as pass/fail.

No training, no packaging, no writes outside this outdir.
"""
import json
from pathlib import Path

import numpy as np
import torch

SHM = Path("/dev/shm/tta8_56k_v2")
ARRS = Path("/dev/shm/boundary_56k_v2")
ROOT = Path("/home/jhy/rare_diag_20260915/m2f_gpu_20260915")
C2 = ROOT / "c2_56k" / "C2_RESULT.json"
OUT = ROOT / "c2_56k"
NAMES = ["Background", "Building", "Road", "Water", "Barren", "Vegetation",
         "Agricultural", "Vehicle"]
EXPECTED_ZERO = 0.8173524135
torch.set_num_threads(32)

gates = {}


def ok(g, detail):
    gates[g] = {"pass": True, "detail": detail}
    print(f"GATE {g} PASS: {detail}", flush=True)


def fail(g, detail):
    gates[g] = {"pass": False, "detail": detail}
    print(f"GATE {g} FAIL: {detail}", flush=True)


def transitions(gt_c, base_c, new_c):
    """The four baseline->biased transition counts. Each is 0 when pb == p0."""
    return {
        "corrected_FN": int((gt_c & ~base_c & new_c).sum()),
        "created_FP": int((~gt_c & ~base_c & new_c).sum()),
        "removed_FP": int((~gt_c & base_c & ~new_c).sum()),
        "destroyed_TP": int((gt_c & base_c & ~new_c).sum()),
    }


def g0_semantics():
    """Hand-computed micro cases for the four baseline->biased transitions.

    Fixed fixture (length 6), class of interest c = 2:
        pix:    0    1    2    3    4    5
        gt      1    1    2    2    3    3
        base    1    2    2    1    3    2
        => base_c = [F, T, T, F, F, T]

    Each case changes only the `new` vector and is counted by hand pixel by pixel.
    The point of this gate is that it fails loudly if `transitions()` ever drifts
    back to counting post-bias totals instead of transitions.
    """
    gt = [1, 1, 2, 2, 3, 3]
    base = [1, 2, 2, 1, 3, 2]
    gt_c = torch.tensor([g == 2 for g in gt])
    base_c = torch.tensor([b == 2 for b in base])
    cases = []

    def run(new, want, label, why):
        got = transitions(gt_c, base_c, torch.tensor([x == 2 for x in new]))
        assert got == want, f"G0 {label}: got {got} want {want}"
        cases.append({"case": label, "new": new, "got": got, "why": why})

    run([1, 1, 2, 1, 2, 3],
        {"corrected_FN": 0, "created_FP": 1, "removed_FP": 2, "destroyed_TP": 0},
        "case1",
        "pix4 gt!=c,base!=c,new=c -> created_FP; "
        "pix1 gt!=c,base=c,new!=c -> removed_FP; "
        "pix5 gt!=c,base=c,new!=c -> removed_FP; pix3 gt=c,base!=c,new!=c -> nothing")

    run([1, 2, 2, 2, 3, 3],
        {"corrected_FN": 1, "created_FP": 0, "removed_FP": 1, "destroyed_TP": 0},
        "case2",
        "pix3 gt=c,base!=c,new=c -> corrected_FN; "
        "pix5 gt!=c,base=c,new!=c -> removed_FP; "
        "pix1 gt!=c,base=c,new=c -> already c at baseline, no transition")

    run([2, 2, 2, 1, 2, 2],
        {"corrected_FN": 0, "created_FP": 2, "removed_FP": 0, "destroyed_TP": 0},
        "case3",
        "pix0 gt!=c,base!=c,new=c -> created_FP; "
        "pix4 gt!=c,base!=c,new=c -> created_FP; "
        "pix1 gt!=c,base=c,new=c -> no transition; pix3 gt=c,base!=c,new!=c -> nothing")

    run([1, 2, 1, 1, 3, 2],
        {"corrected_FN": 0, "created_FP": 0, "removed_FP": 0, "destroyed_TP": 1},
        "case4",
        "pix2 gt=c,base=c,new!=c -> destroyed_TP; "
        "pix1 gt!=c,base=c,new=c -> no transition")

    run([1, 2, 2, 1, 3, 2],
        {"corrected_FN": 0, "created_FP": 0, "removed_FP": 0, "destroyed_TP": 0},
        "case5_identity",
        "new == base -> all four transitions must be exactly 0")

    ok("G0_transition_semantics",
       f"{len(cases)} hand-computed micro cases match; each of corrected_FN, "
       f"created_FP, removed_FP, destroyed_TP exercised at least once, plus an "
       f"identity case proving all four are 0 when the prediction does not move")
    gates["G0_detail"] = cases


def main():
    g0_semantics()

    stems = sorted(p.stem for p in SHM.glob("*.npy")
                   if not p.stem.startswith("gt_") and p.stem != "_DONE")
    n = len(stems)
    S = [torch.from_numpy(np.load(SHM / f"{s}.npy")) for s in stems]
    GTraw = [np.load(SHM / f"gt_{s}.npy") for s in stems]
    GT = [torch.from_numpy(g).long().reshape(-1) for g in GTraw]
    H, W = S[0].shape[:2]
    print(f"loaded {n} images {H}x{W}", flush=True)

    def conf_and_miou(pred_list, idx=None):
        idx = range(n) if idx is None else idx
        tot = np.zeros((9, 9), dtype=np.int64)
        for i in idx:
            g = GT[i]
            p = pred_list[i]
            m = g != 0
            tot += np.bincount((g[m] * 9 + p[m]).numpy(), minlength=81).reshape(9, 9)
        sub = tot[1:, 1:].astype(np.float64)
        inter = np.diag(sub)
        union = sub.sum(1) + sub.sum(0) - inter
        pp = union > 0
        iou = np.where(pp, inter / np.maximum(union, 1e-12), np.nan)
        return tot, float(np.nanmean(iou[pp])), iou

    def full_pred(vec):
        b = torch.tensor(vec, dtype=torch.float32).view(1, 1, 8)
        return [((t + b).argmax(-1).reshape(-1) + 1).to(torch.uint8) for t in S]

    # ================= G1 =================
    print("G1 baseline", flush=True)
    nonfinite = 0
    for i in range(n):
        if not bool(torch.isfinite(S[i]).all()):
            nonfinite += 1
    print(f"   nonfinite score maps: {nonfinite}", flush=True)

    p0 = full_pred([0.0] * 8)
    cm_a, base, base_iou = conf_and_miou(p0)
    valid_a = int(cm_a.sum())

    cm_b = np.zeros((9, 9), dtype=np.int64)
    valid_b = 0
    total_raw = 0
    for i in range(n):
        g = GTraw[i].reshape(-1).astype(np.int64)
        sb = np.load(SHM / f"{stems[i]}.npy").reshape(-1, 8)
        pb = sb.argmax(1).astype(np.int64) + 1
        m = g != 0
        valid_b += int(m.sum())
        total_raw += int(g.size)
        cm_b += np.bincount(g[m] * 9 + pb[m], minlength=81).reshape(9, 9)
        del sb, pb
    sub_a = cm_a[1:, 1:].astype(np.float64)
    ia = np.diag(sub_a)
    ua = sub_a.sum(1) + sub_a.sum(0) - ia
    pa = ua > 0
    iou_a = np.where(pa, ia / np.maximum(ua, 1e-12), np.nan)
    sub_b = cm_b[1:, 1:].astype(np.float64)
    ib = np.diag(sub_b)
    ub = sub_b.sum(1) + sub_b.sum(0) - ib
    pb_ = ub > 0
    iou_b = np.where(pb_, ib / np.maximum(ub, 1e-12), np.nan)

    g1 = {
        "valid_px_pathA": valid_a, "valid_px_pathB": valid_b,
        "valid_px_equal": valid_a == valid_b,
        "confusion_exact_equal": bool((cm_a == cm_b).all()),
        "max_abs_confusion_diff": int(np.abs(cm_a - cm_b).max()),
        "per_class_iou_max_abs_diff": float(np.nanmax(np.abs(iou_a - iou_b))),
        "miou_pathA": base, "miou_pathB": float(np.nanmean(iou_b[pb_])),
        "miou_expected": EXPECTED_ZERO,
        "miou_abs_diff_vs_expected": abs(base - EXPECTED_ZERO),
        "per_class_iou": {NAMES[i]: float(iou_a[i]) for i in range(8)},
        "total_raw_px": total_raw,
        "nonfinite_score_maps": nonfinite,
        "all_scores_finite": nonfinite == 0,
    }
    g1["pass"] = (g1["valid_px_equal"] and g1["confusion_exact_equal"]
                  and g1["per_class_iou_max_abs_diff"] == 0.0
                  and g1["miou_abs_diff_vs_expected"] < 5e-10
                  and g1["all_scores_finite"])
    print(f"   valid A={valid_a} B={valid_b} | CM equal={g1['confusion_exact_equal']} "
          f"| IoU maxdiff={g1['per_class_iou_max_abs_diff']:.3e} "
          f"| mIoU {base:.10f} diff={g1['miou_abs_diff_vs_expected']:.3e}", flush=True)
    if g1["pass"]:
        ok("G1_baseline", json.dumps({k: v for k, v in g1.items() if k != "per_class_iou"}))
    else:
        fail("G1_baseline", json.dumps({k: v for k, v in g1.items() if k != "per_class_iou"}))
    gates["G1_detail"] = g1

    # ================= G4a =================
    print("G4a prediction mapping", flush=True)
    # Do NOT assume the semantic score lies in [0,1]: it is a sum over 100 query
    # slots of class-probability x mask-probability and can exceed 1. Derive the
    # force bias from the measured range so the target channel provably dominates
    # whatever the other seven hold.
    s_lo, s_hi = np.inf, -np.inf
    for i in range(n):
        s_lo = min(s_lo, float(S[i].min()))
        s_hi = max(s_hi, float(S[i].max()))
    force_bias = float(s_hi - s_lo) + 1.0
    print(f"   measured score range [{s_lo:.10g}, {s_hi:.10g}] "
          f"-> force_bias = {force_bias:.10f}", flush=True)
    assert s_hi - s_lo < 1e6, "score range implausibly large"

    g4a, g4a_ok = [], True
    for k in range(8):
        vec = [0.0] * 8
        vec[k] = force_bias
        p = full_pred(vec)
        hits = tot = 0
        for i in range(n):
            m = GT[i] != 0
            hits += int((p[i][m] == (k + 1)).sum())
            tot += int(m.sum())
        frac = hits / max(tot, 1)
        g4a.append({"channel": k, "expected_official": k + 1, "name": NAMES[k],
                    "frac_correct": frac, "n_valid": tot,
                    "force_bias_used": force_bias})
        print(f"   ch{k} -> official {k+1} ({NAMES[k]:13s}) frac={frac:.10f}", flush=True)
        if frac != 1.0:
            g4a_ok = False
        del p
    if g4a_ok:
        ok("G4a_pred_mapping",
           f"all 8 channels map 100% to their official id "
           f"(force_bias={force_bias:.6f} derived from measured range "
           f"[{s_lo:.6g}, {s_hi:.6g}], not an assumed [0,1])")
    else:
        fail("G4a_pred_mapping", json.dumps([d for d in g4a if d["frac_correct"] != 1.0]))
    gates["G4a_detail"] = {"score_range": {"min": s_lo, "max": s_hi},
                           "force_bias": force_bias, "per_channel": g4a}

    # ================= G4b (ignore rule corrected) =================
    print("G4b GT mapping", flush=True)
    uniq = set()
    ignore_cnt = 0
    gt_counts = np.zeros(9, dtype=np.int64)
    for g in GTraw:
        a = g.reshape(-1)
        uniq.update(int(x) for x in np.unique(a))
        ignore_cnt += int((a == 0).sum())
        gt_counts += np.bincount(a[a != 0], minlength=9)[:9]
    row_sums = cm_a.sum(1)
    # the docstring declares that label 0 appears and is ignore-only; make the
    # implementation enforce that declaration rather than assume it
    assert 0 in uniq, "no ignore label in the raw masks; G4b assumptions broken"
    assert ignore_cnt > 0, "ignore pixel count is 0; G4b assumptions broken"
    g4b = {
        "unique_raw_gt_labels": sorted(uniq),
        "labels_subset_of_0_to_8": set(uniq).issubset(set(range(9))),
        "total_raw_px": total_raw,
        "ignore_px_raw": ignore_cnt,
        "valid_px": valid_a,
        "total_minus_ignore_equals_valid": (total_raw - ignore_cnt) == valid_a,
        "cm_row0": int(row_sums[0]),
        "cm_row0_is_zero": int(row_sums[0]) == 0,
        "cm_total_equals_valid": int(cm_a.sum()) == valid_a,
        "gt_counts_from_raw": {NAMES[j]: int(gt_counts[j + 1]) for j in range(8)},
        "cm_row_sums": {NAMES[j]: int(row_sums[j + 1]) for j in range(8)},
        "row_sums_equal_raw_counts": bool((gt_counts[1:] == row_sums[1:]).all()),
    }
    g4b["pass"] = (g4b["labels_subset_of_0_to_8"]
                   and g4b["cm_row0_is_zero"]
                   and g4b["total_minus_ignore_equals_valid"]
                   and g4b["cm_total_equals_valid"]
                   and g4b["row_sums_equal_raw_counts"])
    print(f"   labels={sorted(uniq)} | row0={row_sums[0]} (must be 0) "
          f"| total {total_raw} - ignore {ignore_cnt} = {total_raw-ignore_cnt} "
          f"vs valid {valid_a}", flush=True)
    for j in range(8):
        flag = "" if gt_counts[j + 1] == row_sums[j + 1] else "  <== MISMATCH"
        print(f"     {NAMES[j]:13s} raw={gt_counts[j+1]:>12d} cm_rowsum={row_sums[j+1]:>12d}{flag}",
              flush=True)
    if g4b["pass"]:
        ok("G4b_gt_mapping", json.dumps({k: v for k, v in g4b.items()
                                         if not k.startswith("gt_counts")
                                         and not k.startswith("cm_row")}))
    else:
        fail("G4b_gt_mapping", json.dumps({k: v for k, v in g4b.items()
                                           if not k.startswith("gt_counts")
                                           and not k.startswith("cm_row")}))
    gates["G4b_detail"] = g4b

    # ================= G2 =================
    print("G2 exactness", flush=True)
    best_bias = {}
    if C2.exists():
        r = json.loads(C2.read_text())
        for name, d in r.get("per_class_best", {}).items():
            best_bias[name] = float(d["bias"])
    marg = {}
    for k in range(1, 8):
        a = np.load(ARRS / f"tau_FN_{NAMES[k]}.npy")
        b_ = np.load(ARRS / f"tau_FP_{NAMES[k]}.npy")
        marg[NAMES[k]] = (float(np.quantile(a, 0.01)), float(np.quantile(b_, 0.99)))
    g2, total_mism = [], 0
    for k in range(1, 8):
        nm = NAMES[k]
        lo_fn, hi_fp = marg[nm]
        test = set()
        for x in (1e-4, lo_fn, 0.1, abs(best_bias.get(nm, 0.3))):
            test.add(float(np.nextafter(x, np.inf)))
            test.add(float(np.nextafter(-x, -np.inf)))
        # the ACTUAL C2 best bias, unconditionally, whatever its sign
        bb_best = best_bias.get(nm)
        if bb_best is not None:
            test.add(float(bb_best))
        test = sorted(test)
        keep = [j for j in range(8) if j != k]
        kt = torch.tensor(keep, dtype=torch.int64)
        ov = torch.empty(n, H * W, dtype=torch.float32)
        oa = torch.empty(n, H * W, dtype=torch.uint8)
        for i, t in enumerate(S):
            rr = t.reshape(-1, 8)[:, keep]
            a1 = rr.argmax(1)
            ov[i] = rr.gather(1, a1.unsqueeze(1)).squeeze(1)
            oa[i] = kt[a1]
            del rr, a1
        for b in test:
            mism = 0
            vec = [0.0] * 8
            vec[k] = b
            bb = torch.tensor(vec, dtype=torch.float32).view(1, 8)
            for i, t in enumerate(S):
                sc = t.reshape(-1, 8)[:, k] + float(b)
                win = (sc > ov[i]) | ((sc == ov[i]) & (k < oa[i]))
                fast = torch.where(win, torch.full_like(oa[i], k + 1, dtype=torch.int64),
                                   oa[i].to(torch.int64) + 1)
                full = (t.reshape(-1, 8) + bb).argmax(1).to(torch.int64) + 1
                mism += int((fast != full).sum())
                del sc, win, fast, full
            total_mism += mism
            g2.append({"class": nm, "bias": b, "mismatch": mism,
                       "is_c2_best": (bb_best is not None and b == float(bb_best))})
            if mism:
                print(f"   MISMATCH {nm} b={b:+.6f} -> {mism}", flush=True)
        del ov, oa
        print(f"   {nm:13s} {len(test)} biases (incl C2 best {bb_best}) "
              f"cumulative mismatch {total_mism}", flush=True)
    if total_mism == 0:
        ok("G2_fastpath_exact",
           f"{len(g2)} (class,bias) pairs incl. each class's actual C2 best, "
           f"0 mismatched pixels")
    else:
        fail("G2_fastpath_exact", f"{total_mism} mismatched; "
                                  f"first bad {[x for x in g2 if x['mismatch']][:5]}")
    gates["G2_detail"] = g2

    # ================= G3: true transitions =================
    print("G3 monotonicity (true transitions, exact path)", flush=True)
    pos_b = [0.0, 0.05, 0.2, 0.5, 0.8]
    neg_b = [0.0, -0.05, -0.2, -0.5, -0.8]
    g3, g3_ok = {}, True
    for k in range(1, 8):
        gt_c = [GT[i] == (k + 1) for i in range(n)]
        base_c = [p0[i].to(torch.int64) == (k + 1) for i in range(n)]
        rows = []
        for b in pos_b + neg_b[1:]:
            vec = [0.0] * 8
            vec[k] = b
            p = full_pred(vec)
            agg = {"corrected_FN": 0, "created_FP": 0,
                   "removed_FP": 0, "destroyed_TP": 0}
            for i in range(n):
                t = transitions(gt_c[i], base_c[i], p[i].to(torch.int64) == (k + 1))
                for kk in agg:
                    agg[kk] += t[kk]
            del p
            rows.append({"bias": b, **agg})
        pos_rows = rows[:len(pos_b)]
        # negative leg must START at b=0, not at the last positive row
        neg_rows = [rows[0]] + rows[len(pos_b):]
        assert [r["bias"] for r in neg_rows] == neg_b, [r["bias"] for r in neg_rows]
        zero_row = pos_rows[0]
        zero_all_zero = all(zero_row[kk] == 0 for kk in
                            ("corrected_FN", "created_FP", "removed_FP", "destroyed_TP"))
        m_fn = all(pos_rows[i]["corrected_FN"] >= pos_rows[i - 1]["corrected_FN"]
                   for i in range(1, len(pos_rows)))
        m_fp = all(pos_rows[i]["created_FP"] >= pos_rows[i - 1]["created_FP"]
                   for i in range(1, len(pos_rows)))
        m_rf = all(neg_rows[i]["removed_FP"] >= neg_rows[i - 1]["removed_FP"]
                   for i in range(1, len(neg_rows)))
        m_dt = all(neg_rows[i]["destroyed_TP"] >= neg_rows[i - 1]["destroyed_TP"]
                   for i in range(1, len(neg_rows)))
        g3[NAMES[k]] = {"rows": rows, "zero_row_all_zero": zero_all_zero,
                        "pos_corrected_FN_monotone": m_fn,
                        "pos_created_FP_monotone": m_fp,
                        "neg_removed_FP_monotone": m_rf,
                        "neg_destroyed_TP_monotone": m_dt}
        print(f"   {NAMES[k]:13s} zero_all_zero={zero_all_zero} "
              f"posFN={m_fn} posFP={m_fp} negFP={m_rf} negTP={m_dt}", flush=True)
        for r in rows:
            print(f"      b={r['bias']:+.2f} corrFN={r['corrected_FN']:>10d} "
                  f"creatFP={r['created_FP']:>10d} remFP={r['removed_FP']:>10d} "
                  f"destTP={r['destroyed_TP']:>10d}", flush=True)
        g3_ok &= (zero_all_zero and m_fn and m_fp and m_rf and m_dt)
    if g3_ok:
        ok("G3_monotonic", "true transitions: zero at b=0, monotone in both "
                           "directions, all 7 classes")
    else:
        bad3 = {k: v for k, v in g3.items()
                if not (v["zero_row_all_zero"] and v["pos_corrected_FN_monotone"]
                        and v["pos_created_FP_monotone"] and v["neg_removed_FP_monotone"]
                        and v["neg_destroyed_TP_monotone"])}
        fail("G3_monotonic", json.dumps(bad3))
    gates["G3_detail"] = g3

    # ================= G5: exact re-score + transition-matched histogram error ===
    print("G5 re-score", flush=True)
    if not C2.exists():
        fail("G5_rescore", f"{C2} missing")
    else:
        res = json.loads(C2.read_text())
        picks = res.get("top20", [])[:5] + list(res.get("per_class_best", {}).values())
        seen, g5 = set(), []
        for r in picks:
            key = (r["class"], round(float(r["bias"]), 9))
            if key in seen:
                continue
            seen.add(key)
            k = NAMES.index(r["class"])
            vec = [0.0] * 8
            vec[k] = float(r["bias"])
            p = full_pred(vec)
            cm, m, iou = conf_and_miou(p)
            agg = {"corrected_FN": 0, "created_FP": 0, "removed_FP": 0, "destroyed_TP": 0}
            for i in range(n):
                t = transitions(GT[i] == (k + 1), p0[i].to(torch.int64) == (k + 1),
                                p[i].to(torch.int64) == (k + 1))
                for kk in agg:
                    agg[kk] += t[kk]
            rep = r.get("events", {})
            b = float(r["bias"])
            # The HISTOGRAM-approximated quantities are created_FP (from the TN
            # histogram) and destroyed_TP (from the TP histogram). corrected_FN
            # and removed_FP come from the EXACT sorted FN/FP arrays and are only
            # used here as a sanity cross-check, not as the error measurement.
            if b > 0:
                hist_key, exact_key = "created_FP", "created_FP"
                sanity_key = "corrected_FN"
            else:
                hist_key, exact_key = "destroyed_TP", "destroyed_TP"
                sanity_key = "removed_FP"
            ap = rep.get(hist_key, 0)
            ex = agg[exact_key]
            san_ap = rep.get(sanity_key, 0)
            san_ex = agg[sanity_key]
            g5.append({
                "class": r["class"], "bias": b,
                "reported_miou": r["miou"], "exact_miou": m,
                "miou_match": abs(m - r["miou"]) < 1e-12,
                "reported_gain_pp": r.get("gain_pp"),
                "exact_gain_pp": (m - base) * 100,
                "exact_per_class_iou": {NAMES[j]: float(iou[j]) for j in range(8)},
                "exact_transitions": agg,
                "reported_events": rep,
                "hist_compared_key": hist_key,
                "hist_approx": ap, "exact": ex,
                "hist_abs_err": ap - ex,
                "hist_rel_err": (ap - ex) / max(ex, 1),
                "sanity_key_from_exact_array": sanity_key,
                "sanity_approx": san_ap, "sanity_exact": san_ex,
                "sanity_abs_err": san_ap - san_ex,
            })
            print(f"   {r['class']:13s} b={b:+.6f} rep={r['miou']:.10f} exact={m:.10f} "
                  f"match={g5[-1]['miou_match']} | HIST {hist_key}: approx={ap} exact={ex} "
                  f"err={ap-ex} ({g5[-1]['hist_rel_err']*100:+.2f}%) | "
                  f"SANITY {sanity_key}: approx={san_ap} exact={san_ex} "
                  f"err={san_ap-san_ex}", flush=True)
        all_match = all(x["miou_match"] for x in g5)
        if all_match:
            ok("G5_rescore",
               f"{len(g5)} candidates re-scored exactly; exact mIoU all match. "
               f"Histogram transition error recorded, not used for pass/fail.")
        else:
            fail("G5_rescore", json.dumps([x for x in g5 if not x["miou_match"]]))
        gates["G5_detail"] = g5

    (OUT / "C2_ACCEPTANCE.json").write_text(json.dumps(
        {"gates": gates,
         "all_pass": all(v.get("pass") for v in gates.values()
                         if isinstance(v, dict) and "pass" in v)},
        indent=2, ensure_ascii=False, allow_nan=False))
    print("ACCEPTANCE_DONE", flush=True)
    for k, v in gates.items():
        if isinstance(v, dict) and "pass" in v:
            print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
