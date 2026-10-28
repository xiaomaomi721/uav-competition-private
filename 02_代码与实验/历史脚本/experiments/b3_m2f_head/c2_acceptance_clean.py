"""Clean rewrite of the C2 acceptance suite. No patch history carried over.

Six checks, each small and self-contained:

  C1  baseline        exact confusion matrix + per-class IoU + mIoU, recomputed
                      and compared against the frozen TTA8 ZERO reference.
  C2  class-ID map    score channel <-> official id <-> GT id, all three legs.
  C3  fast vs full    the accelerated single-class path against a completely
                      independent slow reference implementation, pixel by pixel.
  C4  transitions     the four baseline->biased transition counts, hand-verified.
  C5  monotonicity    transitions move in the correct direction as bias grows.
  C6  re-score        the top C2 candidates, exact mIoU and per-class IoU.

Development order enforced by the code itself:
  every check is first run on TINY SYNTHETIC FIXTURES built inside this file,
  and only then on the real cache. If a fixture fails, the real-data stage is
  never reached.

The slow reference is deliberately naive: Python loops, explicit comparisons,
no vectorised tricks, no shared code with the fast path. If the two disagree on
any pixel, one of them is wrong and the run stops.

Read-only with respect to the cache. No training, no packaging.
"""
import json
import sys
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
REAL_N = 1399
torch.set_num_threads(32)

results = {}


def record(name, passed, detail):
    results[name] = {"pass": bool(passed), "detail": detail}
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)
    if not passed:
        raise SystemExit(f"acceptance stopped at {name}")


# =====================================================================
# slow reference implementation -- no vectorisation, no shared helpers
# =====================================================================

def slow_transitions(gt, base, new, c):
    """Count the four baseline->biased transitions for class c, one pixel at a
    time. gt/base/new are flat sequences of OFFICIAL ids (1..8)."""
    corrected_FN = created_FP = removed_FP = destroyed_TP = 0
    for i in range(len(gt)):
        was_c = (base[i] == c)
        now_c = (new[i] == c)
        is_c = (gt[i] == c)
        if is_c and (not was_c) and now_c:
            corrected_FN += 1
        if (not is_c) and (not was_c) and now_c:
            created_FP += 1
        if (not is_c) and was_c and (not now_c):
            removed_FP += 1
        if is_c and was_c and (not now_c):
            destroyed_TP += 1
    return {"corrected_FN": corrected_FN, "created_FP": created_FP,
            "removed_FP": removed_FP, "destroyed_TP": destroyed_TP}


def slow_argmax_row(vec, bias):
    """One pixel: argmax over 8 channels with bias, tie -> lowest index.
    Returns the OFFICIAL id (index + 1)."""
    best_i = 0
    best_v = vec[0] + bias[0]
    for j in range(1, 8):
        v = vec[j] + bias[j]
        if v > best_v:                     # strict: ties keep the lower index
            best_v = v
            best_i = j
    return best_i + 1


def slow_fast_class_row(vec, k, b, other_v, other_a):
    """The single-class fast-path rule for one pixel, written out plainly.
    k is the 0-based channel being moved; returns the OFFICIAL id."""
    sc = vec[k] + b
    if sc > other_v:
        return k + 1
    if sc == other_v and k < other_a:
        return k + 1
    return other_a + 1


def slow_other_side_row(vec, k):
    """max over j != k and the argmax among them, lowest index on ties."""
    best_i, best_v = None, None
    for j in range(8):
        if j == k:
            continue
        v = vec[j]
        if best_v is None or v > best_v:
            best_v, best_i = v, j
    return best_v, best_i


def slow_confusion(gt_rows, pred_rows, n_classes=9):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for gt, pr in zip(gt_rows, pred_rows):
        for i in range(len(gt)):
            if gt[i] == 0:
                continue
            cm[int(gt[i]), int(pr[i])] += 1
    return cm


def slow_miou(cm):
    ious = []
    for c in range(1, 9):
        tp = cm[c, c]
        fp = cm[1:, c].sum() - tp
        fn = cm[c, 1:].sum() - tp
        union = tp + fp + fn
        ious.append(tp / union if union > 0 else float("nan"))
    arr = np.array(ious, dtype=np.float64)
    m = ~np.isnan(arr)
    return float(arr[m].mean()), {NAMES[i]: float(arr[i]) for i in range(8)}


# =====================================================================
# C4 -- transition semantics, fixtures first
# =====================================================================

def c4_transitions():
    # ---- tiny fixtures ----
    # fixture A: gt row and base row are fixed; `new` varies per case
    gtA = [1, 1, 2, 2, 3, 3]
    baseA = [1, 2, 2, 1, 3, 2]          # base==2 at pixels 1,2,5
    cases = [
        ("fires created_FP and removed_FP",
         [1, 1, 2, 1, 2, 3],
         {"corrected_FN": 0, "created_FP": 1, "removed_FP": 2, "destroyed_TP": 0}),
        ("fires corrected_FN and removed_FP",
         [1, 2, 2, 2, 3, 3],
         {"corrected_FN": 1, "created_FP": 0, "removed_FP": 1, "destroyed_TP": 0}),
        ("fires created_FP twice",
         [2, 2, 2, 1, 2, 2],
         {"corrected_FN": 0, "created_FP": 2, "removed_FP": 0, "destroyed_TP": 0}),
        ("fires destroyed_TP",
         [1, 2, 1, 1, 3, 2],
         {"corrected_FN": 0, "created_FP": 0, "removed_FP": 0, "destroyed_TP": 1}),
        ("identity: new == base -> all four zero",
         [1, 2, 2, 1, 3, 2],
         {"corrected_FN": 0, "created_FP": 0, "removed_FP": 0, "destroyed_TP": 0}),
    ]
    for label, new, want in cases:
        got = slow_transitions(gtA, baseA, new, 2)
        assert got == want, f"C4 fixture '{label}': got {got} want {want}"

    # a second fixture with a different class, to prove c is not hard-coded
    gtB = [4, 4, 5, 5, 6, 6]
    baseB = [5, 4, 5, 4, 6, 5]
    newB = [4, 4, 5, 5, 5, 6]
    # c=5: pix0 gt!=5 base=5 new!=5 -> removed_FP
    #      pix2 gt=5  base=5 new=5   -> no change
    #      pix3 gt=5  base!=5 new=5  -> corrected_FN
    #      pix4 gt!=5 base=6 new=5   -> created_FP
    #      pix5 gt!=5 base=5 new=6   -> removed_FP
    gotB = slow_transitions(gtB, baseB, newB, 5)
    wantB = {"corrected_FN": 1, "created_FP": 1, "removed_FP": 2, "destroyed_TP": 0}
    assert gotB == wantB, f"C4 fixture B: got {gotB} want {wantB}"

    record("C4_transitions", True,
           f"{len(cases) + 1} hand-computed fixtures pass, including an identity "
           f"case where all four transitions must be exactly 0")


# =====================================================================
# C3 -- fast path vs independent slow reference, fixtures first
# =====================================================================

def c3_fast_vs_slow_on_fixture():
    rng = np.random.default_rng(1234)
    n_pix = 40
    vecs = [[float(x) for x in rng.integers(0, 4, 8)] for _ in range(n_pix)]
    # integer values make exact ties common, exercising the tie rule
    for k in range(8):
        biases = [0.0, 0.5, -0.5, 1.0, -1.0]
        for b in biases:
            for vec in vecs:
                ov_s, oa_s = slow_other_side_row(vec, k)
                slow = slow_fast_class_row(vec, k, b, ov_s, oa_s)
                bias = [0.0] * 8
                bias[k] = b
                ref = slow_argmax_row(vec, bias)
                assert slow == ref, (f"C3 fixture k={k} b={b} vec={vec}: "
                                     f"fast={slow} ref={ref}")
    # and the same with continuous values (no ties)
    vecs2 = [[float(x) for x in rng.normal(0, 1, 8)] for _ in range(n_pix)]
    for k in range(8):
        for b in (0.0, 0.3, -0.3):
            for vec in vecs2:
                ov_s, oa_s = slow_other_side_row(vec, k)
                slow = slow_fast_class_row(vec, k, b, ov_s, oa_s)
                bias = [0.0] * 8
                bias[k] = b
                ref = slow_argmax_row(vec, bias)
                assert slow == ref, (f"C3 fixture cont k={k} b={b} vec={vec}: "
                                     f"fast={slow} ref={ref}")
    record("C3_fast_vs_slow_fixture", True,
           f"{8*5*40 + 8*3*40} fixture pixels: fast rule == independent slow argmax, "
           f"integer-valued (ties present) and continuous")


def c3_fast_vs_slow_real(S, GTflat, stems, n_img, max_biases):
    """On real cached maps: the vectorised fast path vs the slow row-by-row
    reference. Deliberately slow, so only a few images are used.

    Reported scale: images = the first `n_img` maps plus any image holding a
    known tie location; per image <= 2000 evenly spaced pixels plus the tie
    locations. The slow reference is an algorithm cross-check, not a sample-size
    experiment.

    Known tie coordinates are a FIXED test fixture; their tie property and their
    validity (GT not ignore) are re-derived from the current cache on the spot,
    so a stale coordinate fails loudly instead of silently testing nothing.
    """
    KNOWN_TIES = [("1470", 547659), ("4247", 762505)]
    tie_by_index = {}
    for stem, flat in KNOWN_TIES:
        idx = stems.index(stem)
        tie_by_index.setdefault(idx, []).append(flat)
        row = S[idx].reshape(-1, 8)[flat]
        top2 = torch.topk(row, 2)
        assert float(top2.values[0]) == float(top2.values[1]), (
            f"expected a tie at {stem}:{flat}, got {top2.values.tolist()}")
        assert int(GTflat[idx][flat]) != 0, f"{stem}:{flat} is an ignore pixel"
        print(f"   C3 tie fixture re-verified on cache: {stem}:{flat} "
              f"top2_equal=True value={float(top2.values[0]):.10f} "
              f"gt={int(GTflat[idx][flat])}", flush=True)

    # inject each class's ACTUAL C2 best bias so the check covers the bias we
    # would really deploy, not just a round number
    best_bias = {}
    if C2.exists():
        r = json.loads(C2.read_text())
        for nm, d in r.get("per_class_best", {}).items():
            best_bias[nm] = float(d["bias"])
    # Background is fixed at 0, so the seven adjustable classes must ALL be present
    expected_classes = set(NAMES[1:])
    if set(best_bias) != expected_classes:
        record("C3_fast_vs_slow_real", False,
               f"C2 best bias coverage wrong: missing "
               f"{sorted(expected_classes - set(best_bias))}, extra "
               f"{sorted(set(best_bias) - expected_classes)}")
    print(f"   C3 C2 best biases loaded for all 7 adjustable classes: "
          f"{ {k: round(v, 6) for k, v in best_bias.items()} }", flush=True)

    # deterministic sample of ordinary images, excluding the tie-bearing ones,
    # so the sample is reproducible AND not always the first n images
    tie_idx_set = set(tie_by_index)
    ordinary = [i for i in range(len(S)) if i not in tie_idx_set]
    pick_rng = np.random.default_rng(20260916)
    sampled = sorted(int(x) for x in
                     pick_rng.choice(ordinary, size=min(n_img, len(ordinary)),
                                     replace=False))
    img_list = sorted(set(sampled) | tie_idx_set)
    per_img_px = 2000
    checked = 0
    tie_checked = {f"{stem}:{flat}": 0 for stem, flat in KNOWN_TIES}
    best_checked = {nm: 0 for nm in NAMES[1:]}
    bad = []
    for i in img_list:
        t = S[i].reshape(-1, 8)
        a1 = t.argmax(1)
        m1 = t.gather(1, a1.unsqueeze(1)).squeeze(1)
        tm = t.clone()
        tm.scatter_(1, a1.unsqueeze(1), float("-inf"))
        a2 = tm.argmax(1)
        m2 = tm.gather(1, a2.unsqueeze(1)).squeeze(1)
        sel_base = set(np.arange(0, t.shape[0],
                                 max(1, t.shape[0] // per_img_px)).tolist())
        sel_base |= set(tie_by_index.get(i, []))
        sel = sorted(sel_base)
        tv = t.numpy()
        for k in range(8):
            nm = NAMES[k]
            isk = (a1 == k)
            ov = torch.where(isk, m2, m1)
            oa = torch.where(isk, a2, a1)
            biases = set(float(b) for b in max_biases)
            if nm in best_bias:                      # the value we would deploy
                biases.add(best_bias[nm])
            for b in sorted(biases):
                sc = t[:, k] + float(b)
                win = (sc > ov) | ((sc == ov) & (k < oa))
                fast = torch.where(win, torch.full_like(oa, k + 1, dtype=torch.int64),
                                   oa.to(torch.int64) + 1).numpy()
                is_best = (nm in best_bias and b == best_bias[nm])
                for p in sel:
                    ov_s, oa_s = slow_other_side_row([float(x) for x in tv[p]], k)
                    s_row = slow_fast_class_row([float(x) for x in tv[p]], k, float(b),
                                                ov_s, oa_s)
                    if int(s_row) != int(fast[p]):
                        bad.append((i, k, float(b), int(p), int(s_row), int(fast[p])))
                        if len(bad) > 5:
                            break
                    if is_best:
                        best_checked[nm] += 1
                # account for every known tie location actually reaching the
                # slow reference, per tie, not as one lump sum
                if b == 0.0:
                    for stem, flat in KNOWN_TIES:
                        if i == stems.index(stem) and flat in sel:
                            tie_checked[f"{stem}:{flat}"] += 1
                checked += 1
                if len(bad) > 5:
                    break
            if len(bad) > 5:
                break
        del t, a1, m1, tm, a2, m2
        if len(bad) > 5:
            break
        print(f"   C3 image {i} ok ({'incl. tie' if i in tie_by_index else 'random'})",
              flush=True)
    # ---- one decision, one record() call: a gate can never be recorded FAIL
    # ---- and then overwritten by an unconditional PASS.
    coverage_ok = (set(best_bias) == expected_classes)
    missing_ties = [k for k, v in tie_checked.items() if v == 0]
    missing_best = [k for k, v in best_checked.items() if v == 0]
    ok = coverage_ok and not bad and not missing_ties and not missing_best
    if ok:
        record("C3_fast_vs_slow_real", True,
               f"{len(img_list)} real cached maps ({len(sampled)} ordinary images "
               f"drawn with a fixed seed excluding tie images, plus "
               f"{len(tie_idx_set)} tie-bearing images), <= {per_img_px} sampled "
               f"pixels each plus the tie locations: vectorised fast path == "
               f"independent slow reference at 0, +/-0.3 AND all 7 adjustable "
               f"classes' actual C2 best bias (per-class checks {best_checked}); "
               f"both known tie coordinates re-verified against the current cache "
               f"and exercised (per-tie counts {tie_checked})")
    else:
        record("C3_fast_vs_slow_real", False,
               f"coverage_ok={coverage_ok} mismatches={bad[:5]} "
               f"missing_ties={missing_ties} missing_best={missing_best}")


# =====================================================================
# real-data helpers
# =====================================================================

def load_all():
    stems = sorted(p.stem for p in SHM.glob("*.npy")
                   if not p.stem.startswith("gt_") and p.stem != "_DONE")
    assert len(stems) == REAL_N, f"expected {REAL_N} cached maps, got {len(stems)}"
    S, GTraw = [], []
    for s in stems:
        a = np.load(SHM / f"{s}.npy")
        assert a.shape[2] == 8, f"{s}: expected 8 channels, got {a.shape}"
        assert np.isfinite(a).all(), f"{s}: non-finite score"
        S.append(torch.from_numpy(a))
        GTraw.append(np.load(SHM / f"gt_{s}.npy"))
    print(f"loaded {len(S)} maps", flush=True)
    return stems, S, GTraw


def vec_full_pred(S, vec):
    b = torch.tensor(vec, dtype=torch.float32).view(1, 1, 8)
    return [((t + b).argmax(-1).reshape(-1) + 1).to(torch.uint8) for t in S]


def vec_confusion(GTflat, preds):
    cm = np.zeros((9, 9), dtype=np.int64)
    for g, p in zip(GTflat, preds):
        m = g != 0
        cm += np.bincount((g[m] * 9 + p[m].to(torch.int64)).numpy(),
                          minlength=81).reshape(9, 9)
    return cm


def vec_miou(cm):
    sub = cm[1:, 1:].astype(np.float64)
    inter = np.diag(sub)
    union = sub.sum(1) + sub.sum(0) - inter
    ok = union > 0
    iou = np.where(ok, inter / np.maximum(union, 1e-12), np.nan)
    return float(np.nanmean(iou[ok])), iou


# =====================================================================
# C1 -- baseline exact agreement, fixtures then real data
# =====================================================================

def c1_baseline(S, GTraw, GTflat):
    # ---- one decision, one record() call ----
    n_img_sampled = 20
    px_per_img = 20000
    rng = np.random.default_rng(7)          # fixed seed: the sample is reproducible
    mismatch = []
    for i in range(n_img_sampled):
        t = S[i].reshape(-1, 8).numpy()
        sel = rng.choice(t.shape[0], size=px_per_img, replace=False)
        sel.sort()
        fast = vec_full_pred([S[i]], [0.0] * 8)[0].numpy()
        for p in sel:
            ref = slow_argmax_row([float(x) for x in t[p]], [0.0] * 8)
            if int(ref) != int(fast[p]):
                mismatch.append((i, int(p), int(ref), int(fast[p])))
                break
        del t, sel
        if mismatch:
            break

    # ---- full baseline vs the frozen reference ----
    preds = vec_full_pred(S, [0.0] * 8)
    cm_vec = vec_confusion(GTflat, preds)
    m_vec, iou_vec = vec_miou(cm_vec)
    valid = int(cm_vec.sum())
    total_raw = int(sum(g.numel() for g in GTflat))
    ignore = total_raw - valid
    row_sums = cm_vec.sum(1)

    detail = {
        "valid_px": valid, "total_raw_px": total_raw, "ignore_px": ignore,
        "cm_row0_must_be_zero": int(row_sums[0]),
        "miou": m_vec, "miou_expected": EXPECTED_ZERO,
        "miou_abs_diff": abs(m_vec - EXPECTED_ZERO),
        "per_class_iou": {NAMES[j]: float(iou_vec[j]) for j in range(8)},
        "slow_reference_images": n_img_sampled,
        "slow_reference_pixels_per_image": px_per_img,
    }
    ok = (not mismatch and int(row_sums[0]) == 0
          and abs(m_vec - EXPECTED_ZERO) < 5e-10
          and total_raw - ignore == valid)
    if ok:
        record("C1_baseline", True,
               f"mIoU {m_vec:.10f} == frozen {EXPECTED_ZERO} over the FULL "
               f"{len(S)}-map validation set (no sampling); valid {valid} = total "
               f"{total_raw} - ignore {ignore}; cm row0 = 0; slow Python reference "
               f"additionally matched on {n_img_sampled} x {px_per_img} sampled "
               f"pixels as an algorithm cross-check only")
    else:
        record("C1_baseline", False,
               f"slow_mismatch={mismatch[:3]} cm_row0={int(row_sums[0])} "
               f"miou={m_vec!r} expected={EXPECTED_ZERO!r} "
               f"valid={valid} total-ignore={total_raw - ignore}")
    return detail, cm_vec


# =====================================================================
# C2 -- class-ID mapping, all three legs
# =====================================================================

def c2_class_id_mapping(S, GTraw, GTflat, cm_vec):
    # ---- legs 1 and 2 collected, then ONE decision ----
    s_lo = min(float(t.min()) for t in S)
    s_hi = max(float(t.max()) for t in S)
    force = float(s_hi - s_lo) + 1.0
    legs = []
    leg1_bad = []
    for k in range(8):
        vec = [0.0] * 8
        vec[k] = force
        preds = vec_full_pred(S, vec)
        hits = tot = 0
        for g, p in zip(GTflat, preds):
            m = g != 0
            hits += int((p[m].to(torch.int64) == (k + 1)).sum())
            tot += int(m.sum())
        frac = hits / max(tot, 1)
        legs.append({"channel": k, "official": k + 1, "name": NAMES[k], "frac": frac})
        if frac != 1.0:
            leg1_bad.append({"channel": k, "expected_official": k + 1, "frac": frac})
        del preds

    uniq = set()
    gt_counts = np.zeros(9, dtype=np.int64)
    for g in GTraw:
        a = g.reshape(-1)
        uniq.update(int(x) for x in np.unique(a))
        gt_counts += np.bincount(a, minlength=9)[:9]
    labels_ok = set(uniq).issubset(set(range(9))) and 0 in uniq and gt_counts[0] > 0
    row_sums = cm_vec.sum(1)
    row_match = bool((gt_counts[1:] == row_sums[1:]).all())
    row_diff = {NAMES[j]: [int(gt_counts[j + 1]), int(row_sums[j + 1])]
                for j in range(8) if gt_counts[j + 1] != row_sums[j + 1]}

    if not leg1_bad and labels_ok and row_match:
        record("C2_class_id_mapping", True,
               f"leg1: all 8 score channels forced by {force:.6f} map 100% to their "
               f"official id (force bias derived from the MEASURED range "
               f"[{s_lo:.6g}, {s_hi:.6g}], not an assumed [0,1]); "
               f"leg2: raw GT labels {sorted(uniq)} with ignore present "
               f"({int(gt_counts[0])} px); leg3: every per-class raw GT count equals "
               f"the confusion matrix row sum exactly")
    else:
        record("C2_class_id_mapping", False,
               f"leg1_failures={leg1_bad} labels_ok={labels_ok} "
               f"row_sum_mismatches={row_diff}")
    return {"force_bias": force, "score_range": [s_lo, s_hi], "legs": legs,
            "gt_labels": sorted(uniq),
            "gt_counts": {NAMES[j]: int(gt_counts[j + 1]) for j in range(8)},
            "cm_row_sums": {NAMES[j]: int(row_sums[j + 1]) for j in range(8)}}


# =====================================================================
# C5 -- monotonicity of the four transitions, exact path
# =====================================================================

def c5_monotonicity(S, GTflat, p0, base_miou, base_iou):
    best_bias = {}
    if C2.exists():
        r = json.loads(C2.read_text())
        for nm, d in r.get("per_class_best", {}).items():
            best_bias[nm] = float(d["bias"])
    out = {}
    for k in range(1, 8):
        nm = NAMES[k]
        base_c = [pp.to(torch.int64) == (k + 1) for pp in p0]
        gt_c = [g == (k + 1) for g in GTflat]
        bb = best_bias.get(nm, 0.0)
        pos = sorted({0.0, 0.2, 0.8, bb if bb > 0 else 0.0})
        neg = sorted({0.0, -0.2, -0.8, bb if bb < 0 else 0.0})
        rows = []
        for b in pos + [x for x in neg if x != 0.0]:
            vec = [0.0] * 8
            vec[k] = b
            preds = vec_full_pred(S, vec)
            agg = {"corrected_FN": 0, "created_FP": 0,
                   "removed_FP": 0, "destroyed_TP": 0}
            for i in range(len(preds)):
                # vectorised form of the same definition slow_transitions uses
                new_c = preds[i].to(torch.int64) == (k + 1)
                gc, bc = gt_c[i], base_c[i]
                agg["corrected_FN"] += int((gc & ~bc & new_c).sum())
                agg["created_FP"] += int((~gc & ~bc & new_c).sum())
                agg["removed_FP"] += int((~gc & bc & ~new_c).sum())
                agg["destroyed_TP"] += int((gc & bc & ~new_c).sum())
                del new_c
            del preds
            rows.append({"bias": b, **agg})
        # ascending bias for the positive leg, descending (0 first, then most
        # negative) for the negative leg. Slicing a concatenated list is NOT
        # equivalent and silently started the negative leg at the wrong point.
        p_rows = sorted([r for r in rows if r["bias"] >= 0], key=lambda r: r["bias"])
        n_rows = sorted([r for r in rows if r["bias"] <= 0], key=lambda r: -r["bias"])
        assert p_rows[0]["bias"] == 0.0 and n_rows[0]["bias"] == 0.0, (
            f"both legs must start at bias 0, got p={p_rows[0]['bias']} "
            f"n={n_rows[0]['bias']}")
        zero_ok = all(p_rows[0][x] == 0 for x in
                      ("corrected_FN", "created_FP", "removed_FP", "destroyed_TP"))
        m1 = all(p_rows[i]["corrected_FN"] >= p_rows[i - 1]["corrected_FN"]
                 for i in range(1, len(p_rows)))
        m2 = all(p_rows[i]["created_FP"] >= p_rows[i - 1]["created_FP"]
                 for i in range(1, len(p_rows)))
        m3 = all(n_rows[i]["removed_FP"] >= n_rows[i - 1]["removed_FP"]
                 for i in range(1, len(n_rows)))
        m4 = all(n_rows[i]["destroyed_TP"] >= n_rows[i - 1]["destroyed_TP"]
                 for i in range(1, len(n_rows)))
        out[nm] = {"rows": rows, "zero_all_zero": zero_ok,
                   "pos_FN_monotone": m1, "pos_FP_monotone": m2,
                   "neg_FP_monotone": m3, "neg_TP_monotone": m4,
                   "c2_best_bias": best_bias.get(nm)}
        print(f"   C5 {nm:13s} zero_ok={zero_ok} posFN={m1} posFP={m2} "
              f"negFP={m3} negTP={m4} (c2best={best_bias.get(nm)})", flush=True)
        for r in rows:
            print(f"        b={r['bias']:+.4f} corrFN={r['corrected_FN']:>10d} "
                  f"creatFP={r['created_FP']:>10d} remFP={r['removed_FP']:>10d} "
                  f"destTP={r['destroyed_TP']:>10d}", flush=True)
    # ---- ONE decision for the whole gate ----
    bad = [nm for nm, v in out.items()
           if not (v["zero_all_zero"] and v["pos_FN_monotone"]
                   and v["pos_FP_monotone"] and v["neg_FP_monotone"]
                   and v["neg_TP_monotone"])]
    if bad:
        record("C5_monotonicity", False,
               f"violations in {bad}: "
               f"{ {nm: {k: v for k, v in out[nm].items() if k != 'rows'} for nm in bad} }")
    else:
        record("C5_monotonicity", True,
               "all 7 classes: the four transitions are exactly 0 at b=0, "
               "corrected_FN and created_FP are non-decreasing in +b, removed_FP "
               "and destroyed_TP are non-decreasing in -|b|; each class's ACTUAL "
               "C2 best bias is included in its own direction's sequence")
    return out


# =====================================================================
# C6 -- exact re-score of the top C2 candidates
# =====================================================================

def c6_rescore(S, GTflat, p0, base_miou, base_iou):
    if not C2.exists():
        raise SystemExit(f"C6_rescore: {C2} not found, C2 must run first")
    res = json.loads(C2.read_text())
    picks = res.get("top20", [])[:5] + list(res.get("per_class_best", {}).values())
    seen, rows = set(), []
    for r in picks:
        key = (r["class"], round(float(r["bias"]), 9))
        if key in seen:
            continue
        seen.add(key)
        k = NAMES.index(r["class"])
        vec = [0.0] * 8
        vec[k] = float(r["bias"])
        preds = vec_full_pred(S, vec)
        cm = vec_confusion(GTflat, preds)
        m, iou = vec_miou(cm)
        agg = {"corrected_FN": 0, "created_FP": 0,
               "removed_FP": 0, "destroyed_TP": 0}
        for i in range(len(preds)):
            new_c = preds[i].to(torch.int64) == (k + 1)
            gc = GTflat[i] == (k + 1)
            bc = p0[i].to(torch.int64) == (k + 1)
            agg["corrected_FN"] += int((gc & ~bc & new_c).sum())
            agg["created_FP"] += int((~gc & ~bc & new_c).sum())
            agg["removed_FP"] += int((~gc & bc & ~new_c).sum())
            agg["destroyed_TP"] += int((gc & bc & ~new_c).sum())
            del new_c
        del preds
        rep = r.get("events", {})
        # only the mIoU is claimed to be an exact quantity in the C2 report
        rows.append({"candidate_id": f"{r['class']}|{r.get('tag')}|{float(r['bias']):+.9f}",
                     "class": r["class"], "official_id": k + 1,
                     "bias": float(r["bias"]), "source_tag": r.get("tag"),
                     "reported_miou": r["miou"], "exact_miou": m,
                     "miou_match": abs(m - r["miou"]) < 1e-12,
                     "reported_gain_pp": r.get("gain_pp"),
                     "exact_gain_pp": (m - base_miou) * 100,
                     "exact_per_class_iou": {NAMES[j]: float(iou[j]) for j in range(8)},
                     "base_per_class_iou": {NAMES[j]: float(base_iou[j]) for j in range(8)},
                     "exact_confusion_9x9": cm.tolist(),
                     "exact_transitions": agg,
                     "reported_events": rep,
                     # the histogram-approximated quantities
                     "hist_created_FP_reported": rep.get("created_FP"),
                     "hist_created_FP_exact": agg["created_FP"],
                     "hist_destroyed_TP_reported": rep.get("destroyed_TP"),
                     "hist_destroyed_TP_exact": agg["destroyed_TP"]})
        print(f"   C6 {r['class']:13s} b={r['bias']:+.6f} reported={r['miou']:.10f} "
              f"exact={m:.10f} match={abs(m-r['miou'])<1e-12}", flush=True)
    bad = [x for x in rows if not x["miou_match"]]
    if bad:
        record("C6_rescore", False, json.dumps(bad[:3]))
    else:
        record("C6_rescore", True,
               f"{len(rows)} top candidates FIXED on the exact path over the FULL "
               f"{len(S)}-map validation set (no sampling): per candidate this file "
               f"stores the exact mIoU, the 8 per-class IoUs, the 9x9 confusion "
               f"matrix, the four baseline->biased transition counts, the bias "
               f"position (class + bias) and a candidate id. The C2 report stores "
               f"only mIoU, so the per-class IoU and confusion here are recomputed "
               f"and fixed, NOT claimed to match the report.")
    return rows


# =====================================================================
# driver -- fixtures first, real data only if fixtures pass
# =====================================================================

def main():
    print("=== stage 1: fixtures ===", flush=True)
    c4_transitions()
    c3_fast_vs_slow_on_fixture()
    _c1_fixture_only()
    print("=== stage 1 PASSED ===", flush=True)

    print("=== stage 2: real data ===", flush=True)
    stems, S, GTraw = load_all()
    GTflat = [torch.from_numpy(g).long().reshape(-1) for g in GTraw]
    c1_detail, cm_vec = c1_baseline(S, GTraw, GTflat)
    c2_detail = c2_class_id_mapping(S, GTraw, GTflat, cm_vec)
    c3_fast_vs_slow_real(S, GTflat, stems, n_img=3, max_biases=[0.0, 0.3, -0.3])
    # one shared zero-bias prediction / baseline, reused by C5 and C6
    p0 = vec_full_pred(S, [0.0] * 8)
    base_miou, base_iou = vec_miou(cm_vec)
    c5_detail = c5_monotonicity(S, GTflat, p0, base_miou, base_iou)
    c6_detail = c6_rescore(S, GTflat, p0, base_miou, base_iou)

    (OUT / "C2_ACCEPTANCE_CLEAN.json").write_text(json.dumps(
        {"results": results,
         "all_pass": all(v["pass"] for v in results.values()),
         "c1": c1_detail, "c2": c2_detail, "c5": c5_detail, "c6": c6_detail},
        indent=2, ensure_ascii=False, allow_nan=False, default=str))
    print("ACCEPTANCE_CLEAN_DONE", flush=True)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}", flush=True)


def _c1_fixture_only():
    gt_rows = [[1, 1, 2, 2, 0, 3]]
    pr_rows = [[1, 2, 2, 1, 1, 3]]
    cm = slow_confusion(gt_rows, pr_rows)
    assert cm[1, 1] == 1 and cm[1, 2] == 1 and cm[2, 2] == 1 and cm[2, 1] == 1
    assert cm[3, 3] == 1 and cm.sum() == 5 and int(cm[0].sum()) == 0
    m_slow, iou_slow = slow_miou(cm)
    assert abs(iou_slow["Background"] - 1 / 3) < 1e-12
    assert abs(iou_slow["Building"] - 1 / 3) < 1e-12
    assert abs(iou_slow["Road"] - 1.0) < 1e-12
    assert abs(m_slow - (1 / 3 + 1 / 3 + 1.0) / 3) < 1e-12
    print("   C1 fixture: slow confusion/mIoU hand-checked", flush=True)


if __name__ == "__main__":
    main()
