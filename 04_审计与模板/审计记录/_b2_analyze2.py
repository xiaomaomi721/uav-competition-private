#!/usr/bin/env python3
"""B2 CPU analysis, single-pass. Audit + FN/FP overlap + affine scale sweep.

The first version looped the scale sweep as (classes x scales x images), which
meant ~59k reloads of a 16 MB array -- far too slow. This version makes ONE pass
over the val set and derives everything from it.

Scale sweep without re-argmax: for class c with scale a, only channel c moves,
so the argmax of the perturbed score is fully determined by the unperturbed
top-2. Let z_c be channel c and (m1, k1) the largest score and its index:
  * if k1 == c: c survives iff a*z_c > second_max
  * else:       c wins    iff a*z_c > m1
So each scale is O(1) per pixel after one top-2 pass, with no per-scale argmax.
Correct as long as c is not the argmax at a>1 in a way that changes the runner-up
ordering -- for the a-range used here (0.8..1.2) and a single channel moved, the
only competitor ordering that matters is against the max, which is exact.

Bias b is fixed at b* -- per GPT's correction this is a FIRST-STAGE SCREEN, not
a verdict: a_c and b_c are coupled and b* was found at a_c = 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
IGNORE = 255
CLASSES = ["Background", "Building", "Road", "Water",
           "Barren", "Vegetation", "Agricultural", "Vehicle"]
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
KEY = [4, 7, 0, 6, 1, 2, 3, 5]
SWEEP_SCALES = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.cache)
    stems = sorted(p.stem for p in (root / "gt").glob("*.npy"))
    n = len(stems)
    if n != 1399:
        raise SystemExit(f"expected 1399, got {n}")

    cm_c = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_b = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    K = len(KEY)
    ov = {c: {"fn_i": 0, "fn_u": 0, "fp_i": 0, "fp_u": 0,
              "fn_c": 0, "fn_o": 0, "fp_c": 0, "fp_o": 0,
              "dis": 0, "tot": 0} for c in KEY}
    # scale sweep accumulators: for each key class and scale, per-class TP/FP/FN
    sw = {c: {a: np.zeros((3, NUM_CLASSES), dtype=np.int64) for a in SWEEP_SCALES}
          for c in KEY}
    base_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    base_gt = np.zeros(NUM_CLASSES, dtype=np.int64)
    base_pr = np.zeros(NUM_CLASSES, dtype=np.int64)
    base_ok = np.zeros(NUM_CLASSES, dtype=np.int64)   # class-c pixels argmax==c

    for si, s in enumerate(stems):
        gt = np.load(root / "gt" / f"{s}.npy")
        zc = np.load(root / "logits_champ" / f"{s}.npy").astype(np.float32)
        zb = np.load(root / "logits_barrenAgri" / f"{s}.npy").astype(np.float32)
        valid = gt != IGNORE

        sc = zc + BSTAR[:, None, None]
        sb = zb + BSTAR[:, None, None]
        pc = sc.argmax(0)
        pb = sb.argmax(0)

        g = gt[valid].astype(np.int64)
        p1 = pc[valid].astype(np.int64)
        p2 = pb[valid].astype(np.int64)
        np.add.at(cm_c, (g, p1), 1)
        np.add.at(cm_b, (g, p2), 1)

        for c in KEY:
            fna = valid & (gt == c) & (pc != c)
            fnb = valid & (gt == c) & (pb != c)
            fpa = valid & (pc == c) & (gt != c)
            fpb = valid & (pb == c) & (gt != c)
            d = ov[c]
            d["fn_i"] += int(np.logical_and(fna, fnb).sum())
            d["fn_u"] += int(np.logical_or(fna, fnb).sum())
            d["fp_i"] += int(np.logical_and(fpa, fpb).sum())
            d["fp_u"] += int(np.logical_or(fpa, fpb).sum())
            d["fn_c"] += int(fna.sum()); d["fn_o"] += int(fnb.sum())
            d["fp_c"] += int(fpa.sum()); d["fp_o"] += int(fpb.sum())
            d["dis"] += int((pc != pb).sum()); d["tot"] += int(valid.sum())

        # baseline counts (unperturbed champ)
        for c in range(NUM_CLASSES):
            base_tp[c] += int(((gt == c) & (pc == c) & valid).sum())
            base_gt[c] += int(((gt == c) & valid).sum())
            base_pr[c] += int(((pc == c) & valid).sum())
            base_ok[c] += int(((pc == c) & valid).sum())

        # ---- scale sweep, analytic flip test ----
        m1 = sc.max(0)
        k1 = sc.argmax(0)
        srt = np.partition(sc, -2, axis=0)
        m2 = srt[-2]                       # second largest per pixel
        for c in KEY:
            z = sc[c]
            is_k1 = (k1 == c)
            for a in SWEEP_SCALES:
                if a == 1.0:
                    newc = (pc == c)
                else:
                    wins = np.where(is_k1, (a * z) > m2, (a * z) > m1)
                    newc = wins & valid
                prev = (pc == c) & valid
                tp = int((newc & (gt == c)).sum())
                fp = int((newc & (gt != c)).sum())
                fn = int(((gt == c) & valid & ~newc).sum())
                acc = sw[c][a]
                acc[0, c] += tp; acc[1, c] += fp; acc[2, c] += fn

        if (si + 1) % 300 == 0:
            print(f"  {si+1}/{n}", flush=True)

    def iou_from(cm):
        out = []
        for c in range(NUM_CLASSES):
            tp = cm[c, c]
            den = cm[c, :].sum() + cm[:, c].sum() - tp
            out.append(float(tp) / float(den) if den else float("nan"))
        return np.array(out)

    rep = {"cache": str(root), "n": n, "bstar": BSTAR.tolist(),
           "protocol": "single-view val, GT grid, evaluate_model convention, b fixed at b*"}
    ic, ib = iou_from(cm_c), iou_from(cm_b)
    rep["champ"] = {"per_class_iou": ic.tolist(), "mIoU": float(np.nanmean(ic)),
                    "confusion": cm_c.tolist()}
    rep["barrenAgri"] = {"per_class_iou": ib.tolist(), "mIoU": float(np.nanmean(ib)),
                         "confusion": cm_b.tolist()}

    print("\n=== champion of record @b* ===")
    for i, c in enumerate(CLASSES):
        tp = cm_c[i, i]
        gt_n = cm_c[i, :].sum(); pr_n = cm_c[:, i].sum()
        print("  %-12s IoU %.4f  P %.4f  R %.4f  GT %10d  pred %10d"
              % (c, ic[i], tp / max(pr_n, 1), tp / max(gt_n, 1), gt_n, pr_n))
    print("  mIoU = %.6f" % np.nanmean(ic))

    print("\n=== barrenAgri (Barren/Agri-targeted model) @b* ===")
    for i, c in enumerate(CLASSES):
        print("  %-12s IoU %.4f" % (c, ib[i]))
    print("  mIoU = %.6f" % np.nanmean(ib))

    print("\n=== per-class FN / FP overlap: champ vs barrenAgri ===")
    print("  Jaccard ~0 -> the two models err on DIFFERENT pixels (complementary)")
    print("  Jaccard ~1 -> they err on the SAME pixels (shared bottleneck)")
    orep = {}
    for c in KEY:
        d = ov[c]
        fnj = d["fn_i"] / d["fn_u"] if d["fn_u"] else float("nan")
        fpj = d["fp_i"] / d["fp_u"] if d["fp_u"] else float("nan")
        both = d["fn_i"] / min(d["fn_c"], d["fn_o"]) if min(d["fn_c"], d["fn_o"]) else float("nan")
        print("  %-12s FN J %6.3f (both-miss/min %5.3f)  FP J %6.3f  disagree %.4f%%"
              % (CLASSES[c], fnj, both, fpj, 100.0 * d["dis"] / d["tot"]))
        orep[CLASSES[c]] = {"fn_jaccard": fnj, "fn_both_frac": both,
                            "fp_jaccard": fpj,
                            "pixel_disagree_pct": 100.0 * d["dis"] / d["tot"], **d}
    rep["overlap"] = orep

    base_m = float(np.nanmean(ic))
    print("\n=== single-class affine scale sweep, b fixed at b* (FIRST-STAGE SCREEN) ===")
    print("  baseline champ mIoU %.6f" % base_m)
    srep = {}
    for c in KEY:
        print("  --- %s ---" % CLASSES[c])
        rows = []
        for a in SWEEP_SCALES:
            acc = sw[c][a]
            iou = np.array([base_tp[i] - 0 + 0 for i in range(NUM_CLASSES)], dtype=np.float64)
            ious = []
            for i in range(NUM_CLASSES):
                tp = base_tp[i] if i != c else acc[0, i]
                fp = base_pr[i] - base_tp[i] if i != c else acc[1, i]
                fn = base_gt[i] - base_tp[i] if i != c else acc[2, i]
                den = tp + fp + fn
                ious.append(tp / den if den else float("nan"))
            m = float(np.nanmean(ious))
            rows.append({"a": a, "mIoU": m, "d_pp": 100 * (m - base_m), "iou_c": ious[c]})
            print("     a=%.2f  mIoU %.6f  %+.4fpp   IoU(%s) %.4f"
                  % (a, m, 100 * (m - base_m), CLASSES[c][:8], ious[c]))
        srep[CLASSES[c]] = rows
    rep["affine_scale_sweep"] = srep

    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
