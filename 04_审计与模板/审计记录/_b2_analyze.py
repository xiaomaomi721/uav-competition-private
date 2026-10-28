#!/usr/bin/env python3
"""B2 CPU analysis: champion per-class audit + FN/FP error overlap + affine sweep.

Reads the logits cache written by _b2_val_logits.py (single-view val, GT grid,
exact evaluate_model convention). No GPU.

Three outputs
-------------
1. per-class audit of the champion at the champion operator b*.
2. error overlap vs the Barren/Agri-targeted model, split by class into FN and
   FP with Jaccard, so "both miss the same pixels" can be told apart from "each
   misses different pixels". Whole-image error overlap is not informative --
   Background dominates it.
3. single-class affine scale sweep  z'_c = a_c * z_c + b_c  with b fixed at b*.
   Per GPT's correction this is a FIRST-STAGE SCREEN only: a_c and b_c are
   coupled and b* was found at a_c = 1, so a class showing promise here earns a
   small joint (a_c, b_c) search rather than a verdict.
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
# sweep only the classes that the audit flags; Barren first by prior evidence
SWEEP_CLASSES = [4, 7, 0, 6, 1, 2]      # Barren, Vehicle, Background, Agri, Building, Road
SWEEP_SCALES = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]


def cm_from(pred, gt, cm):
    valid = gt != IGNORE
    p = pred[valid].astype(np.int64)
    g = gt[valid].astype(np.int64)
    np.add.at(cm, (g, p), 1)


def iou_from(cm):
    out = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        den = cm[c, :].sum() + cm[:, c].sum() - tp
        out.append(float(tp) / float(den) if den else float("nan"))
    return np.array(out)


def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (float(inter) / float(union) if union else float("nan")), int(inter), int(union)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.cache)
    stems = sorted(p.stem for p in (root / "gt").glob("*.npy"))
    if len(stems) != 1399:
        raise SystemExit(f"expected 1399, got {len(stems)}")

    report = {"cache": str(root), "val_count": len(stems),
              "bstar": BSTAR.tolist(),
              "protocol": "single-view val, GT grid, exact evaluate_model convention"}

    # ---------------- 1. per-class audit at b* ----------------
    cms = {}
    for tag in ("champ", "barrenAgri"):
        d = root / f"logits_{tag}"
        if not d.is_dir():
            raise SystemExit(f"missing {d}")
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for s in stems:
            lg = np.load(d / f"{s}.npy").astype(np.float32)
            gt = np.load(root / "gt" / f"{s}.npy")
            pred = (lg + BSTAR[:, None, None]).argmax(0)
            cm_from(pred, gt, cm)
        cms[tag] = cm
        iou = iou_from(cm)
        report[tag] = {"per_class_iou": iou.tolist(),
                       "mIoU": float(np.nanmean(iou)),
                       "confusion": cm.tolist()}
        print(f"=== {tag} @b* ===")
        for i, c in enumerate(CLASSES):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            gt_n = cm[i, :].sum()
            pr_n = cm[:, i].sum()
            rec = tp / gt_n if gt_n else float("nan")
            pre = tp / pr_n if pr_n else float("nan")
            print("  %-12s IoU %.4f  P %.4f  R %.4f  GT %10d  pred %10d"
                  % (c, iou[i], pre, rec, gt_n, pr_n))
        print("  mIoU = %.6f\n" % np.nanmean(iou))

    if "barrenAgri" in cms:
        a, b = cms["champ"], cms["barrenAgri"]
        print("=== per-class FN/FP error overlap: champ vs barrenAgri ===")
        rep = {}
        for i, c in enumerate(CLASSES):
            tp_a = a[i, i]; tp_b = b[i, i]
            fn_a = a[i, :].sum() - tp_a; fn_b = b[i, :].sum() - tp_b
            fp_a = a[:, i].sum() - tp_a; fp_b = b[:, i].sum() - tp_b
            print("  %-12s FN champ %9d  barrenAgri %9d   FP champ %9d  barrenAgri %9d"
                  % (c, fn_a, fn_b, fp_a, fp_b))
            rep[c] = {"fn_champ": int(fn_a), "fn_other": int(fn_b),
                      "fp_champ": int(fp_a), "fp_other": int(fp_b)}
        report["overlap_totals"] = rep
        print()

    # pixel-level FN/FP overlap needs the argmax maps; redo for key classes
    print("=== pixel-level overlap (Jaccard) for key classes ===")
    key = [4, 7, 0, 6]     # Barren, Vehicle, Background, Agricultural
    acc = {c: {"fn_inter": 0, "fn_union": 0, "fp_inter": 0, "fp_union": 0,
               "fn_a": 0, "fn_b": 0, "fp_a": 0, "fp_b": 0,
               "disagree": 0, "total": 0} for c in key}
    for s in stems:
        ga = np.load(root / "logits_champ" / f"{s}.npy").astype(np.float32)
        gb = np.load(root / "logits_barrenAgri" / f"{s}.npy").astype(np.float32)
        gt = np.load(root / "gt" / f"{s}.npy")
        pa = (ga + BSTAR[:, None, None]).argmax(0)
        pb = (gb + BSTAR[:, None, None]).argmax(0)
        valid = gt != IGNORE
        for c in key:
            fna = valid & (gt == c) & (pa != c)
            fnb = valid & (gt == c) & (pb != c)
            fpa = valid & (pa == c) & (gt != c)
            fpb = valid & (pb == c) & (gt != c)
            d = acc[c]
            d["fn_inter"] += int(np.logical_and(fna, fnb).sum())
            d["fn_union"] += int(np.logical_or(fna, fnb).sum())
            d["fp_inter"] += int(np.logical_and(fpa, fpb).sum())
            d["fp_union"] += int(np.logical_or(fpa, fpb).sum())
            d["fn_a"] += int(fna.sum()); d["fn_b"] += int(fnb.sum())
            d["fp_a"] += int(fpa.sum()); d["fp_b"] += int(fpb.sum())
            d["disagree"] += int((pa != pb).sum()); d["total"] += int(valid.sum())
    ov = {}
    for c in key:
        d = acc[c]
        fnj = d["fn_inter"] / d["fn_union"] if d["fn_union"] else float("nan")
        fpj = d["fp_inter"] / d["fp_union"] if d["fp_union"] else float("nan")
        both = d["fn_inter"] / min(d["fn_a"], d["fn_b"]) if min(d["fn_a"], d["fn_b"]) else float("nan")
        print("  %-12s FN Jaccard %.4f (both-miss/min %.4f)   FP Jaccard %.4f   pixel-disagree %.4f%%"
              % (CLASSES[c], fnj, both, fpj, 100.0 * d["disagree"] / d["total"]))
        ov[CLASSES[c]] = {"fn_jaccard": fnj, "fn_both_miss_frac": both,
                          "fp_jaccard": fpj,
                          "pixel_disagree_pct": 100.0 * d["disagree"] / d["total"], **d}
    report["pixel_overlap"] = ov
    print()

    # ---------------- 3. affine scale sweep at fixed b* ----------------
    print("=== single-class affine scale sweep (b fixed at b*), champ ===")
    print("  FIRST-STAGE SCREEN: a_c and b_c are coupled and b* was found at a_c=1,")
    print("  so a positive class earns a joint (a_c,b_c) search, not this table alone.")
    print()
    base_iou = iou_from(cms["champ"])
    base_m = float(np.nanmean(base_iou))
    print("  baseline mIoU %.6f" % base_m)
    sweep = {}
    for c in SWEEP_CLASSES:
        row = []
        for a in SWEEP_SCALES:
            cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            for s in stems:
                lg = np.load(root / "logits_champ" / f"{s}.npy").astype(np.float32)
                gt = np.load(root / "gt" / f"{s}.npy")
                lg[c] *= a
                pred = (lg + BSTAR[:, None, None]).argmax(0)
                cm_from(pred, gt, cm)
            iou = iou_from(cm)
            m = float(np.nanmean(iou))
            row.append({"a": a, "mIoU": m, "d_mIoU_pp": 100 * (m - base_m),
                        "iou_c": float(iou[c])})
        sweep[CLASSES[c]] = row
        print("  %-12s" % CLASSES[c])
        for r in row:
            print("     a=%.2f  mIoU %.6f  d %+.4fpp   IoU_%s %.4f"
                  % (r["a"], r["mIoU"], r["d_mIoU_pp"], CLASSES[c][:4], r["iou_c"]))
    report["affine_scale_sweep"] = sweep

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
