#!/usr/bin/env python3
"""CPU-only probe: what would the Axis T v2 pseudo labels look like?

Reads the champion's cached test1 TTA8 mean logits (champ237_test1_cache_20260918)
and applies the HYPOTHETICAL Axis T v2 rule for several operators, reporting the
per-class coverage of the accepted labels. No GPU, no model, no new inference.

Question it answers
-------------------
The historical pseudo500 (teacher = Exp23) kept, at conf>=0.9:
    ignore 27.85% of pixels, and Barren was only 0.31% of what it kept.
The champion predicts Barren on test1 about 3x more often than that teacher did
(0.865% ZERO vs the historical ~0.31% kept). So a champion teacher should carry
MORE Barren -- but how much survives the same filter? That decides whether
Axis T v2 is worth 5.4h of GPU or not.

Operators probed (all on the SAME cached mean logits):
    0   argmax(z)                      raw, no calibration, no filter
    1   argmax(z + b*)                 platform-verified decision rule
    and, for each, the conf>=0.9 kept subset (pixel rule; the historical run
    additionally used a connected-component region rule, which can only remove
    more, so these numbers are an UPPER BOUND on the historical-style output).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ["Background", "Building", "Road", "Water",
           "Barren", "Vegetation", "Agricultural", "Vehicle"]
B_STAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()

    fs = sorted(args.scores.glob("*.npy"),
                key=lambda p: int(p.stem.split("_")[-1]))
    if len(fs) != 500:
        raise SystemExit(f"expected 500 score files, found {len(fs)}")

    # accumulators: [operator][class] over ALL pixels, and over KEPT pixels
    tot_raw = np.zeros(9, dtype=np.int64)      # 9 slots: 8 classes in train order
    tot_b = np.zeros(9, dtype=np.int64)
    kept_raw = np.zeros(9, dtype=np.int64)
    kept_b = np.zeros(9, dtype=np.int64)
    conf_sum = 0.0
    total = 0

    for f in fs:
        lo = np.load(f).astype(np.float32)                     # [8,H,W]
        mx = lo.max(0)
        conf = 1.0 / np.exp(lo - mx).sum(0)                    # softmax max-prob
        a_raw = lo.argmax(0).astype(np.int64)
        a_b = (lo + B_STAR[:, None, None]).argmax(0).astype(np.int64)
        keep = conf >= args.threshold
        tot_raw += np.bincount(a_raw.ravel(), minlength=9)[:9]
        tot_b += np.bincount(a_b.ravel(), minlength=9)[:9]
        kept_raw += np.bincount(a_raw[keep].ravel(), minlength=9)[:9]
        kept_b += np.bincount(a_b[keep].ravel(), minlength=9)[:9]
        conf_sum += float(conf.sum())
        total += int(a_raw.size)

    def shares(counts):
        t = counts.sum()
        return [float(c) / t for c in counts], int(t)

    out = {"n_images": len(fs), "pixels_total": total,
           "threshold": args.threshold, "b_star": B_STAR.tolist(),
           "note": ("pixel rule only; the historical run also applied a "
                    "connected-component region rule, so these kept-fractions are "
                    "an upper bound on the historical-style output")}
    for tag, cnt in (("raw_allpix", tot_raw), ("bstar_allpix", tot_b),
                     ("raw_kept_conf", kept_raw), ("bstar_kept_conf", kept_b)):
        sh, t = shares(cnt)
        out[tag] = {"pixels": t, "ignore_ratio": float(1.0 - t / total),
                    "per_class_pct_of_kept": {CLASSES[i]: round(100 * sh[i], 4)
                                              for i in range(8)}}
    out["mean_confidence"] = conf_sum / total

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
