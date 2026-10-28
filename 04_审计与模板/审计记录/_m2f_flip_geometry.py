#!/usr/bin/env python3
"""Local calibration-geometry readout from the cached M2F Run1b TTA8 scores.

Why
---
R_C (the calibration response) needs platform scores we don't have tonight. But
flip% -- the fraction of pixels that change class when a bias is applied -- is a
pure GEOMETRY quantity and is fully measurable locally from the cached scores.

The E-table definition is E = d mIoU(pp) / flip%. If flip% moves by an order of
magnitude across the training trajectory, the score space's movability is
training-state dependent, which is exactly the hypothesis behind any further
GPU work. If flip% is flat, the geometry is fixed and more training of this kind
is not indicated.

Reports, per checkpoint and per bias vector:
  flip%        total fraction of pixels whose argmax changes
  BG->FG       of those, the fraction going background -> foreground
  per-class    the full transition breakdown for the largest transitions

Pure CPU. Reads the score cache written by _m2f_calib_traj_cache.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
FG020 = [0.0] + [0.20] * 7
C = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
V13 = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 2.9215, -0.5]
VARIANTS = [("FG020", FG020), ("C", C), ("13", V13)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[20000, 40000, 56000])
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    report = {"cache_dir": str(cache), "steps": {}}

    for step in args.steps:
        cdir = cache / f"score_{step}"
        stems = sorted(p.stem for p in cdir.glob("*.npy"))
        if len(stems) != 500:
            raise SystemExit(f"{cdir}: expected 500, got {len(stems)}")

        # confusion[from, to] accumulated over all pixels of all images
        conf = {name: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                for name, _ in VARIANTS}
        total_px = 0

        for stem in stems:
            mean = np.load(cdir / f"{stem}.npy")
            if mean.shape != (NUM_CLASSES, 1024, 1024):
                raise SystemExit(f"{stem}: shape {mean.shape}")
            base = mean.argmax(0)
            total_px += base.size
            for name, vec in VARIANTS:
                alt = (mean + np.asarray(vec, dtype=np.float32)[:, None, None]).argmax(0)
                moved = base != alt
                if not moved.any():
                    continue
                idx = base[moved].astype(np.int64) * NUM_CLASSES + alt[moved].astype(np.int64)
                conf[name] += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES
                                          ).reshape(NUM_CLASSES, NUM_CLASSES)

        entry = {"total_pixels": total_px, "variants": {}}
        for name, _ in VARIANTS:
            c = conf[name]
            flips = int(c.sum() - np.trace(c))
            bg2fg = int(c[0, 1:].sum())
            fg2bg = int(c[1:, 0].sum())
            pairs = []
            for a in range(NUM_CLASSES):
                for b in range(NUM_CLASSES):
                    if a != b and c[a, b] > 0:
                        pairs.append((int(c[a, b]), a + 1, b + 1))
            pairs.sort(reverse=True)
            entry["variants"][name] = {
                "flip_pct": 100.0 * flips / total_px,
                "flips": flips,
                "bg_to_fg_pct_of_flips": 100.0 * bg2fg / flips if flips else 0.0,
                "fg_to_bg_pct_of_flips": 100.0 * fg2bg / flips if flips else 0.0,
                "top_transitions": [{"n": n, "from_official": a, "to_official": b,
                                     "pct_of_flips": 100.0 * n / flips if flips else 0.0}
                                    for n, a, b in pairs[:8]],
            }
            v = entry["variants"][name]
            print(f"  step {step:6d} {name:6s} flip={v['flip_pct']:.4f}%  "
                  f"BG->FG {v['bg_to_fg_pct_of_flips']:.1f}%  "
                  f"FG->BG {v['fg_to_bg_pct_of_flips']:.1f}%", flush=True)
        report["steps"][str(step)] = entry
        print(flush=True)

    print("=== flip% trajectory (the key readout) ===")
    header = "  step    " + "".join(f"{n:>12s}" for n, _ in VARIANTS)
    print(header)
    for step in args.steps:
        row = f"  {step:<6d}  "
        for name, _ in VARIANTS:
            row += f"{report['steps'][str(step)]['variants'][name]['flip_pct']:12.4f}"
        print(row)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
