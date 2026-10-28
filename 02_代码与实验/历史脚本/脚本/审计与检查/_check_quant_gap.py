#!/usr/bin/env python3
"""Why does the cached teacher argmax differ from ensemble 30 on 0.031% of pixels?

Hypothesis: uint8 quantisation. The cache stores round(p*255)/255, so two
classes whose probabilities differ by less than 1/255 can round to the same
value, and argmax then falls back to the lower class index.

Test: for every differing pixel, measure the gap between the top-2 probabilities
in the cache. If the hypothesis is right, essentially all differences should sit
at or below the 1/255 quantisation step.

CPU only.
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

NUM_CLASSES = 8
Q = 1.0 / 255.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft-dir", required=True)
    ap.add_argument("--reference-zip", required=True)
    args = ap.parse_args()

    prob_dir = Path(args.soft_dir) / "probs"
    zf = zipfile.ZipFile(args.reference_zip)
    ref = {}
    for n in zf.namelist():
        if n.lower().endswith(".png"):
            with Image.open(io.BytesIO(zf.read(n))) as im:
                ref[Path(n).stem] = np.asarray(im).copy()

    gaps = []
    n_img = 0
    for f in sorted(prob_dir.glob("*.npy")):
        stem = f.stem
        if stem not in ref:
            continue
        a = np.load(f).astype(np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        mine = (a.argmax(axis=2) + 1).astype(np.uint8)
        theirs = ref[stem]
        if mine.shape != theirs.shape:
            mine = np.asarray(Image.fromarray(mine).resize(
                (theirs.shape[1], theirs.shape[0]), Image.NEAREST)).copy()
        bad = mine != theirs
        if not bad.any():
            n_img += 1
            continue
        srt = np.sort(a, axis=2)
        # gap between best and second-best probability at the differing pixels
        g = (srt[..., -1] - srt[..., -2])[bad]
        gaps.append(g)
        n_img += 1
        if n_img >= 500:
            break

    if not gaps:
        print("no differing pixels found")
        return 0
    g = np.concatenate(gaps)
    print(f"differing pixels analysed: {g.size}")
    print(f"quantisation step 1/255 = {Q:.6f}")
    print(f"top-2 probability gap:  mean={g.mean():.6f}  median={np.median(g):.6f}  "
          f"max={g.max():.6f}")
    for thr in (Q, Q / 2, Q / 4):
        frac = float((g <= thr).mean())
        print(f"  fraction with gap <= {thr:.6f}: {frac:.4%}")
    if (g <= Q).mean() > 0.99:
        print("\nCONCLUSION: differences are explained by uint8 quantisation "
              "(top-2 gap below the rounding step). The teacher target is the "
              "right ensemble, just quantised.")
    else:
        print("\nCONCLUSION: some differences are NOT near-ties -- there may be a "
              "real pipeline difference (resize / aggregation / argmax timing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
