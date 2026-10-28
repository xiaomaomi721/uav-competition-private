#!/usr/bin/env python3
"""INVARIANT CHECK: does the cached teacher soft target actually reproduce the
already-validated raw ensemble package 30.zip (0.683237)?

The entire KD line rests on the claim "the student is distilled from the same
ensemble that scored 0.683237". That must be verified at the MASK level, not
argued theoretically.

argmax(softmax(x / T)) == argmax(x) for any T > 0, so the argmax of the cached
soft target must be pixel-identical to the zero-bias TTA8 logit-mean ensemble
package -- provided aggregation order, resize timing and argmax timing match.

CPU only; no GPU needed.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
ZERO = [0.0] * NUM_CLASSES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft-dir", required=True, help="dir containing probs/*.npy")
    ap.add_argument("--reference-zip", required=True, help="e.g. 30.zip (0.683237)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    prob_dir = Path(args.soft_dir) / "probs"
    files = sorted(prob_dir.glob("*.npy"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .npy under {prob_dir}")

    zf = zipfile.ZipFile(args.reference_zip)
    ref = {}
    for n in zf.namelist():
        if n.lower().endswith(".png"):
            import io
            from PIL import Image
            with Image.open(io.BytesIO(zf.read(n))) as im:
                ref[Path(n).stem] = np.asarray(im).copy()

    total_px = 0
    diff_px = 0
    n_compared = 0
    missing = []
    per_image = []

    for f in files:
        stem = f.stem
        if stem not in ref:
            missing.append(stem)
            continue
        a = np.load(f).astype(np.float32)
        if a.dtype == np.uint8 or a.max() > 1.5:      # uint8 cache
            a = a / 255.0
        if a.shape[2] != NUM_CLASSES:
            raise ValueError(f"unexpected channel count in {f}: {a.shape}")
        mine = (a.argmax(axis=2) + 1).astype(np.uint8)  # train id -> official id
        theirs = ref[stem]
        if mine.shape != theirs.shape:
            # reference is 1024x1024; cache may be the native 256x256 grid
            from PIL import Image
            mine = np.asarray(Image.fromarray(mine).resize(
                (theirs.shape[1], theirs.shape[0]), Image.NEAREST)).copy()
        d = int((mine != theirs).sum())
        total_px += mine.size
        diff_px += d
        n_compared += 1
        per_image.append((stem, d, mine.size))

    print(f"compared {n_compared} images, {total_px} pixels")
    if missing:
        print(f"missing from reference zip: {len(missing)} (e.g. {missing[:3]})")
    if total_px:
        agree = 1.0 - diff_px / total_px
        print(f"pixel agreement = {agree:.6f}   ({diff_px} differing pixels)")
        if diff_px == 0:
            print("RESULT: EXACT MATCH -- cache reproduces the validated ensemble")
        elif agree > 0.999:
            print("RESULT: near match -- investigate resize/argmax timing")
        else:
            print("RESULT: MISMATCH -- do NOT run KD until reconciled")
    bad = sorted(per_image, key=lambda x: -x[1])[:5]
    if bad and bad[0][1] > 0:
        print("\nworst images:")
        for s, d, t in bad:
            print(f"   {s}: {d}/{t} pixels differ ({100.0*d/t:.3f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
