#!/usr/bin/env python3
"""INVARIANT: image and teacher soft map must receive IDENTICAL geometry.

GPT flagged this as the most dangerous silent bug class: if flip/rotate is applied
to the test image but not to the cached soft map, the KD loss still goes down --
the student just learns spatially misaligned targets.

Rather than read the code and reason about it, this encodes the geometry IN the
data and checks it survives:

  image[y, x] = (y, x) as a 2-channel coordinate field
  aux[y, x]   = the same coordinate field

After _apply_geom both must still agree pixel-for-pixel. Any axis mix-up
(e.g. flipping axis=1 on one but axis=0 on the other, or transposing the
channels-last aux) shows up immediately as a coordinate mismatch.
"""
from __future__ import annotations

import random
import sys

import pathlib

import numpy as np

sys.path.insert(0, "/home/jhy/rare_diag_20260915")
sys.path.insert(0, "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")


def decode_geom(img):
    """Recover (y, x) from the encoded coordinate field."""
    return img[..., 0].astype(np.int32), img[..., 1].astype(np.int32)


def main() -> int:
    H, W, C = 16, 24, 8
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    def make():
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[..., 0] = yy
        img[..., 1] = xx
        # aux carries the same coordinate field, one-hot-ish per row/col so a
        # wrong axis transposition cannot accidentally still agree
        aux = np.zeros((H, W, C), dtype=np.float32)
        aux[..., 0] = yy / H
        aux[..., 1] = xx / W
        return img, aux

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_kd", "/home/jhy/rare_diag_20260915/_kd_train.py")
    kd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kd)

    # reuse the exact helper the Dataset uses
    import tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "one.txt").write_text("dummy", encoding="utf-8")
    loader = kd.build_loader(
        {"crop_size": 16}, tmp, tmp, "one.txt", None, tmp, tmp, use_probs=False)
    # _apply_geom is defined on the inner class; grab it off the instance
    ds = loader.dataset

    fails = 0
    for trial in range(30):
        rng = random.Random(trial)
        img, aux = make()
        p = ds._aug_params(rng)
        img2, aux2 = ds._apply_geom(img, aux, p)
        y_img, x_img = decode_geom(img2)
        # aux[...,0] holds y/H and aux[...,1] holds x/W
        y_aux = np.round(aux2[..., 0] * H).astype(np.int32)
        x_aux = np.round(aux2[..., 1] * W).astype(np.int32)
        ok = np.array_equal(y_img, y_aux) and np.array_equal(x_img, x_aux)
        if not ok:
            fails += 1
            print(f"  MISALIGNED trial={trial} params={p}")
            print(f"    image y range {y_img.min()}..{y_img.max()}  "
                  f"aux y range {y_aux.min()}..{y_aux.max()}")
    print(f"\n30 random geometries: {30 - fails} aligned, {fails} misaligned")
    print("RESULT:", "PASS - image and teacher map share identical geometry"
          if fails == 0 else "FAIL - spatial targets are misaligned")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
