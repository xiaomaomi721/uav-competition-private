#!/usr/bin/env python3
"""CPU-only validation of the ATDA packaging path against a real submission.

Takes the already-verified champ237 TTA8 cache, runs it through exactly the
arithmetic and PNG writing that _atda_export.py performs (+Bstar, argmax, +1),
and compares the result pixel-by-pixel against the masks actually inside the
submitted 237_S_semeda_M.zip. If the disagreement is at fp16-rounding level,
the packer is trustworthy and every ATDA package uses the same path.

No GPU, no model forward. Reads only.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cache = Path(args.cache)
    zpath = Path(args.zip)
    vec = np.asarray(BSTAR, dtype=np.float32)[:, None, None]

    with zipfile.ZipFile(zpath) as zf:
        names = sorted(zf.namelist(), key=lambda n: int(Path(n).stem.split("_")[-1]))
        if len(names) != 500:
            raise SystemExit(f"expected 500 entries, got {len(names)}")

        total = 0
        mismatched = 0
        per_class_ours = np.zeros(8, dtype=np.int64)
        per_class_theirs = np.zeros(8, dtype=np.int64)
        worst_image = None
        worst_frac = -1.0
        png_roundtrip_ok = True

        for name in names:
            arr = np.load(cache / f"{Path(name).stem}.npy").astype(np.float32)
            ours = ((arr + vec).argmax(0).astype(np.uint8) + 1)
            theirs = np.asarray(Image.open(io.BytesIO(zf.read(name))))

            # does our own PNG write survive a decode unchanged?
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(ours), mode="L").save(buf, format="PNG")
            back = np.asarray(Image.open(io.BytesIO(buf.getvalue())))
            if not np.array_equal(back, ours):
                png_roundtrip_ok = False

            d = int((ours != theirs).sum())
            total += ours.size
            mismatched += d
            frac = d / ours.size
            if frac > worst_frac:
                worst_frac, worst_image = frac, name
            # predictions are official class ids 1..8, so index 0 is always empty
            per_class_ours += np.bincount(ours.ravel(), minlength=9)[1:]
            per_class_theirs += np.bincount(theirs.ravel(), minlength=9)[1:]

    frac = 100.0 * mismatched / total
    report = {
        "cache": str(cache),
        "submitted_zip": str(zpath),
        "zip_sha256_size": zpath.stat().st_size,
        "entries": len(names),
        "pixels": total,
        "mismatched_pixels": mismatched,
        "mismatch_pct": round(frac, 6),
        "worst_image": worst_image,
        "worst_image_mismatch_pct": round(100.0 * worst_frac, 6),
        "ours_class_share_pct": [round(float(x), 4) for x in
                                 100.0 * per_class_ours / per_class_ours.sum()],
        "theirs_class_share_pct": [round(float(x), 4) for x in
                                   100.0 * per_class_theirs / per_class_theirs.sum()],
        "max_class_share_diff_pp": round(float(np.max(np.abs(
            100.0 * per_class_ours / per_class_ours.sum()
            - 100.0 * per_class_theirs / per_class_theirs.sum()))), 6),
        "own_png_roundtrip_lossless": png_roundtrip_ok,
        # the cache is fp16 while the submission was written from fp32 scores,
        # so a few boundary pixels are expected to move; 0.01% is far beyond
        # any real disagreement.
        "pass": frac <= 0.01 and png_roundtrip_ok,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
