#!/usr/bin/env python3
"""Axis T post-step: train-ID masks -> official masks + manifest/stats, and
(optional) replay comparison against the historical s55b pseudo labels.

Historical pipeline being replicated (verified on the server, 2026-09-18):

    gen_pseudo_labels.py  --mode region --threshold 0.9   (UNMODIFIED, original file)
        teacher = stage23_runs/.../best_miou.pt   (Exp23 / Stage23B)
        TTA8 mean logits -> softmax -> conf/argmax
        region mode: a connected component of class c is kept iff >=80% of its
        pixels have conf >= threshold; everything else -> 255 (train-ID ignore)
        output: masks_region_t0p90/<name>.png   (train-ID 0-7, 255=ignore)
    convert_pseudo_to_official.py
        255 -> 0 ; 0-7 -> 1-8                     (official space, 0=ignore)
        output: masks_official/<name>.png
    prepare_pseudo_train.sh
        symlink as 7000..7499 + build train_ids_5597pseudo.txt

This file ONLY does the second step (parametrised) plus bookkeeping. The
generative logic lives in the ORIGINAL gen_pseudo_labels.py and is never
re-implemented here.

Axis T v1 changes exactly one thing: the teacher checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

IGNORE_TRAIN = 255
CLASSES = ["Background", "Building", "Road", "Water",
           "Barren", "Vegetation", "Agricultural", "Vehicle"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def convert(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png"))
    if not files:
        raise SystemExit(f"no masks in {src}")
    for f in files:
        a = np.asarray(Image.open(f).convert("L")).astype(np.int64)
        out = np.zeros_like(a)
        valid = a != IGNORE_TRAIN
        out[valid] = a[valid] + 1
        Image.fromarray(out.astype(np.uint8), mode="L").save(dst / f.name, format="PNG")
    return files


def histogram(d: Path):
    h = np.zeros(9, dtype=np.int64)
    files = sorted(d.glob("*.png"))
    for f in files:
        a = np.asarray(Image.open(f).convert("L"))
        h += np.bincount(a.ravel(), minlength=9)[:9]
    return h, len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-region", type=Path, required=True,
                    help="dir written by the ORIGINAL gen_pseudo_labels.py (train-ID space)")
    ap.add_argument("--out-official", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--teacher-checkpoint", type=str, required=True)
    ap.add_argument("--teacher-sha256", type=str, default="")
    ap.add_argument("--mode", type=str, default="region")
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--label-tag", type=str, required=True,
                    help="free text: which Axis T variant this is")
    ap.add_argument("--compare-to", type=Path, default=None,
                    help="replay check: historical masks_official dir to diff against")
    args = ap.parse_args()

    files = convert(args.src_region, args.out_official)
    h, n = histogram(args.out_official)
    total = int(h.sum())
    manifest = {
        "schema": "AXIS_T_PSEUDO_MANIFEST_V1",
        "created": datetime.now(timezone.utc).isoformat(),
        "label_tag": args.label_tag,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_sha256": args.teacher_sha256,
        "generator": "gen_pseudo_labels.py (ORIGINAL, unmodified)",
        "mode": args.mode,
        "threshold": args.threshold,
        "region_rule": "component kept iff >=80% of its pixels have conf >= threshold",
        "label_space": "official (0=ignore, 1-8=class)",
        "images": n,
        "pixels_total": total,
        "ignore_ratio": float(h[0] / total) if total else None,
        "per_class_share_pct": {
            CLASSES[i]: float(100 * h[i + 1] / total) for i in range(8)
        },
        "per_class_share_of_kept_pct": {
            CLASSES[i]: float(100 * h[i + 1] / (total - h[0]))
            for i in range(8)
        } if total - h[0] else None,
        "src_region_dir": str(args.src_region),
        "out_official_dir": str(args.out_official),
    }

    if args.compare_to is not None:
        ref_dir = args.compare_to
        ref_files = sorted(ref_dir.glob("*.png"))
        cur_files = sorted(args.out_official.glob("*.png"))
        same_names = [p.name for p in ref_files] == [p.name for p in cur_files]
        identical = 0
        per_class_delta = np.zeros(9, dtype=np.int64)
        mismatch_examples = []
        if same_names:
            for rf, cf in zip(ref_files, cur_files):
                ra = np.asarray(Image.open(rf).convert("L"))
                ca = np.asarray(Image.open(cf).convert("L"))
                d = (ra != ca)
                if not d.any():
                    identical += 1
                elif len(mismatch_examples) < 5:
                    mismatch_examples.append(
                        {"name": rf.name, "diff_px": int(d.sum()),
                         "diff_pct": float(100 * d.sum() / d.size)})
                per_class_delta += np.bincount(ca.ravel(), minlength=9)[:9].astype(np.int64)
                per_class_delta -= np.bincount(ra.ravel(), minlength=9)[:9].astype(np.int64)
        manifest["replay_check"] = {
            "reference_dir": str(ref_dir),
            "ref_images": len(ref_files),
            "cur_images": len(cur_files),
            "same_filenames": bool(same_names),
            "bit_identical_images": int(identical),
            "identity_ratio": float(identical / len(ref_files)) if ref_files else None,
            "mismatch_examples": mismatch_examples,
            "per_class_pixel_delta_cur_minus_ref": {
                ("ignore" if i == 0 else CLASSES[i - 1]): int(per_class_delta[i])
                for i in range(9)
            },
        }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1))

    if args.compare_to is not None:
        rc = manifest.get("replay_check") or {}
        if not rc.get("same_filenames", False):
            print("REPLAY_GATE FAIL: filename sets differ")
            return 1
        # The historical run used a different physical GPU (GPU0 back then); TTA8 in
        # fp32 should be reproducible, but near-threshold pixels can still flip on a
        # different device. Accept within tolerance, print the raw numbers either way.
        ident = rc.get("identity_ratio")
        deltas = rc.get("per_class_pixel_delta_cur_minus_ref") or {}
        worst_rel = 0.0
        total = manifest.get("pixels_total") or 1
        for k, v in deltas.items():
            worst_rel = max(worst_rel, abs(v) / total)
        ok = (ident is not None and ident >= 0.999 and worst_rel < 1e-4)
        print(f"REPLAY_GATE identity_ratio={ident} worst_class_delta_rel={worst_rel:.2e} "
              f"-> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
