#!/usr/bin/env python3
"""Axis T dataset assembler.

Builds a NEW dataset root holding

    <out>/train/images/<id>.png   real 0000..6995 (symlink to official) + 7000..7499 (test1_x, symlink)
    <out>/train/masks/<id>.png    real masks (symlink)                   + 7000..7499 (AXIS T pseudo masks)
    <out>/train_ids_5597pseudo.txt   = splits/train.txt (5597) + 7000..7499 (500)

which is byte-for-byte the same SHAPE as the historical
/home/jhy/datasets/pseudo_train_6796, so a stage55b-recipe config can be pointed
at it with train_split/dataset_root changed and nothing else.

HARD GUARD
----------
The historical root is the training data of real, scored models (stage55b, s62).
Re-linking its 7000..7499 entries silently rewrites their training set and
destroys reproducibility -- that is exactly what relink_pseudo_v2.sh did on
2026-09-08. This script therefore REFUSES to write anywhere inside
/home/jhy/datasets/pseudo_train_6796, and refuses to overwrite an existing out.

Writes only under --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

FORBIDDEN = Path("/home/jhy/datasets/pseudo_train_6796").resolve()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-root", type=Path, required=True,
                    help="official train root containing train/images + train/masks")
    ap.add_argument("--real-ids", type=Path, required=True,
                    help="split file of real train ids (e.g. <bundle>/splits/train.txt, 5597)")
    ap.add_argument("--test-root", type=Path, required=True,
                    help="official root containing images/test1_1..500.png")
    ap.add_argument("--pseudo-masks", type=Path, required=True,
                    help="AXIS T masks_official dir (official space, 0=ignore, 1-8)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label-tag", type=str, required=True)
    ap.add_argument("--teacher-sha256", type=str, default="")
    args = ap.parse_args()

    out = args.out.resolve()
    if out == FORBIDDEN or FORBIDDEN in out.parents:
        raise SystemExit(
            f"REFUSED: --out is inside the historical root {FORBIDDEN}. "
            f"Axis T must never mutate the dataset that stage55b/s62 were trained on.")
    if out.exists():
        raise SystemExit(f"REFUSED: --out already exists: {out}")

    real_ids = [l.strip() for l in args.real_ids.read_text().split() if l.strip()]
    pm = sorted(args.pseudo_masks.glob("*.png"))
    if len(pm) != 500:
        raise SystemExit(f"expected 500 pseudo masks, found {len(pm)}")
    if len(real_ids) != 5597:
        raise SystemExit(f"expected 5597 real ids, found {len(real_ids)}")

    (out / "train/images").mkdir(parents=True)
    (out / "train/masks").mkdir(parents=True)
    src_img = args.real_root / "train/images"
    src_msk = args.real_root / "train/masks"
    test_img = args.test_root / "images"

    for stem in real_ids:
        for d, base in ((out / "train/images", src_img), (out / "train/masks", src_msk)):
            s = base / f"{stem}.png"
            if not s.is_file():
                raise SystemExit(f"missing official file: {s}")
            os.symlink(s, d / f"{stem}.png")

    pseudo_ids = []
    for i in range(1, 501):
        pid = f"{7000 + i - 1:04d}"
        ti = test_img / f"test1_{i}.png"
        if not ti.is_file():
            raise SystemExit(f"missing test image: {ti}")
        os.symlink(ti, out / "train/images" / f"{pid}.png")
        os.symlink(pm[i - 1].resolve(), out / "train/masks" / f"{pid}.png")
        pseudo_ids.append(pid)

    split = out / "train_ids_5597pseudo.txt"
    split.write_text("\n".join(real_ids + pseudo_ids) + "\n", encoding="utf-8")

    imgs = sorted(p.name for p in (out / "train/images").glob("*.png"))
    msks = sorted(p.name for p in (out / "train/masks").glob("*.png"))
    if imgs != msks:
        raise SystemExit("image/mask filename mismatch")

    rec = {
        "schema": "AXIS_T_DATASET_V1",
        "created": datetime.now(timezone.utc).isoformat(),
        "label_tag": args.label_tag,
        "out": str(out),
        "real_root": str(args.real_root),
        "real_ids_file": str(args.real_ids),
        "real_ids_file_sha256": sha256_file(args.real_ids),
        "test_root": str(args.test_root),
        "pseudo_masks_dir": str(args.pseudo_masks),
        "teacher_sha256": args.teacher_sha256,
        "real_count": len(real_ids),
        "pseudo_count": len(pseudo_ids),
        "total": len(real_ids) + len(pseudo_ids),
        "pseudo_index_range": [pseudo_ids[0], pseudo_ids[-1]],
        "split_file": str(split),
        "split_file_sha256": sha256_file(split),
        "pseudo_fraction": round(len(pseudo_ids) / (len(real_ids) + len(pseudo_ids)), 6),
        "note": ("Shape-identical to the historical pseudo_train_6796 used by stage55b "
                 "(single uniform pool, no cadence). The historical root was NOT touched."),
    }
    (out / "AXIS_T_DATASET.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
