#!/usr/bin/env python3
"""Fail-fast integrity check for the immutable split and fixed quick subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_stems  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=PROJECT_ROOT / "splits")
    parser.add_argument(
        "--fixed-quick-val",
        type=Path,
        default=PROJECT_ROOT / "splits" / "fixed_quick_val.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    split_dir = args.split_dir.resolve()
    image_dir, mask_dir = dataset_root / "train" / "images", dataset_root / "train" / "masks"
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    mask_stems = {path.stem for path in mask_dir.glob("*.png")}
    train, val = read_stems(split_dir / "train.txt"), read_stems(split_dir / "val.txt")
    train_set, val_set = set(train), set(val)
    errors: list[str] = []
    if train_set & val_set:
        errors.append(f"train/val overlap: {len(train_set & val_set)}")
    if len(train) + len(val) != 6996:
        errors.append(f"expected 6996 stems, got {len(train) + len(val)}")
    if image_stems != mask_stems:
        errors.append(f"dataset image/mask mismatch: {len(image_stems)} vs {len(mask_stems)}")
    if train_set | val_set != image_stems:
        errors.append(
            f"split coverage mismatch: missing={len(image_stems-(train_set|val_set))} unknown={len((train_set|val_set)-image_stems)}"
        )
    for stem in train + val:
        if not (image_dir / f"{stem}.png").is_file() or not (mask_dir / f"{stem}.png").is_file():
            errors.append(f"missing pair: {stem}")
            break
    stats = json.loads((split_dir / "split_statistics.json").read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256("\n".join(sorted(image_stems)).encode("utf-8")).hexdigest()
    if stats.get("dataset_stem_sha256") != fingerprint:
        errors.append("dataset fingerprint differs from split metadata")
    for split_name in ("train", "val"):
        presence = stats[split_name]["images_with_official_class"]
        missing_classes = [class_id for class_id in range(1, 9) if int(presence[str(class_id)]) <= 0]
        if missing_classes:
            errors.append(f"{split_name} is missing official classes {missing_classes}")
    val_presence = stats["val"]["images_with_official_class"]
    if int(val_presence["8"]) < 50 or int(val_presence["5"]) < 50:
        errors.append(
            f"validation rare-class coverage is too small: Vehicle={val_presence['8']} Barren={val_presence['5']}"
        )
    fixed = read_stems(args.fixed_quick_val.resolve())
    fixed_set = set(fixed)
    if len(fixed) != 256:
        errors.append(f"expected 256 fixed quick validation stems, got {len(fixed)}")
    if not fixed_set.issubset(val_set):
        errors.append("fixed quick validation subset contains stems outside val.txt")
    if fixed_set & train_set:
        errors.append("fixed quick validation subset contains stems from train.txt")
    if errors:
        raise RuntimeError("split check failed:\n- " + "\n- ".join(errors))
    print(
        f"split check passed: train={len(train)} val={len(val)} fixed_quick_val={len(fixed)} "
        f"Vehicle_val={val_presence['8']} Barren_val={val_presence['5']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
