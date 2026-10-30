#!/usr/bin/env python3
"""Create one immutable, deterministic multi-label-presence train/val split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_yaml, read_stems, save_json, utc_now  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "splits")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "split_config.yaml")
    parser.add_argument(
        "--fixed-quick-val",
        type=Path,
        default=PROJECT_ROOT / "splits" / "fixed_quick_val.txt",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def stem_fingerprint(stems: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(stems)).encode("utf-8")).hexdigest()


def inspect_mask(item: tuple[str, Path]) -> tuple[str, int, dict[int, int]]:
    stem, path = item
    with Image.open(path) as image:
        if image.mode not in {"L", "P"}:
            array = image.convert("L")
        else:
            array = image
        colors = array.getcolors(maxcolors=256)
    if colors is None:
        raise RuntimeError(f"could not enumerate label values: {path}")
    counts = {int(value): int(count) for count, value in colors}
    illegal = set(counts) - set(range(9))
    if illegal:
        raise ValueError(f"illegal official IDs in {path}: {sorted(illegal)}")
    combination = 0
    for official_id in range(1, 9):
        if counts.get(official_id, 0) > 0:
            combination |= 1 << (official_id - 1)
    return stem, combination, counts


def create_partition(records: list[tuple[str, int, dict[int, int]]], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for stem, combination, _ in records:
        groups[combination].append(stem)
    target_val = int(round(len(records) * val_fraction))
    quotas: dict[int, int] = {}
    remainders: list[tuple[float, int]] = []
    for combination, stems in groups.items():
        exact = len(stems) * val_fraction
        quotas[combination] = math.floor(exact)
        remainders.append((exact - math.floor(exact), combination))
    remaining = target_val - sum(quotas.values())
    for _, combination in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if quotas[combination] < len(groups[combination]):
            quotas[combination] += 1
            remaining -= 1
    if remaining != 0:
        raise RuntimeError(f"could not allocate exact validation size, remaining={remaining}")

    val: set[str] = set()
    for combination, stems in sorted(groups.items()):
        shuffled = sorted(stems)
        random.Random(seed + combination * 1_000_003).shuffle(shuffled)
        val.update(shuffled[: quotas[combination]])
    all_stems = {record[0] for record in records}
    train = sorted(all_stems - val)
    return train, sorted(val)


def aggregate(stems: list[str], by_stem: dict[str, tuple[int, dict[int, int]]]) -> dict[str, Any]:
    pixels = {str(index): 0 for index in range(9)}
    image_counts = {str(index): 0 for index in range(9)}
    combinations: dict[str, int] = defaultdict(int)
    for stem in stems:
        combination, counts = by_stem[stem]
        combinations[str(combination)] += 1
        for class_id in range(9):
            count = int(counts.get(class_id, 0))
            pixels[str(class_id)] += count
            image_counts[str(class_id)] += int(count > 0)
    return {
        "sample_count": len(stems),
        "pixel_counts_official": pixels,
        "images_with_official_class": image_counts,
        "presence_combination_counts": dict(sorted(combinations.items(), key=lambda item: int(item[0]))),
    }


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def ensure_fixed_quick_val(path: Path, val: list[str], count: int, seed: int) -> list[str]:
    if not path.is_file():
        raise RuntimeError(
            f"fixed quick validation list is missing: {path}. Restore the repository file before running split checks."
        )
    existing = read_stems(path)
    if len(existing) != min(count, len(val)) or not set(existing).issubset(set(val)):
        raise RuntimeError(f"existing fixed quick validation list is incompatible; refusing to overwrite: {path}")
    return existing


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    config = load_yaml(args.config.resolve())
    seed = int(config.get("seed", 42))
    val_fraction = float(config.get("val_fraction", 0.2))
    quick_val_count = int(config.get("fixed_quick_val_count", 256))
    image_dir = dataset_root / "train" / "images"
    mask_dir = dataset_root / "train" / "masks"
    image_stems = {path.stem for path in image_dir.glob("*.png")}
    mask_stems = {path.stem for path in mask_dir.glob("*.png")}
    if image_stems != mask_stems or not image_stems:
        raise RuntimeError(
            f"image/mask stem mismatch: images={len(image_stems)} masks={len(mask_stems)} "
            f"image_only={sorted(image_stems-mask_stems)[:10]} mask_only={sorted(mask_stems-image_stems)[:10]}"
        )
    all_stems = sorted(image_stems)
    fingerprint = stem_fingerprint(all_stems)
    train_path, val_path = output_dir / "train.txt", output_dir / "val.txt"
    stats_path = output_dir / "split_statistics.json"
    existing = [train_path.is_file(), val_path.is_file(), stats_path.is_file()]
    if any(existing):
        if not all(existing):
            raise RuntimeError("partial split artifacts exist; refusing to overwrite")
        train, val = read_stems(train_path), read_stems(val_path)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if set(train) & set(val) or set(train) | set(val) != set(all_stems):
            raise RuntimeError("existing split does not exactly cover the current dataset; refusing to overwrite")
        if stats.get("dataset_stem_sha256") != fingerprint or int(stats.get("seed", -1)) != seed:
            raise RuntimeError("existing split metadata differs from current dataset/config; refusing to overwrite")
        fixed = ensure_fixed_quick_val(args.fixed_quick_val.resolve(), val, quick_val_count, seed)
        print(f"existing split preserved: train={len(train)} val={len(val)} fixed_quick_val={len(fixed)}")
        return 0

    items = [(stem, mask_dir / f"{stem}.png") for stem in all_stems]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(inspect_mask, items))
    train, val = create_partition(records, val_fraction, seed)
    by_stem = {stem: (combination, counts) for stem, combination, counts in records}
    stats = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "strategy": "stratified_presence_combination_largest_remainder",
        "seed": seed,
        "val_fraction": val_fraction,
        "dataset_sample_count": len(all_stems),
        "dataset_stem_sha256": fingerprint,
        "train": aggregate(train, by_stem),
        "val": aggregate(val, by_stem),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(train_path, train)
    write_lines(val_path, val)
    save_json(stats_path, stats)
    fixed = ensure_fixed_quick_val(args.fixed_quick_val.resolve(), val, quick_val_count, seed)
    report_lines = [
        "# Fixed train/validation split report",
        "",
        "This split is immutable unless the dataset or split policy is deliberately versioned.",
        "",
        f"- strategy: `{stats['strategy']}`",
        f"- seed: {seed}",
        f"- train: {len(train)}",
        f"- validation: {len(val)}",
        f"- fixed quick validation subset: {len(fixed)}",
        f"- dataset stem SHA256: `{fingerprint}`",
        "",
        "| Official ID | Train images | Validation images |",
        "|---:|---:|---:|",
    ]
    for class_id in range(1, 9):
        report_lines.append(
            f"| {class_id} | {stats['train']['images_with_official_class'][str(class_id)]} | "
            f"{stats['val']['images_with_official_class'][str(class_id)]} |"
        )
    (output_dir / "split_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"created split: train={len(train)} val={len(val)} fixed_quick_val={len(fixed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
