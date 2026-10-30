#!/usr/bin/env python3
"""Read-only inspection for an aerial semantic-segmentation dataset.

The script never writes below --dataset-root. All artifacts are written below
--output-dir. Labels are read in their native Pillow mode and converted directly
to NumPy arrays, without RGB conversion, interpolation, or normalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_CONFIG_EXTENSIONS = {".md", ".txt", ".json", ".csv", ".yaml", ".yml"}
SKIP_DIR_NAMES = {".git", ".venv", "venv", "node_modules", "runs", "__pycache__"}
THEORETICAL_SIZE = (1024, 1024)
THEORETICAL_CLASS_IDS = set(range(9))

# Visualization only. These colors are not claimed to be an official palette.
VISUAL_PALETTE = {
    0: (0, 0, 0),
    1: (128, 128, 128),
    2: (220, 20, 60),
    3: (255, 215, 0),
    4: (30, 144, 255),
    5: (210, 180, 140),
    6: (34, 139, 34),
    7: (154, 205, 50),
    8: (255, 105, 180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict read-only dataset inspection")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute-hash", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def is_skipped(path: Path) -> bool:
    return any(part.lower() in SKIP_DIR_NAMES for part in path.parts)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and not is_skipped(path.relative_to(root)):
            yield path


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not is_skipped(p.relative_to(root))),
        key=lambda p: str(p).casefold(),
    )


def relative_text(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def directory_file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and not is_skipped(p.relative_to(path)))


def discover_layout(root: Path) -> dict[str, Any]:
    all_dirs = [root] + [p for p in root.rglob("*") if p.is_dir() and not is_skipped(p.relative_to(root))]
    examples = [p for p in all_dirs if any(x.lower() in {"example", "examples", "sample", "samples"} for x in p.parts)]

    exact_train_images = root / "train" / "images"
    exact_train_masks = root / "train" / "masks"
    exact_test_images = root / "images"

    def not_example(path: Path) -> bool:
        lowered = {part.lower() for part in path.parts}
        return not bool(lowered & {"example", "examples", "sample", "samples"})

    train_image_candidates: list[Path] = []
    train_label_candidates: list[Path] = []
    test_image_candidates: list[Path] = []
    for directory in all_dirs:
        if not not_example(directory):
            continue
        parts = [x.lower() for x in directory.relative_to(root).parts]
        leaf = directory.name.lower()
        has_images = any(p.suffix.lower() in IMAGE_EXTENSIONS for p in directory.glob("*"))
        if not has_images:
            continue
        train_context = any("train" in x for x in parts)
        test_context = any("test" in x for x in parts)
        if train_context and leaf in {"image", "images", "img", "imgs"}:
            train_image_candidates.append(directory)
        if train_context and leaf in {"mask", "masks", "label", "labels", "annotation", "annotations"}:
            train_label_candidates.append(directory)
        if test_context and leaf in {"image", "images", "img", "imgs", "test"}:
            test_image_candidates.append(directory)

    if exact_train_images.is_dir():
        train_image_candidates = [exact_train_images]
    if exact_train_masks.is_dir():
        train_label_candidates = [exact_train_masks]
    if exact_test_images.is_dir() and exact_test_images != exact_train_images:
        test_image_candidates = [exact_test_images]

    def unique(paths: list[Path]) -> list[Path]:
        result: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = str(path.resolve()).casefold()
            if key not in seen:
                seen.add(key)
                result.append(path.resolve())
        return sorted(result, key=lambda p: str(p).casefold())

    train_image_candidates = unique(train_image_candidates)
    train_label_candidates = unique(train_label_candidates)
    test_image_candidates = unique(test_image_candidates)
    if len(train_image_candidates) != 1 or len(train_label_candidates) != 1:
        raise RuntimeError(
            "Could not select a unique train image/mask pair. "
            f"train image candidates={train_image_candidates}; train label candidates={train_label_candidates}"
        )
    if len(test_image_candidates) > 1:
        raise RuntimeError(f"Multiple test image candidates found; refusing to choose: {test_image_candidates}")

    official_files = [
        p.resolve()
        for p in iter_files(root)
        if p.suffix.lower() in TEXT_CONFIG_EXTENSIONS
        and not any(x.lower() in {"image", "images", "mask", "masks"} for x in p.relative_to(root).parts[:-1])
    ]
    submission_candidates = [
        p.resolve()
        for p in iter_files(root)
        if re.search(r"(?i)(submit|submission|提交)", p.name)
    ]

    return {
        "train_image_candidates": [str(p) for p in train_image_candidates],
        "train_label_candidates": [str(p) for p in train_label_candidates],
        "test_image_candidates": [str(p) for p in test_image_candidates],
        "example_directories": [str(p.resolve()) for p in unique(examples) if p != root],
        "official_files": [str(p) for p in sorted(official_files, key=lambda x: str(x).casefold())],
        "submission_candidates": [str(p) for p in sorted(submission_candidates, key=lambda x: str(x).casefold())],
        "selected_train_images": str(train_image_candidates[0]),
        "selected_train_labels": str(train_label_candidates[0]),
        "selected_test_images": str(test_image_candidates[0]) if test_image_candidates else None,
    }


def safe_read_text(path: Path) -> tuple[str, str | None]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding), None
        except UnicodeDecodeError:
            continue
    return "", "Unable to decode as utf-8-sig, utf-8, gb18030, or utf-16"


def parse_official_files(paths: list[Path], dataset_root: Path) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    mappings: list[dict[int, str]] = []
    ignore_mentions: set[int] = set()
    background_mentions: set[int] = set()
    for path in paths:
        text, error = safe_read_text(path)
        entry: dict[str, Any] = {
            "path": str(path),
            "relative_path": relative_text(path, dataset_root),
            "size_bytes": path.stat().st_size,
            "read_error": error,
        }
        if error is None:
            mapping = {
                int(class_id): name.strip()
                for class_id, name in re.findall(r"(?m)(\d+)\s*:\s*[\"']([^\"']+)[\"']", text)
            }
            if mapping:
                mappings.append(mapping)
                entry["parsed_class_mapping"] = {str(k): v for k, v in sorted(mapping.items())}
                for class_id, name in mapping.items():
                    normalized = name.casefold()
                    if "ignore" in normalized or "void" in normalized:
                        ignore_mentions.add(class_id)
                    if "background" in normalized or normalized in {"bg", "背景"}:
                        background_mentions.add(class_id)
            explicit_ignore = re.findall(r"(?i)ignore[_\s-]*(?:index|id)?\s*[:=]\s*(\d+)", text)
            ignore_mentions.update(int(x) for x in explicit_ignore)
            entry["content"] = text.strip()
        summaries.append(entry)

    mapping_conflict = False
    confirmed_mapping: dict[int, str] = {}
    if mappings:
        confirmed_mapping = mappings[0]
        mapping_conflict = any(mapping != confirmed_mapping for mapping in mappings[1:])
    mapping_confirmed = bool(confirmed_mapping) and not mapping_conflict
    ignore_confirmed = len(ignore_mentions) == 1 and not mapping_conflict
    background_confirmed = len(background_mentions) == 1 and not mapping_conflict
    return {
        "files": summaries,
        "class_mapping_confirmed": mapping_confirmed,
        "class_mapping": {str(k): v for k, v in sorted(confirmed_mapping.items())} if mapping_confirmed else {},
        "mapping_conflict": mapping_conflict,
        "ignore_index_confirmed": ignore_confirmed,
        "ignore_index": sorted(ignore_mentions)[0] if ignore_confirmed else None,
        "background_class_confirmed": background_confirmed,
        "background_class_id": sorted(background_mentions)[0] if background_confirmed else None,
        "official_palette_confirmed": False,
        "official_palette": None,
        "filename_rule_confirmed": False,
        "split_rule_confirmed": False,
        "submission_format_confirmed_from_files": False,
    }


def channels_for_array(array: np.ndarray) -> int:
    if array.ndim == 2:
        return 1
    if array.ndim == 3:
        return int(array.shape[2])
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    small = image.convert("L").resize((17, 16), Image.Resampling.BILINEAR)
    values = np.asarray(small, dtype=np.int16)
    bits = values[:, 1:] >= values[:, :-1]
    packed = np.packbits(bits.reshape(-1))
    return packed.tobytes().hex()


def inspect_input_image(path: Path, compute_hash: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size, "error": None}
    if result["size_bytes"] == 0:
        result["error"] = "empty_file"
        return result
    try:
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
            result.update(
                {
                    "format": image.format,
                    "mode": image.mode,
                    "width": int(image.width),
                    "height": int(image.height),
                    "channels": channels_for_array(array),
                    "dtype": str(array.dtype),
                    "min": float(np.min(array)),
                    "max": float(np.max(array)),
                    "mean": float(np.mean(array, dtype=np.float64)),
                    "std": float(np.std(array, dtype=np.float64)),
                    "near_constant": bool(float(np.std(array, dtype=np.float64)) < 1.0),
                    "all_black": bool(float(np.max(array)) <= 1.0),
                    "all_white": bool(float(np.min(array)) >= 254.0 and np.issubdtype(array.dtype, np.integer)),
                    "per_channel_mean": (
                        [float(x) for x in np.mean(array, axis=(0, 1), dtype=np.float64)]
                        if array.ndim == 3
                        else [float(np.mean(array, dtype=np.float64))]
                    ),
                    "dhash": difference_hash(image),
                }
            )
        if compute_hash:
            result["sha256"] = sha256_file(path)
    except Exception as exc:  # report corrupt/unsupported inputs without stopping the audit
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_label(path: Path, compute_hash: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size, "error": None}
    if result["size_bytes"] == 0:
        result["error"] = "empty_file"
        return result
    try:
        with Image.open(path) as image:
            image.load()
            # Critical: no convert(), resize(), interpolation, or normalization.
            array = np.asarray(image)
            unique_values, counts = np.unique(array, return_counts=True)
            result.update(
                {
                    "format": image.format,
                    "mode": image.mode,
                    "width": int(image.width),
                    "height": int(image.height),
                    "channels": channels_for_array(array),
                    "dtype": str(array.dtype),
                    "has_palette": image.mode == "P" or image.getpalette() is not None,
                    "min": float(np.min(array)),
                    "max": float(np.max(array)),
                    "unique_values": [int(x) for x in unique_values.tolist()],
                    "value_counts": {str(int(k)): int(v) for k, v in zip(unique_values.tolist(), counts.tolist())},
                    "single_value": len(unique_values) == 1,
                }
            )
        if compute_hash:
            result["sha256"] = sha256_file(path)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def parallel_inspect(
    paths: list[Path], worker: Callable[[Path, bool], dict[str, Any]], compute_hash: bool, workers: int, label: str
) -> list[dict[str, Any]]:
    if not paths:
        return []
    results: list[dict[str, Any]] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(worker, path, compute_hash): path for path in paths}
        for future in as_completed(futures):
            path = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"path": str(path), "size_bytes": path.stat().st_size, "error": f"worker_failure: {exc}"})
            completed += 1
            if completed % 250 == 0 or completed == len(paths):
                print(f"[{label}] {completed}/{len(paths)}", flush=True)
    return sorted(results, key=lambda x: x["path"].casefold())


def distribution(results: list[dict[str, Any]], key: str, formatter: Callable[[Any], str] = str) -> dict[str, int]:
    counter = Counter(formatter(item[key]) for item in results if item.get("error") is None and key in item)
    return dict(sorted(counter.items(), key=lambda pair: pair[0]))


def stem_index(paths: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[path.stem.casefold()].append(path)
    return index


def duplicate_groups(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[str]] = defaultdict(list)
    for result in results:
        value = result.get(key)
        if result.get("error") is None and value is not None:
            groups[value].append(result["path"])
    return [
        {key: value, "count": len(paths), "files": sorted(paths, key=str.casefold)}
        for value, paths in sorted(groups.items(), key=lambda pair: str(pair[0]))
        if len(paths) > 1
    ]


def cross_duplicate_groups(
    train_results: list[dict[str, Any]], test_results: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    train_map: dict[Any, list[str]] = defaultdict(list)
    test_map: dict[Any, list[str]] = defaultdict(list)
    for result in train_results:
        if result.get("error") is None and result.get(key) is not None:
            train_map[result[key]].append(result["path"])
    for result in test_results:
        if result.get("error") is None and result.get(key) is not None:
            test_map[result[key]].append(result["path"])
    return [
        {
            key: value,
            "train_files": sorted(train_map[value], key=str.casefold),
            "test_files": sorted(test_map[value], key=str.casefold),
        }
        for value in sorted(set(train_map) & set(test_map), key=str)
    ]


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q1": None, "median": None, "q3": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q1": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q3": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def numeric_prefix(path: Path) -> str:
    stem = path.stem
    match = re.match(r"([^0-9]+)", stem)
    return match.group(1) if match else "[numeric-only]"


def tree_summary(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directories = [root] + [p for p in root.rglob("*") if p.is_dir() and not is_skipped(p.relative_to(root))]
    for directory in sorted(directories, key=lambda p: str(p).casefold()):
        direct = [p for p in directory.iterdir() if p.is_file()]
        rows.append(
            {
                "path": "." if directory == root else relative_text(directory, root),
                "direct_file_count": len(direct),
                "direct_size_bytes": int(sum(p.stat().st_size for p in direct)),
            }
        )
    return rows


def build_abnormal_rows(
    train_images: list[dict[str, Any]],
    test_images: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    missing_labels: list[Path],
    orphan_labels: list[Path],
    image_stem_conflicts: dict[str, list[Path]],
    label_stem_conflicts: dict[str, list[Path]],
    image_by_stem: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(severity: str, split: str, path: str | Path, issue: str, detail: str) -> None:
        rows.append({"severity": severity, "split": split, "file": str(path), "issue": issue, "detail": detail})

    for split, results in (("train_image", train_images), ("test_image", test_images)):
        for result in results:
            if result.get("error"):
                add("error", split, result["path"], "unreadable_or_empty", result["error"])
                continue
            if (result["width"], result["height"]) != THEORETICAL_SIZE:
                add("error", split, result["path"], "unexpected_size", f"{result['width']}x{result['height']}")
            if result["channels"] != 3:
                add("warning", split, result["path"], "unexpected_channels", f"mode={result['mode']}, channels={result['channels']}")
            if result["all_black"]:
                add("warning", split, result["path"], "all_black", f"range={result['min']}..{result['max']}")
            elif result["all_white"]:
                add("warning", split, result["path"], "all_white", f"range={result['min']}..{result['max']}")
            elif result["near_constant"]:
                add("warning", split, result["path"], "near_constant", f"std={result['std']:.6f}")

    for result in labels:
        if result.get("error"):
            add("error", "train_label", result["path"], "unreadable_or_empty", result["error"])
            continue
        if result["channels"] != 1:
            add("error", "train_label", result["path"], "label_not_single_channel", f"mode={result['mode']}, channels={result['channels']}")
        invalid = sorted(set(result["unique_values"]) - THEORETICAL_CLASS_IDS)
        if invalid:
            add("error", "train_label", result["path"], "class_id_outside_0_8", str(invalid))
        if result["single_value"]:
            add("warning", "train_label", result["path"], "single_class_label", str(result["unique_values"]))
        image = image_by_stem.get(Path(result["path"]).stem.casefold())
        if image and image.get("error") is None and (image["width"], image["height"]) != (result["width"], result["height"]):
            add(
                "error",
                "pair",
                result["path"],
                "image_label_size_mismatch",
                f"image={image['width']}x{image['height']}, label={result['width']}x{result['height']}",
            )

    for path in missing_labels:
        add("error", "pair", path, "missing_label", "No label with the same case-insensitive stem")
    for path in orphan_labels:
        add("error", "pair", path, "orphan_label", "No train image with the same case-insensitive stem")
    for stem, paths in image_stem_conflicts.items():
        add("error", "train_image", " | ".join(map(str, paths)), "stem_conflict", stem)
    for stem, paths in label_stem_conflicts.items():
        add("error", "train_label", " | ".join(map(str, paths)), "stem_conflict", stem)
    return sorted(rows, key=lambda row: (row["severity"], row["split"], row["file"].casefold(), row["issue"]))


def analyze_classes(
    label_results: list[dict[str, Any]], official: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    good = [x for x in label_results if x.get("error") is None and x.get("channels") == 1]
    class_ids = sorted({int(k) for item in good for k in item["value_counts"]})
    image_count = len(good)
    total_all_pixels = sum(sum(item["value_counts"].values()) for item in good)
    ignore_id = official["ignore_index"] if official["ignore_index_confirmed"] else None
    background_id = official["background_class_id"] if official["background_class_confirmed"] else None
    class_mapping = {int(k): v for k, v in official["class_mapping"].items()}
    total_by_class = {class_id: sum(item["value_counts"].get(str(class_id), 0) for item in good) for class_id in class_ids}
    valid_pixel_count = total_all_pixels - (total_by_class.get(ignore_id, 0) if ignore_id is not None else 0)

    rows: list[dict[str, Any]] = []
    for class_id in class_ids:
        per_image = []
        present_proportions = []
        present_count = 0
        for item in good:
            denominator = sum(item["value_counts"].values())
            proportion = item["value_counts"].get(str(class_id), 0) / denominator if denominator else 0.0
            per_image.append(proportion)
            if proportion > 0:
                present_count += 1
                present_proportions.append(proportion)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_mapping.get(class_id, ""),
                "is_ignore": class_id == ignore_id,
                "total_pixels": total_by_class[class_id],
                "pixel_ratio_all_pixels": total_by_class[class_id] / total_all_pixels if total_all_pixels else 0.0,
                "pixel_ratio_valid_pixels": (
                    None if class_id == ignore_id else total_by_class[class_id] / valid_pixel_count if valid_pixel_count else None
                ),
                "images_with_class": present_count,
                "image_frequency": present_count / image_count if image_count else 0.0,
                "mean_image_pixel_ratio": statistics.fmean(per_image) if per_image else 0.0,
                "median_image_pixel_ratio": statistics.median(per_image) if per_image else 0.0,
                "max_image_pixel_ratio": max(per_image, default=0.0),
                "present_images_below_0_1_percent": sum(0 < x < 0.001 for x in present_proportions),
                "present_images_below_1_percent": sum(0 < x < 0.01 for x in present_proportions),
                "present_images_below_5_percent": sum(0 < x < 0.05 for x in present_proportions),
                "images_without_class": image_count - present_count,
            }
        )

    valid_class_totals = [total for class_id, total in total_by_class.items() if class_id != ignore_id and total > 0]
    imbalance_ratio = max(valid_class_totals) / min(valid_class_totals) if valid_class_totals else None
    combo_counter: Counter[tuple[int, ...]] = Counter()
    class_count_counter: Counter[int] = Counter()
    all_class_count_counter: Counter[int] = Counter()
    dominant_over = {"50_percent": 0, "80_percent": 0, "95_percent": 0}
    background_over = {"50_percent": 0, "80_percent": 0, "95_percent": 0}
    small_target_images: dict[int, int] = {class_id: 0 for class_id in class_ids if class_id != ignore_id}
    for item in good:
        counts = {int(k): v for k, v in item["value_counts"].items()}
        total = sum(counts.values())
        valid_ids = tuple(sorted(k for k, value in counts.items() if value > 0 and k != ignore_id))
        all_ids = tuple(sorted(k for k, value in counts.items() if value > 0))
        combo_counter[valid_ids] += 1
        class_count_counter[len(valid_ids)] += 1
        all_class_count_counter[len(all_ids)] += 1
        dominant = max(counts.values(), default=0) / total if total else 0.0
        for threshold, name in ((0.5, "50_percent"), (0.8, "80_percent"), (0.95, "95_percent")):
            dominant_over[name] += int(dominant > threshold)
            if background_id is not None:
                background_over[name] += int(counts.get(background_id, 0) / total > threshold)
        for class_id, count in counts.items():
            proportion = count / total if total else 0.0
            if class_id != ignore_id and 0 < proportion < 0.001:
                small_target_images[class_id] = small_target_images.get(class_id, 0) + 1

    analysis = {
        "total_label_images_analyzed": image_count,
        "total_all_pixels": total_all_pixels,
        "total_valid_pixels": valid_pixel_count,
        "valid_pixel_definition": f"all pixels except confirmed ignore_index={ignore_id}" if ignore_id is not None else "all pixels; ignore_index unconfirmed",
        "pixel_level_imbalance_ratio_max_to_min_non_ignore": imbalance_ratio,
        "class_count_per_image_distribution_excluding_ignore": {str(k): v for k, v in sorted(class_count_counter.items())},
        "class_count_per_image_distribution_including_ignore": {str(k): v for k, v in sorted(all_class_count_counter.items())},
        "top_class_combinations_excluding_ignore": [
            {"class_ids": list(combo), "image_count": count, "image_frequency": count / image_count if image_count else 0.0}
            for combo, count in combo_counter.most_common(20)
        ],
        "dominant_single_class_image_counts": dominant_over,
        "background_high_ratio_image_counts": background_over if background_id is not None else None,
        "extremely_small_target_candidate_images_below_0_1_percent": {str(k): v for k, v in sorted(small_target_images.items())},
    }
    return rows, analysis


def make_visual_samples(
    pairs: list[tuple[Path, Path]], output_dir: Path, sample_count: int, seed: int, official: dict[str, Any]
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("sample_*.png"):
        old.unlink()
    rng = random.Random(seed)
    selected = rng.sample(pairs, min(max(20, sample_count), len(pairs))) if pairs else []
    class_mapping = {int(k): v for k, v in official["class_mapping"].items()}
    written: list[str] = []
    pane = 480
    for index, (image_path, label_path) in enumerate(selected, 1):
        with Image.open(image_path) as source_image:
            rgb = source_image.convert("RGB")
            original = ImageOps.fit(rgb, (pane, pane), method=Image.Resampling.LANCZOS)
        with Image.open(label_path) as label_image:
            label_array = np.asarray(label_image)
        if label_array.ndim != 2:
            continue
        ids, counts = np.unique(label_array, return_counts=True)
        maximum = max(int(np.max(label_array)), 1)
        raw_display = Image.fromarray(np.round(label_array.astype(np.float32) * (255.0 / maximum)).astype(np.uint8), mode="L").convert("RGB")
        raw_display = raw_display.resize((pane, pane), Image.Resampling.NEAREST)
        color = np.zeros((*label_array.shape, 3), dtype=np.uint8)
        for class_id in ids.tolist():
            color[label_array == class_id] = VISUAL_PALETTE.get(int(class_id), ((int(class_id) * 53) % 256, (int(class_id) * 97) % 256, (int(class_id) * 193) % 256))
        pseudo = Image.fromarray(color, mode="RGB").resize((pane, pane), Image.Resampling.NEAREST)
        overlay = Image.blend(original, pseudo, alpha=0.45)
        canvas = Image.new("RGB", (pane * 2, pane * 2 + 120), "white")
        canvas.paste(original, (0, 0))
        canvas.paste(raw_display, (pane, 0))
        canvas.paste(pseudo, (0, pane))
        canvas.paste(overlay, (pane, pane))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 150, 20), fill="white")
        draw.text((5, 4), "Original thumbnail", fill="black")
        draw.rectangle((pane, 0, pane + 260, 20), fill="white")
        draw.text((pane + 5, 4), "Raw IDs (display-scaled only)", fill="black")
        draw.rectangle((0, pane, 135, pane + 20), fill="white")
        draw.text((5, pane + 4), "Pseudo-color", fill="black")
        draw.rectangle((pane, pane, pane + 95, pane + 20), fill="white")
        draw.text((pane + 5, pane + 4), "Overlay", fill="black")
        total = int(np.sum(counts))
        descriptions = []
        for class_id, count in zip(ids.tolist(), counts.tolist()):
            name = class_mapping.get(int(class_id), "")
            name_text = f" {name}" if name else ""
            descriptions.append(f"{int(class_id)}{name_text}: {count / total:.4%}")
        draw.text((8, pane * 2 + 8), f"stem={image_path.stem} | class_ids={list(map(int, ids.tolist()))}", fill="black")
        for line_index in range(0, len(descriptions), 3):
            draw.text((8, pane * 2 + 32 + 20 * (line_index // 3)), " | ".join(descriptions[line_index : line_index + 3]), fill="black")
        output_path = output_dir / f"sample_{index:02d}_{image_path.stem}.png"
        canvas.save(output_path, format="PNG", optimize=True)
        written.append(str(output_path.resolve()))
    return written


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6%}"


def render_markdown(report: dict[str, Any], class_rows: list[dict[str, Any]], abnormalities: list[dict[str, str]]) -> str:
    counts = report["counts"]
    official = report["official_documentation"]
    labels = report["label_inspection"]
    duplicates = report["duplicate_inspection"]
    classes = report["class_distribution"]
    mapping_status = "已确认" if official["class_mapping_confirmed"] else "类别名称与 ID 映射尚未确认"
    ignore_status = f"已确认：{official['ignore_index']}" if official["ignore_index_confirmed"] else "尚未确认"
    severity = Counter(row["severity"] for row in abnormalities)
    lines = [
        "# 无人机低空航拍语义分割数据集只读体检报告",
        "",
        f"- 生成时间（UTC）：{report['generated_at_utc']}",
        f"- 数据集根目录：{report['dataset_root']}",
        f"- 检查模式：全量解码；完整 SHA256={'开启' if report['parameters']['compute_hash'] else '关闭'}",
        f"- 原始数据写入：无（报告输出目录位于 {report['output_dir']}）",
        "",
        "## 1. 数据集实际目录结构",
        "",
        "| 路径 | 直属文件数 | 直属文件字节数 |",
        "|---|---:|---:|",
    ]
    for row in report["directory_tree"]:
        lines.append(f"| {row['path']} | {row['direct_file_count']} | {row['direct_size_bytes']} |")
    lines.extend(
        [
            "",
            "## 2. 官方说明文件摘要",
            "",
            f"- 发现说明/配置文件：{len(official['files'])} 个。",
            f"- 类别映射状态：{mapping_status}。",
            f"- ignore_index：{ignore_status}。",
            f"- 背景类别：{official['background_class_id'] if official['background_class_confirmed'] else '尚未确认'}。",
            "- 官方调色板：未在数据包说明中发现；可视化使用的颜色仅用于查看。",
            "- 文件命名规则、官方训练/验证划分、提交格式：数据包内未发现明确文字说明。根目录 images 依据目录结构和 test1_ 文件名仅识别为测试候选。",
            "",
        ]
    )
    if official["class_mapping_confirmed"]:
        lines.extend(["| class_id | 官方名称 |", "|---:|---|"])
        for class_id, name in official["class_mapping"].items():
            lines.append(f"| {class_id} | {name} |")
        lines.append("")
    lines.extend(
        [
            "## 3. 数量与配对",
            "",
            f"- 训练图像：{counts['train_images']} 张；训练标签：{counts['train_labels']} 张；测试图像：{counts['test_images']} 张。",
            f"- 缺少标签的训练图像：{counts['missing_labels']}；孤立标签：{counts['orphan_labels']}。",
            f"- 训练图像 stem 冲突：{counts['train_image_stem_conflicts']}；训练标签 stem 冲突：{counts['train_label_stem_conflicts']}。",
            f"- 图像/标签尺寸不一致：{labels['image_label_size_mismatch_count']}。",
            "",
            "## 4. 图像格式与尺寸",
            "",
            f"- 训练图像尺寸分布：{report['image_inspection']['train']['size_distribution']}。",
            f"- 测试图像尺寸分布：{report['image_inspection']['test']['size_distribution']}。",
            f"- 训练图像通道分布：{report['image_inspection']['train']['channel_distribution']}。",
            f"- 测试图像通道分布：{report['image_inspection']['test']['channel_distribution']}。",
            f"- 训练图像 dtype 分布：{report['image_inspection']['train']['dtype_distribution']}。",
            f"- 测试图像 dtype 分布：{report['image_inspection']['test']['dtype_distribution']}。",
            "",
            "## 5. 标签格式和实际像素值",
            "",
            f"- 标签尺寸分布：{labels['size_distribution']}。",
            f"- 标签模式分布：{labels['mode_distribution']}；通道分布：{labels['channel_distribution']}；dtype 分布：{labels['dtype_distribution']}。",
            f"- 实际出现的全部唯一像素值：{labels['all_unique_values']}。",
            f"- 理论 0–8 范围外像素值：{labels['values_outside_0_8']}。",
            f"- 含 255 的标签数量：{labels['labels_containing_255']}。255 只按实际存在情况报告，不被自动解释为 ignore_index。",
            f"- Palette 模式或带调色板标签：{labels['palette_label_count']}。",
            f"- 非真正单通道标签：{labels['non_single_channel_count']}。",
            f"- 单一类别标签：{labels['single_class_label_count']}。",
            "",
            "## 6. 类别分布与长尾",
            "",
            f"- 有效像素定义：{classes['valid_pixel_definition']}。",
            f"- 非 ignore 类像素量最大/最小比：{classes['pixel_level_imbalance_ratio_max_to_min_non_ignore']:.3f}。" if classes['pixel_level_imbalance_ratio_max_to_min_non_ignore'] is not None else "- 非 ignore 类像素量最大/最小比：无法计算。",
            f"- 每图非 ignore 类别数量分布：{classes['class_count_per_image_distribution_excluding_ignore']}。",
            f"- 单一类别占比超过 50%/80%/95% 的图像数：{classes['dominant_single_class_image_counts']}。",
            f"- 背景占比超过 50%/80%/95% 的图像数：{classes['background_high_ratio_image_counts']}。",
            "",
            "| class_id | 名称 | 总像素 | 有效像素占比 | 出现图片数 | 图片频率 | 每图占比中位数 | 最大占比 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| {row['class_id']} | {row['class_name'] or '-'} | {row['total_pixels']} | {percentage(row['pixel_ratio_valid_pixels'])} | "
            f"{row['images_with_class']} | {percentage(row['image_frequency'])} | {percentage(row['median_image_pixel_ratio'])} | {percentage(row['max_image_pixel_ratio'])} |"
        )
    lines.extend(
        [
            "",
            f"主要类别组合（排除 ignore，前 20）：{classes['top_class_combinations_excluding_ignore']}。",
            f"极小目标候选（该类占整图小于 0.1% 且大于 0）：{classes['extremely_small_target_candidate_images_below_0_1_percent']}。",
            "",
            "## 7. 异常文件",
            "",
            f"- error：{severity.get('error', 0)}；warning：{severity.get('warning', 0)}。完整清单见 abnormal_files.csv。",
            f"- 无法读取/空文件：{report['integrity']['unreadable_or_empty_count']}。",
            f"- 尺寸异常输入图像：{report['integrity']['unexpected_image_size_count']}；异常通道输入图像：{report['integrity']['unexpected_image_channel_count']}。",
            "",
            "## 8. 疑似重复数据",
            "",
            f"- 训练集内部同文件大小组：{duplicates['train_same_size_group_count']}（仅弱信号）。",
            f"- 训练集内部相同 dHash 组：{duplicates['train_identical_dhash_group_count']}。",
            f"- 训练与测试相同 dHash 交叉组：{duplicates['train_test_identical_dhash_group_count']}。",
            f"- 完整 SHA256：{'已计算' if duplicates['sha256_computed'] else '未计算；需显式传入 --compute-hash'}。",
            "",
            "## 9. 数据域与划分风险（初步）",
            "",
            f"- 训练文件名前缀分布：{report['domain_signals']['train_filename_prefix_distribution']}。",
            f"- 测试文件名前缀分布：{report['domain_signals']['test_filename_prefix_distribution']}。",
            f"- 训练亮度均值分位数：{report['domain_signals']['train_mean_intensity_quantiles']}。",
            f"- 测试亮度均值分位数：{report['domain_signals']['test_mean_intensity_quantiles']}。",
            "- 数据域判断仅基于目录、命名、尺寸、全局图像统计和固定种子可视化；这些信号不足以证明存在或不存在真实数据域。",
            "- 连续数字命名、疑似重复图像和相邻航拍帧都可能让随机切分高估泛化能力。正式建验证集前应按重复簇、文件序列与视觉场景分组审查；本次未自动划分。",
            "",
            "## 10. 训练前必须解决的问题",
            "",
            "1. 在官方赛题规则页再次核对评测端是否明确忽略 class_id=0，以及 mIoU 是否计入背景类。数据包 Label.txt 仅确认 0 的名称为 Ignore。",
            "2. 核对测试提交必须使用 test1_*.png 同名、单通道 PNG，且像素 ID 不经调色或归一化；数据包内未发现提交示例或格式说明。",
            "3. 根据重复簇和航拍连续场景建立无泄漏验证策略，并固定 evaluator 的 ignore_index 和类别数。",
            "4. 针对长尾类别和极小目标设计采样与损失消融，避免仅凭总体像素准确率判断模型。",
            "",
            "## 11. 推荐的下一步 baseline 方向（本报告未训练）",
            "",
            "优先验证一个可复现的 SegFormer-B0 正式基线：原始官方标签在数据加载时动态映射为官方 0→训练 255、官方 1–8→训练 0–7，使用 num_classes=8、ignore_index=255 的纯交叉熵 evaluator，并逐类报告 IoU。不得改写官方原始标签。",
            "",
            "## 12. 可视化",
            "",
            f"固定随机种子生成 {len(report['visual_samples'])} 组四联图，保存在 visual_samples。灰度面板仅为显示而缩放，伪彩色和叠加图均为派生可视化，未写回原标签。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    started = time.time()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {dataset_root}")
    if output_dir == dataset_root or output_dir.is_relative_to(dataset_root):
        raise SystemExit("Safety stop: --output-dir must not be inside --dataset-root")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root (read-only): {dataset_root}")
    print(f"Output directory: {output_dir}")
    layout = discover_layout(dataset_root)
    official_paths = [Path(path) for path in layout["official_files"]]
    official = parse_official_files(official_paths, dataset_root)
    train_image_dir = Path(layout["selected_train_images"])
    train_label_dir = Path(layout["selected_train_labels"])
    test_image_dir = Path(layout["selected_test_images"]) if layout["selected_test_images"] else None
    train_paths = image_files(train_image_dir)
    label_paths = image_files(train_label_dir)
    test_paths = image_files(test_image_dir) if test_image_dir else []

    image_index = stem_index(train_paths)
    label_index = stem_index(label_paths)
    image_conflicts = {stem: paths for stem, paths in image_index.items() if len(paths) > 1}
    label_conflicts = {stem: paths for stem, paths in label_index.items() if len(paths) > 1}
    missing_labels = [paths[0] for stem, paths in image_index.items() if stem not in label_index]
    orphan_labels = [paths[0] for stem, paths in label_index.items() if stem not in image_index]

    train_results = parallel_inspect(train_paths, inspect_input_image, args.compute_hash, args.workers, "train images")
    test_results = parallel_inspect(test_paths, inspect_input_image, args.compute_hash, args.workers, "test images")
    label_results = parallel_inspect(label_paths, inspect_label, args.compute_hash, args.workers, "train labels")
    train_by_stem = {Path(item["path"]).stem.casefold(): item for item in train_results}

    abnormalities = build_abnormal_rows(
        train_results,
        test_results,
        label_results,
        missing_labels,
        orphan_labels,
        image_conflicts,
        label_conflicts,
        train_by_stem,
    )
    class_rows, class_analysis = analyze_classes(label_results, official)
    good_labels = [item for item in label_results if item.get("error") is None]
    all_unique_values = sorted({value for item in good_labels for value in item.get("unique_values", [])})
    values_outside = sorted(set(all_unique_values) - THEORETICAL_CLASS_IDS)
    mismatch_count = sum(1 for row in abnormalities if row["issue"] == "image_label_size_mismatch")

    train_size_groups = duplicate_groups(train_results, "size_bytes")
    train_dhash_groups = duplicate_groups(train_results, "dhash")
    train_test_dhash = cross_duplicate_groups(train_results, test_results, "dhash")
    sha_train_groups = duplicate_groups(train_results, "sha256") if args.compute_hash else []
    sha_train_test = cross_duplicate_groups(train_results, test_results, "sha256") if args.compute_hash else []

    pairs = [
        (paths[0], label_index[stem][0])
        for stem, paths in image_index.items()
        if len(paths) == 1 and stem in label_index and len(label_index[stem]) == 1
    ]
    visual_samples = make_visual_samples(pairs, output_dir / "visual_samples", args.sample_count, args.seed, official)

    all_dataset_files = list(iter_files(dataset_root))
    extension_counts = Counter(path.suffix.lower() if path.suffix else "[no_extension]" for path in all_dataset_files)
    split_extension_counts = {
        "train_images": dict(sorted(Counter(p.suffix.lower() for p in train_paths).items())),
        "train_labels": dict(sorted(Counter(p.suffix.lower() for p in label_paths).items())),
        "test_images": dict(sorted(Counter(p.suffix.lower() for p in test_paths).items())),
    }
    good_train = [x for x in train_results if x.get("error") is None]
    good_test = [x for x in test_results if x.get("error") is None]
    domain_signals = {
        "train_filename_prefix_distribution": dict(sorted(Counter(numeric_prefix(Path(x["path"])) for x in good_train).items())),
        "test_filename_prefix_distribution": dict(sorted(Counter(numeric_prefix(Path(x["path"])) for x in good_test).items())),
        "train_mean_intensity_quantiles": quantiles([x["mean"] for x in good_train]),
        "test_mean_intensity_quantiles": quantiles([x["mean"] for x in good_test]),
        "train_std_intensity_quantiles": quantiles([x["std"] for x in good_train]),
        "test_std_intensity_quantiles": quantiles([x["std"] for x in good_test]),
        "interpretation_limit": "Preliminary signals only; no definitive domain conclusion is made.",
    }

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "parameters": {
            "sample_count_requested": args.sample_count,
            "sample_count_effective_minimum_20": max(20, args.sample_count),
            "seed": args.seed,
            "compute_hash": args.compute_hash,
            "workers": max(1, args.workers),
        },
        "safety": {
            "dataset_opened_read_only": True,
            "writes_below_dataset_root": False,
            "labels_converted_interpolated_or_normalized": False,
            "training_started": False,
            "data_split_created": False,
        },
        "layout_detection": layout,
        "directory_tree": tree_summary(dataset_root),
        "official_documentation": official,
        "counts": {
            "train_images": len(train_paths),
            "train_labels": len(label_paths),
            "test_images": len(test_paths),
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "train_image_stem_conflicts": len(image_conflicts),
            "train_label_stem_conflicts": len(label_conflicts),
            "all_files_below_dataset_root": len(all_dataset_files),
        },
        "extension_counts": dict(sorted(extension_counts.items())),
        "split_extension_counts": split_extension_counts,
        "pairing": {
            "missing_label_images": [str(p) for p in missing_labels],
            "orphan_labels": [str(p) for p in orphan_labels],
            "train_image_stem_conflicts": {k: [str(p) for p in v] for k, v in image_conflicts.items()},
            "train_label_stem_conflicts": {k: [str(p) for p in v] for k, v in label_conflicts.items()},
        },
        "image_inspection": {
            "train": {
                "size_distribution": distribution(good_train, "width", lambda width: "unused"),
                "mode_distribution": distribution(good_train, "mode"),
                "channel_distribution": distribution(good_train, "channels"),
                "dtype_distribution": distribution(good_train, "dtype"),
                "format_distribution": distribution(good_train, "format"),
                "min_value": min((x["min"] for x in good_train), default=None),
                "max_value": max((x["max"] for x in good_train), default=None),
            },
            "test": {
                "size_distribution": distribution(good_test, "width", lambda width: "unused"),
                "mode_distribution": distribution(good_test, "mode"),
                "channel_distribution": distribution(good_test, "channels"),
                "dtype_distribution": distribution(good_test, "dtype"),
                "format_distribution": distribution(good_test, "format"),
                "min_value": min((x["min"] for x in good_test), default=None),
                "max_value": max((x["max"] for x in good_test), default=None),
            },
        },
        "label_inspection": {
            "size_distribution": {},
            "mode_distribution": distribution(good_labels, "mode"),
            "channel_distribution": distribution(good_labels, "channels"),
            "dtype_distribution": distribution(good_labels, "dtype"),
            "format_distribution": distribution(good_labels, "format"),
            "all_unique_values": all_unique_values,
            "values_outside_0_8": values_outside,
            "labels_containing_255": sum(255 in x.get("unique_values", []) for x in good_labels),
            "palette_label_count": sum(bool(x.get("has_palette")) for x in good_labels),
            "non_single_channel_count": sum(x.get("channels") != 1 for x in good_labels),
            "single_class_label_count": sum(bool(x.get("single_value")) for x in good_labels),
            "image_label_size_mismatch_count": mismatch_count,
        },
        "class_statistics": class_rows,
        "class_distribution": class_analysis,
        "duplicate_inspection": {
            "method_note": "Same file size is a weak signal. Exact dHash equality is a perceptual-hash candidate and is not proof of byte identity.",
            "train_same_size_group_count": len(train_size_groups),
            "train_same_size_groups": train_size_groups,
            "train_identical_dhash_group_count": len(train_dhash_groups),
            "train_identical_dhash_groups": train_dhash_groups,
            "train_test_identical_dhash_group_count": len(train_test_dhash),
            "train_test_identical_dhash_groups": train_test_dhash,
            "sha256_computed": args.compute_hash,
            "train_identical_sha256_groups": sha_train_groups,
            "train_test_identical_sha256_groups": sha_train_test,
        },
        "domain_signals": domain_signals,
        "integrity": {
            "unreadable_or_empty_count": sum(row["issue"] == "unreadable_or_empty" for row in abnormalities),
            "unexpected_image_size_count": sum(row["issue"] == "unexpected_size" for row in abnormalities),
            "unexpected_image_channel_count": sum(row["issue"] == "unexpected_channels" for row in abnormalities),
            "abnormal_row_count": len(abnormalities),
        },
        "visual_samples": visual_samples,
        "elapsed_seconds": time.time() - started,
    }

    def size_dist(items: list[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(f"{x['width']}x{x['height']}" for x in items if x.get("error") is None).items()))

    report["image_inspection"]["train"]["size_distribution"] = size_dist(train_results)
    report["image_inspection"]["test"]["size_distribution"] = size_dist(test_results)
    report["label_inspection"]["size_distribution"] = size_dist(label_results)

    class_fields = [
        "class_id",
        "class_name",
        "is_ignore",
        "total_pixels",
        "pixel_ratio_all_pixels",
        "pixel_ratio_valid_pixels",
        "images_with_class",
        "image_frequency",
        "mean_image_pixel_ratio",
        "median_image_pixel_ratio",
        "max_image_pixel_ratio",
        "present_images_below_0_1_percent",
        "present_images_below_1_percent",
        "present_images_below_5_percent",
        "images_without_class",
    ]
    write_csv(output_dir / "class_statistics.csv", class_rows, class_fields)
    write_csv(output_dir / "abnormal_files.csv", abnormalities, ["severity", "split", "file", "issue", "detail"])
    (output_dir / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset_report.md").write_text(render_markdown(report, class_rows, abnormalities), encoding="utf-8")
    print(f"Inspection complete in {report['elapsed_seconds']:.1f}s")
    print(f"Unique label values: {all_unique_values}")
    print(f"Reports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
