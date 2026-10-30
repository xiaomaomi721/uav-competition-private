#!/usr/bin/env python3
"""Strictly validate prediction PNGs before submission packaging."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EXPECTED_SIZE = (1024, 1024)
ALLOWED_IDS = set(range(9))
OFFICIAL_NAMES = [
    "Ignore",
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Vegetation",
    "Agricultural",
    "Vehicle",
]
FORBIDDEN_NAMES = {"thumbs.db", "desktop.ini", "readme", "readme.md", "readme.txt"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def natural_number(path: Path) -> int:
    suffix = path.stem.split("_")[-1]
    return int(suffix) if suffix.isdigit() else 10**12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    return parser.parse_args()


def casefold_duplicates(paths: list[Path]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        groups[path.name.casefold()].append(path.name)
    return {key: values for key, values in groups.items() if len(values) > 1}


def add_failure(failures: list[dict[str, str]], check: str, file: str, reason: str) -> None:
    failures.append({"check": check, "file": file, "reason": reason})


def render_markdown(report: dict[str, Any]) -> str:
    status = report["status"]
    lines = [
        status,
        "",
        "# 初赛提交严格检查报告",
        "",
        f"- 生成时间（UTC）：{report['generated_at_utc']}",
        f"- 测试图像数量：{report['counts']['test_images']}",
        f"- 预测图像数量：{report['counts']['prediction_pngs']}",
        f"- 硬性失败数：{len(report['failures'])}",
        f"- 警告数：{len(report['warnings'])}",
        "",
        "## 硬性检查",
        "",
        f"- 文件名集合完全一致：{report['checks']['filename_sets_equal']}",
        f"- 缺失预测：{report['missing_predictions']}",
        f"- 多余预测：{report['extra_predictions']}",
        f"- 重复文件名：{report['duplicate_prediction_names']}",
        f"- 全部可读取：{report['checks']['all_readable']}",
        f"- 全部 PNG：{report['checks']['all_png']}",
        f"- 全部 L 模式：{report['checks']['all_mode_l']}",
        f"- Palette 数量：{report['prediction_properties']['palette_count']}",
        f"- 全部 1024×1024：{report['checks']['all_1024x1024']}",
        f"- 全部与测试图尺寸一致：{report['checks']['all_match_test_size']}",
        f"- 全部 uint8：{report['checks']['all_uint8']}",
        f"- 像素值全部属于 0–8：{report['checks']['all_ids_in_0_8']}",
        f"- 实际预测像素范围：{report['prediction_properties']['actual_min_id']}–{report['prediction_properties']['actual_max_id']}",
        f"- 嵌套目录或文件：{report['nested_entries']}",
        f"- 非预测文件：{report['non_prediction_entries']}",
        "",
        "## 预测类别统计",
        "",
        "| 官方 ID | 类别 | 像素数 | 像素比例 | 出现图片数 |",
        "|---:|---|---:|---:|---:|",
    ]
    for class_id in range(9):
        item = report["class_statistics"][str(class_id)]
        lines.append(
            f"| {class_id} | {OFFICIAL_NAMES[class_id]} | {item['pixel_count']} | "
            f"{item['pixel_ratio']:.6%} | {item['images_with_class']} |"
        )
    lines.extend(
        [
            "",
            f"- 完全缺失类别：{report['prediction_properties']['absent_class_ids']}",
            f"- 单一类别预测图片数：{report['prediction_properties']['single_class_image_count']}",
            f"- 全局主导类别：{report['prediction_properties']['dominant_class_id']}，占比 {report['prediction_properties']['dominant_class_ratio']:.6%}",
            "",
            "## 失败明细",
            "",
        ]
    )
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- [{failure['check']}] {failure['file']}: {failure['reason']}")
    else:
        lines.append("- 无")
    lines.extend(["", "## 警告", ""])
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "全部硬性格式检查通过，可以进入 ZIP 打包步骤。" if status == "PASS" else "存在硬性失败，不得生成正式提交 ZIP。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    test_dir = args.test_images.resolve()
    predictions_dir = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    warnings: list[str] = []
    if not test_dir.is_dir():
        add_failure(failures, "test_directory", str(test_dir), "directory does not exist")
    if not predictions_dir.is_dir():
        add_failure(failures, "prediction_directory", str(predictions_dir), "directory does not exist")

    test_paths = sorted(test_dir.glob("*.png"), key=natural_number) if test_dir.is_dir() else []
    prediction_direct_files = [path for path in predictions_dir.iterdir() if path.is_file()] if predictions_dir.is_dir() else []
    prediction_pngs = sorted(
        [path for path in prediction_direct_files if path.suffix.casefold() == ".png"], key=natural_number
    )
    all_recursive_entries = list(predictions_dir.rglob("*")) if predictions_dir.is_dir() else []
    nested_entries = [
        str(path.relative_to(predictions_dir))
        for path in all_recursive_entries
        if len(path.relative_to(predictions_dir).parts) > 1
    ]
    non_prediction_entries = [
        path.name
        for path in prediction_direct_files
        if path.suffix.casefold() != ".png" or path.name.casefold() in FORBIDDEN_NAMES or path.name.startswith(".")
    ]
    direct_directories = [path.name for path in predictions_dir.iterdir() if path.is_dir()] if predictions_dir.is_dir() else []
    if direct_directories or nested_entries:
        add_failure(
            failures,
            "nested_entries",
            str(predictions_dir),
            f"directories={direct_directories}, nested_entries={nested_entries[:20]}",
        )
    if non_prediction_entries:
        add_failure(failures, "non_prediction_entries", str(predictions_dir), str(non_prediction_entries))
    if len(test_paths) != args.expected_count:
        add_failure(failures, "test_count", str(test_dir), f"expected {args.expected_count}, found {len(test_paths)}")
    if len(prediction_pngs) != args.expected_count:
        add_failure(
            failures,
            "prediction_count",
            str(predictions_dir),
            f"expected {args.expected_count}, found {len(prediction_pngs)}",
        )

    test_names = {path.name for path in test_paths}
    prediction_names = {path.name for path in prediction_pngs}
    missing_predictions = sorted(test_names - prediction_names)
    extra_predictions = sorted(prediction_names - test_names)
    if missing_predictions:
        add_failure(failures, "missing_predictions", str(predictions_dir), str(missing_predictions[:50]))
    if extra_predictions:
        add_failure(failures, "extra_predictions", str(predictions_dir), str(extra_predictions[:50]))
    duplicates = casefold_duplicates(prediction_direct_files)
    if duplicates:
        add_failure(failures, "duplicate_names", str(predictions_dir), str(duplicates))

    test_sizes: dict[str, tuple[int, int]] = {}
    for test_path in test_paths:
        try:
            with Image.open(test_path) as image:
                test_sizes[test_path.name] = image.size
        except Exception as exc:
            add_failure(failures, "test_read", test_path.name, f"{type(exc).__name__}: {exc}")

    global_counts = np.zeros(9, dtype=np.int64)
    images_with_class = np.zeros(9, dtype=np.int64)
    actual_min: int | None = None
    actual_max: int | None = None
    palette_count = 0
    mode_counter: Counter[str] = Counter()
    size_counter: Counter[str] = Counter()
    dtype_counter: Counter[str] = Counter()
    single_class_files: list[str] = []
    readable_count = 0
    for prediction_path in prediction_pngs:
        if prediction_path.stat().st_size == 0:
            add_failure(failures, "empty_file", prediction_path.name, "file size is zero")
            continue
        try:
            with Image.open(prediction_path) as image:
                image.load()
                mode = image.mode
                size = image.size
                has_palette = image.getpalette() is not None
                array = np.asarray(image)
            readable_count += 1
            mode_counter[mode] += 1
            size_counter[f"{size[0]}x{size[1]}"] += 1
            dtype_counter[str(array.dtype)] += 1
            if prediction_path.suffix.casefold() != ".png" or image.format != "PNG":
                add_failure(failures, "format", prediction_path.name, f"suffix={prediction_path.suffix}, format={image.format}")
            if mode != "L":
                add_failure(failures, "mode", prediction_path.name, f"expected L, got {mode}")
            if has_palette:
                palette_count += 1
                add_failure(failures, "palette", prediction_path.name, "prediction has a palette")
            if size != EXPECTED_SIZE:
                add_failure(failures, "size", prediction_path.name, f"expected {EXPECTED_SIZE}, got {size}")
            if prediction_path.name in test_sizes and size != test_sizes[prediction_path.name]:
                add_failure(
                    failures,
                    "test_size_match",
                    prediction_path.name,
                    f"prediction={size}, test={test_sizes[prediction_path.name]}",
                )
            if array.dtype != np.uint8:
                add_failure(failures, "dtype", prediction_path.name, f"expected uint8, got {array.dtype}")
            if array.ndim != 2:
                add_failure(failures, "channels", prediction_path.name, f"expected 2D array, got shape {array.shape}")
                continue
            ids, counts = np.unique(array, return_counts=True)
            illegal = set(int(value) for value in ids.tolist()) - ALLOWED_IDS
            if illegal:
                add_failure(failures, "pixel_ids", prediction_path.name, f"illegal IDs {sorted(illegal)}")
                continue
            if len(ids) == 1:
                single_class_files.append(prediction_path.name)
            local_counts = np.bincount(array.reshape(-1), minlength=9).astype(np.int64)
            global_counts += local_counts
            images_with_class += (local_counts > 0).astype(np.int64)
            local_min, local_max = int(ids.min()), int(ids.max())
            actual_min = local_min if actual_min is None else min(actual_min, local_min)
            actual_max = local_max if actual_max is None else max(actual_max, local_max)
        except Exception as exc:
            add_failure(failures, "readability", prediction_path.name, f"{type(exc).__name__}: {exc}")

    total_pixels = int(global_counts.sum())
    ratios = global_counts / total_pixels if total_pixels else np.zeros(9, dtype=np.float64)
    absent_classes = [class_id for class_id in range(9) if global_counts[class_id] == 0]
    if absent_classes:
        warnings.append(f"以下官方类别在 500 张预测中完全未出现：{absent_classes}；仅作 WARNING，不自动修改。")
    if single_class_files:
        warnings.append(f"有 {len(single_class_files)} 张图片只预测出一个类别：{single_class_files[:30]}")
    dominant_class_id = int(np.argmax(global_counts)) if total_pixels else None
    dominant_ratio = float(ratios[dominant_class_id]) if dominant_class_id is not None else 0.0
    if dominant_ratio > 0.90:
        add_failure(
            failures,
            "prediction_collapse",
            "all_predictions",
            f"official class {dominant_class_id} occupies {dominant_ratio:.6%}, above 90%; refuse packaging",
        )
    if global_counts[0] > 0:
        warnings.append(f"模型输出了官方 Ignore ID 0，共 {int(global_counts[0])} 像素。")

    hard_checks = {
        "filename_sets_equal": test_names == prediction_names,
        "all_readable": readable_count == len(prediction_pngs),
        "all_png": all(path.suffix.casefold() == ".png" for path in prediction_direct_files),
        "all_mode_l": mode_counter == {"L": len(prediction_pngs)},
        "all_1024x1024": size_counter == {"1024x1024": len(prediction_pngs)},
        "all_match_test_size": all(test_sizes.get(path.name) == EXPECTED_SIZE for path in prediction_pngs),
        "all_uint8": dtype_counter == {"uint8": len(prediction_pngs)},
        "all_ids_in_0_8": actual_min is not None and actual_min >= 0 and actual_max is not None and actual_max <= 8,
        "no_palette": palette_count == 0,
        "no_nested_or_extra_entries": not nested_entries and not direct_directories and not non_prediction_entries,
        "not_globally_collapsed_over_90_percent": dominant_ratio <= 0.90,
    }
    for check, passed in hard_checks.items():
        if not passed and not any(failure["check"] == check for failure in failures):
            add_failure(failures, check, str(predictions_dir), "aggregate check failed")

    status = "PASS" if not failures else "FAIL"
    report: dict[str, Any] = {
        "status": status,
        "generated_at_utc": utc_now(),
        "test_images_directory": str(test_dir),
        "predictions_directory": str(predictions_dir),
        "counts": {
            "expected": args.expected_count,
            "test_images": len(test_paths),
            "prediction_pngs": len(prediction_pngs),
            "prediction_direct_files": len(prediction_direct_files),
        },
        "checks": hard_checks,
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "duplicate_prediction_names": duplicates,
        "nested_entries": nested_entries,
        "non_prediction_entries": non_prediction_entries,
        "prediction_properties": {
            "mode_distribution": dict(mode_counter),
            "size_distribution": dict(size_counter),
            "dtype_distribution": dict(dtype_counter),
            "palette_count": palette_count,
            "actual_min_id": actual_min,
            "actual_max_id": actual_max,
            "absent_class_ids": absent_classes,
            "single_class_image_count": len(single_class_files),
            "single_class_files": single_class_files,
            "dominant_class_id": dominant_class_id,
            "dominant_class_ratio": dominant_ratio,
        },
        "class_statistics": {
            str(class_id): {
                "class_name": OFFICIAL_NAMES[class_id],
                "pixel_count": int(global_counts[class_id]),
                "pixel_ratio": float(ratios[class_id]),
                "images_with_class": int(images_with_class[class_id]),
            }
            for class_id in range(9)
        },
        "failures": failures,
        "warnings": warnings,
    }
    (output_dir / "submission_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "submission_check.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"submission_check={status}")
    print(f"failures={len(failures)} warnings={len(warnings)}")
    print(f"actual_pixel_range={actual_min}..{actual_max}")
    print(f"class_ratios={{{', '.join(f'{i}: {float(ratios[i]):.8f}' for i in range(9))}}}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
