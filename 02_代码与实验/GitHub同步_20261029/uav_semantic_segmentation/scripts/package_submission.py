#!/usr/bin/env python3
"""Create and re-validate a flat 500-PNG submission ZIP after PASS."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EXPECTED_COUNT = 500
EXPECTED_SIZE = (1024, 1024)
ALLOWED_IDS = set(range(9))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def natural_number(path: Path) -> int:
    suffix = path.stem.split("_")[-1]
    return int(suffix) if suffix.isdigit() else 10**12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--check-json", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=10)
    return parser.parse_args()


def validate_zip(
    zip_path: Path, expected_names: set[str], seed: int, sample_count: int
) -> dict[str, Any]:
    failures: list[str] = []
    sampled: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        directory_entries = [info.filename for info in infos if info.is_dir()]
        nested_paths = [name for name in names if "/" in name or "\\" in name]
        non_png = [name for name in names if not name.casefold().endswith(".png")]
        if len(infos) != EXPECTED_COUNT:
            failures.append(f"ZIP entry count is {len(infos)}, expected {EXPECTED_COUNT}")
        if directory_entries:
            failures.append(f"ZIP contains directory entries: {directory_entries}")
        if nested_paths:
            failures.append(f"ZIP contains nested paths: {nested_paths[:20]}")
        if non_png:
            failures.append(f"ZIP contains non-PNG entries: {non_png[:20]}")
        if set(names) != expected_names:
            failures.append(
                f"ZIP/test filename mismatch; missing={sorted(expected_names - set(names))[:20]}, "
                f"extra={sorted(set(names) - expected_names)[:20]}"
            )
        bad_member = archive.testzip()
        if bad_member is not None:
            failures.append(f"ZipFile.testzip() returned corrupt member {bad_member}")
        sample_names = random.Random(seed).sample(names, min(sample_count, len(names)))
        for name in sample_names:
            try:
                data = archive.read(name)
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    mode = image.mode
                    size = image.size
                    has_palette = image.getpalette() is not None
                    array = np.asarray(image)
                ids = sorted(int(value) for value in np.unique(array).tolist())
                sample_result = {
                    "filename": name,
                    "size": list(size),
                    "mode": mode,
                    "dtype": str(array.dtype),
                    "has_palette": has_palette,
                    "min_id": min(ids),
                    "max_id": max(ids),
                    "unique_ids": ids,
                    "pass": True,
                }
                if size != EXPECTED_SIZE:
                    sample_result["pass"] = False
                    failures.append(f"sample {name} has size {size}")
                if mode != "L":
                    sample_result["pass"] = False
                    failures.append(f"sample {name} has mode {mode}")
                if has_palette:
                    sample_result["pass"] = False
                    failures.append(f"sample {name} has palette")
                if array.dtype != np.uint8:
                    sample_result["pass"] = False
                    failures.append(f"sample {name} has dtype {array.dtype}")
                illegal = set(ids) - ALLOWED_IDS
                if illegal:
                    sample_result["pass"] = False
                    failures.append(f"sample {name} has illegal IDs {sorted(illegal)}")
                sampled.append(sample_result)
            except Exception as exc:
                failures.append(f"failed to inspect ZIP sample {name}: {type(exc).__name__}: {exc}")
                sampled.append({"filename": name, "pass": False, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "status": "PASS" if not failures else "FAIL",
        "checked_at_utc": utc_now(),
        "zip_path": str(zip_path.resolve()),
        "zip_size_bytes": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
        "entry_count": len(infos),
        "all_entries_png": not non_png,
        "directory_entries": directory_entries,
        "nested_paths": nested_paths,
        "filename_set_matches_test": set(names) == expected_names,
        "testzip_result": bad_member,
        "sample_count": len(sampled),
        "sample_results": sampled,
        "failures": failures,
    }


def update_check_reports(check_path: Path, report: dict[str, Any], zip_result: dict[str, Any]) -> None:
    report["zip_verification"] = zip_result
    check_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = check_path.with_suffix(".md")
    markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.is_file() else report["status"]
    marker = "\n## ZIP 复检"
    if marker in markdown:
        markdown = markdown.split(marker, 1)[0].rstrip()
    lines = [
        markdown.rstrip(),
        "",
        "## ZIP 复检",
        "",
        f"- 状态：{zip_result['status']}",
        f"- ZIP：{zip_result['zip_path']}",
        f"- 文件大小：{zip_result['zip_size_bytes']} bytes",
        f"- SHA256：{zip_result['sha256']}",
        f"- 文件数量：{zip_result['entry_count']}",
        f"- 目录项：{zip_result['directory_entries']}",
        f"- 嵌套路径：{zip_result['nested_paths']}",
        f"- 文件名集合与测试集一致：{zip_result['filename_set_matches_test']}",
        f"- testzip 结果：{zip_result['testzip_result']}",
        f"- ZIP 内抽检 PNG：{zip_result['sample_count']} 张，全部通过={all(item.get('pass') for item in zip_result['sample_results'])}",
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    test_dir = args.test_images.resolve()
    predictions_dir = args.predictions.resolve()
    check_path = args.check_json.resolve()
    zip_path = args.zip_path.resolve()
    if not check_path.is_file():
        raise SystemExit(f"submission check JSON does not exist: {check_path}")
    check_report = json.loads(check_path.read_text(encoding="utf-8"))
    if check_report.get("status") != "PASS" or check_report.get("failures"):
        raise SystemExit("submission_check is not PASS; refusing to create ZIP")
    if not test_dir.is_dir() or not predictions_dir.is_dir():
        raise SystemExit("test or prediction directory does not exist")
    test_paths = sorted(test_dir.glob("*.png"), key=natural_number)
    prediction_paths = sorted(predictions_dir.glob("*.png"), key=natural_number)
    test_names = {path.name for path in test_paths}
    prediction_names = {path.name for path in prediction_paths}
    if len(test_paths) != EXPECTED_COUNT or len(prediction_paths) != EXPECTED_COUNT:
        raise SystemExit(
            f"expected {EXPECTED_COUNT} test/prediction PNGs, found {len(test_paths)}/{len(prediction_paths)}"
        )
    if test_names != prediction_names:
        raise SystemExit("prediction filenames do not exactly match test filenames")
    if any(path.is_dir() for path in predictions_dir.iterdir()):
        raise SystemExit("prediction directory contains a nested directory")
    non_png = [path.name for path in predictions_dir.iterdir() if path.is_file() and path.suffix.casefold() != ".png"]
    if non_png:
        raise SystemExit(f"prediction directory contains non-PNG files: {non_png}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = zip_path.with_suffix(zip_path.suffix + ".tmp")
    if temporary_zip.exists():
        temporary_zip.unlink()
    try:
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for prediction_path in prediction_paths:
                archive.write(prediction_path, arcname=prediction_path.name)
        zip_result = validate_zip(temporary_zip, test_names, args.seed, max(10, args.sample_count))
        if zip_result["status"] != "PASS":
            update_check_reports(check_path, check_report, zip_result)
            raise RuntimeError(f"ZIP validation failed: {zip_result['failures']}")
        temporary_zip.replace(zip_path)
        zip_result["zip_path"] = str(zip_path.resolve())
        zip_result["zip_size_bytes"] = zip_path.stat().st_size
        zip_result["sha256"] = sha256_file(zip_path)
        update_check_reports(check_path, check_report, zip_result)
        print(json.dumps(zip_result, ensure_ascii=False, indent=2))
        return 0
    finally:
        if temporary_zip.exists():
            temporary_zip.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
