#!/usr/bin/env python3
"""Static checks for the Stage 03 full-data SegFormer-B0 baseline config."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "full" / "stage03_b0_ce_512_seed42.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part) for part in inner.split(",")]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", value) or re.fullmatch(
        r"[-+]?\d+[eE][-+]?\d+", value
    ):
        return float(value)
    return value.strip("'\"")


def read_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"unsupported YAML line {line_number}: {raw_line}")
        key, value = stripped.split(":", 1)
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = parse_scalar(value)
    return root


def read_stems(path: Path, errors: list[str], label: str) -> list[str]:
    if not path.is_file():
        errors.append(f"{label} missing: {path}")
        return []
    stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicate_count = len(stems) - len(set(stems))
    if duplicate_count:
        errors.append(f"{label} contains {duplicate_count} duplicate stems")
    return stems


def nested_get(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def flatten(data: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(data, dict):
        items: list[tuple[str, Any]] = []
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            items.extend(flatten(value, child))
        return items
    if isinstance(data, list):
        items = []
        for index, value in enumerate(data):
            items.extend(flatten(value, f"{prefix}[{index}]"))
        return items
    return [(prefix, data)]


def is_windows_absolute_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def check_expected_values(config: dict[str, Any], errors: list[str]) -> None:
    expected = {
        "experiment_id": "formal_b0_ce_512_seed42",
        "quick_pilot": False,
        "model_variant": "SegFormer-B0",
        "num_classes": 8,
        "ignore_index": 255,
        "crop_size": 512,
        "loss.name": "cross_entropy",
        "optimizer.name": "AdamW",
        "optimizer.learning_rate": 0.00006,
        "optimizer.weight_decay": 0.01,
        "seed": 42,
        "num_workers": 0,
        "amp": True,
        "train_split": "splits/train.txt",
        "validation_split": "splits/val.txt",
        "max_steps": 56000,
        "validation_interval": 1400,
        "rare_crop.enabled": False,
        "transforms.domain_aug.enabled": False,
    }
    for key, expected_value in expected.items():
        actual = nested_get(config, key)
        if actual != expected_value:
            errors.append(f"{key} expected {expected_value!r}, got {actual!r}")
    for key in ("max_train_samples", "max_val_samples"):
        actual = nested_get(config, key)
        if actual not in (0, None):
            errors.append(f"{key} must be 0 or None, got {actual!r}")


def check_forbidden_content(config: dict[str, Any], errors: list[str]) -> None:
    forbidden_tokens = (
        "test_split",
        "test_root",
        "test/images",
        "test\\images",
        "submission",
        "ensemble",
        "tta",
        "external_train",
        "external_data",
    )
    for key, value in flatten(config):
        text = str(value)
        lower_key = key.lower()
        lower_text = text.lower()
        if isinstance(value, str) and is_windows_absolute_path(value):
            errors.append(f"absolute local path is not allowed at {key}: {value}")
        for token in forbidden_tokens:
            if token in lower_key or token in lower_text:
                errors.append(f"forbidden Stage 03 setting token {token!r} at {key}: {value!r}")


def check_splits(errors: list[str]) -> None:
    split_dir = PROJECT_ROOT / "splits"
    train = read_stems(split_dir / "train.txt", errors, "train.txt")
    val = read_stems(split_dir / "val.txt", errors, "val.txt")
    fixed = read_stems(split_dir / "fixed_quick_val.txt", errors, "fixed_quick_val.txt")
    train_set = set(train)
    val_set = set(val)
    fixed_set = set(fixed)
    if len(train) != 5597:
        errors.append(f"train.txt expected 5597 stems, got {len(train)}")
    if len(val) != 1399:
        errors.append(f"val.txt expected 1399 stems, got {len(val)}")
    if train_set & val_set:
        errors.append(f"train/val overlap: {len(train_set & val_set)}")
    if len(fixed) != 256:
        errors.append(f"fixed_quick_val.txt expected 256 stems, got {len(fixed)}")
    if fixed and not fixed_set.issubset(val_set):
        errors.append("fixed_quick_val.txt must be a subset of val.txt")
    if fixed_set & train_set:
        errors.append("fixed_quick_val.txt must not contain train.txt stems")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    errors: list[str] = []
    try:
        config = read_yaml(config_path)
    except Exception as exc:
        errors.append(f"failed to read config: {exc}")
        config = {}
    check_expected_values(config, errors)
    check_forbidden_content(config, errors)
    check_splits(errors)
    if errors:
        print("Stage 03 config check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Stage 03 config check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
