#!/usr/bin/env python3
"""Run deterministic single-model inference for an official submission.

This script never reads test labels, never modifies official images, and never
uses TTA or model ensembling. It refuses to infer unless the checkpoint loads
strictly into the experiment's real model configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Submission generation must not download models or other remote artifacts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import transformers
import yaml
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model, parameter_count  # noqa: E402
from src.transforms import normalize_to_tensor  # noqa: E402
from src.utils import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, PALETTE  # noqa: E402


EXPECTED_TEST_COUNT = 500
EXPECTED_SIZE = (1024, 1024)
OFFICIAL_IDS = set(range(9))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def natural_test_number(path: Path) -> int:
    stem = path.stem
    if stem.startswith("test1_") and stem[6:].isdigit():
        return int(stem[6:])
    digits = "".join(character for character in stem if character.isdigit())
    return int(digits) if digits else 10**12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a mapping, got {type(checkpoint)!r}")
    for key in ("model", "model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate and all(torch.is_tensor(value) for value in candidate.values()):
            state = dict(candidate)
            if state and all(name.startswith("module.") for name in state):
                state = {name[7:]: value for name, value in state.items()}
            return state, key
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return dict(checkpoint), "checkpoint_root"
    raise KeyError("no tensor state dict found under model/model_state_dict/state_dict or checkpoint root")


def discover_test_directory(dataset_root: Path) -> tuple[Path, list[Path], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for directory in sorted(
        (path for path in dataset_root.rglob("*") if path.is_dir() and path.name.casefold() in {"image", "images", "test", "test_images"}),
        key=lambda path: str(path).casefold(),
    ):
        pngs = [path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() == ".png"]
        candidates.append({"path": str(directory.resolve()), "png_count": len(pngs)})
    print("Test directory candidates:", flush=True)
    for candidate in candidates:
        print(f"  {candidate['path']} -> {candidate['png_count']} PNG", flush=True)

    valid = [Path(candidate["path"]) for candidate in candidates if candidate["png_count"] == EXPECTED_TEST_COUNT]
    expected = (dataset_root / "images").resolve()
    if valid != [expected]:
        raise RuntimeError(f"expected exactly one 500-image test candidate at {expected}; found {valid}")
    paths = sorted(expected.glob("*.png"), key=natural_test_number)
    expected_names = {f"test1_{index}.png" for index in range(1, EXPECTED_TEST_COUNT + 1)}
    actual_names = {path.name for path in paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise RuntimeError(f"test1 naming mismatch; missing={missing[:20]}, extra={extra[:20]}")
    return expected, paths, candidates


def inspect_test_images(paths: list[Path]) -> dict[str, Any]:
    sizes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                sizes[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                if image.size != EXPECTED_SIZE:
                    errors.append({"file": path.name, "error": f"unexpected size {image.size}"})
                if image.mode != "RGB":
                    errors.append({"file": path.name, "error": f"unexpected mode {image.mode}"})
        except Exception as exc:
            errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    result = {
        "count": len(paths),
        "first_10_filenames": [path.name for path in paths[:10]],
        "size_distribution": dict(sizes),
        "mode_distribution": dict(modes),
        "all_1024x1024": not errors and sizes == {"1024x1024": EXPECTED_TEST_COUNT},
        "errors": errors,
    }
    print(f"Test image directory: {paths[0].parent if paths else None}", flush=True)
    print(f"Test image count: {len(paths)}", flush=True)
    print(f"First 10 filenames: {result['first_10_filenames']}", flush=True)
    print(f"Size distribution: {result['size_distribution']}", flush=True)
    print(f"All 1024x1024: {result['all_1024x1024']}", flush=True)
    if len(paths) != EXPECTED_TEST_COUNT or errors:
        raise RuntimeError(f"test image inspection failed: count={len(paths)}, errors={errors[:20]}")
    return result


def strict_load_model(
    experiment_config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict, state_key = extract_state_dict(checkpoint)
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    for key in ("model_name", "model_variant", "num_classes"):
        experiment_value = experiment_config.get(key)
        checkpoint_value = checkpoint_config.get(key)
        if experiment_value != checkpoint_value:
            raise RuntimeError(
                f"configuration mismatch for {key}: experiment={experiment_value!r}, checkpoint={checkpoint_value!r}"
            )
    if int(experiment_config.get("num_classes", -1)) != 8:
        raise RuntimeError(f"num_classes must be 8, got {experiment_config.get('num_classes')}")
    model = build_model(experiment_config)
    incompatible = model.load_state_dict(state_dict, strict=True)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint load failed: missing={missing}, unexpected={unexpected}")
    if int(model.config.num_labels) != 8:
        raise RuntimeError(f"loaded model has num_labels={model.config.num_labels}, expected 8")
    classifier_shape = tuple(int(value) for value in model.decode_head.classifier.weight.shape)
    if classifier_shape[0] != 8:
        raise RuntimeError(f"classifier has invalid shape {classifier_shape}")
    model.to(device)
    model.eval()
    load_info = {
        "checkpoint_top_level_keys": list(checkpoint.keys()),
        "state_dict_key": state_key,
        "state_dict_tensor_count": len(state_dict),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "strict_load": True,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_best_miou": checkpoint.get("best_miou"),
        "parameter_count": parameter_count(model),
        "classifier_shape": list(classifier_shape),
    }
    print(json.dumps(load_info, ensure_ascii=False, indent=2), flush=True)
    return model, load_info


def write_statistics_csv(
    path: Path, records: list[dict[str, Any]], global_counts: np.ndarray, image_count: int
) -> None:
    total_pixels = int(global_counts.sum())
    class_image_counts = {
        class_id: sum(class_id in record["unique_class_ids_list"] for record in records)
        for class_id in range(9)
    }
    fieldnames = [
        "record_type",
        "filename",
        "width",
        "height",
        "mode",
        "dtype",
        "min_id",
        "max_id",
        "unique_class_ids",
        "single_class",
        "file_size_bytes",
        "inference_seconds",
        "official_class_id",
        "class_name",
        "total_pixels",
        "pixel_ratio",
        "images_with_class",
    ] + [f"class_{class_id}_pixels" for class_id in range(9)] + [f"class_{class_id}_ratio" for class_id in range(9)]
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "unique_class_ids_list"}
        rows.append(row)
    official_names = ["Ignore"] + CLASS_NAMES
    for class_id in range(9):
        rows.append(
            {
                "record_type": "global_class",
                "official_class_id": class_id,
                "class_name": official_names[class_id],
                "total_pixels": int(global_counts[class_id]),
                "pixel_ratio": float(global_counts[class_id] / total_pixels) if total_pixels else 0.0,
                "images_with_class": class_image_counts[class_id],
                "filename": f"__CLASS_{class_id}__",
                "unique_class_ids": "",
                "single_class": "",
                "width": "",
                "height": "",
                "mode": "",
                "dtype": "",
                "min_id": "",
                "max_id": "",
                "file_size_bytes": "",
                "inference_seconds": "",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_visual_checks(
    test_paths: list[Path], predictions_dir: Path, visual_dir: Path, seed: int, count: int
) -> list[str]:
    visual_dir.mkdir(parents=True, exist_ok=True)
    selected = random.Random(seed).sample(test_paths, min(count, len(test_paths)))
    written: list[str] = []
    pane = 480
    official_palette = np.zeros((9, 3), dtype=np.uint8)
    official_palette[1:] = PALETTE
    for index, image_path in enumerate(selected, 1):
        prediction_path = predictions_dir / image_path.name
        with Image.open(image_path) as image_handle:
            image = ImageOps.fit(image_handle.convert("RGB"), (pane, pane), Image.Resampling.LANCZOS)
        with Image.open(prediction_path) as prediction_handle:
            prediction = np.asarray(prediction_handle).copy()
        color = official_palette[prediction]
        color_image = Image.fromarray(color, mode="RGB").resize((pane, pane), Image.Resampling.NEAREST)
        overlay = Image.blend(image, color_image, alpha=0.45)
        ids, counts = np.unique(prediction, return_counts=True)
        total = int(counts.sum())
        canvas = Image.new("RGB", (pane * 3, pane + 115), "white")
        canvas.paste(image, (0, 0))
        canvas.paste(color_image, (pane, 0))
        canvas.paste(overlay, (pane * 2, 0))
        draw = ImageDraw.Draw(canvas)
        for x, label in ((0, "Original test image"), (pane, "Prediction pseudo-color"), (pane * 2, "Overlay")):
            draw.rectangle((x, 0, x + 190, 22), fill="white")
            draw.text((x + 5, 5), label, fill="black")
        draw.text((8, pane + 8), f"filename={image_path.name} | official_ids={list(map(int, ids.tolist()))}", fill="black")
        descriptions = [
            f"{int(class_id)} {(['Ignore'] + CLASS_NAMES)[int(class_id)]}: {int(pixel_count) / total:.4%}"
            for class_id, pixel_count in zip(ids.tolist(), counts.tolist())
        ]
        for row_index in range(0, len(descriptions), 4):
            draw.text((8, pane + 34 + 20 * (row_index // 4)), " | ".join(descriptions[row_index : row_index + 4]), fill="black")
        output_path = visual_dir / f"visual_{index:02d}_{image_path.stem}.png"
        canvas.save(output_path, format="PNG", optimize=True)
        written.append(str(output_path.resolve()))
    return written


def environment_info(device: torch.device) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "generated_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "transformers_version": transformers.__version__,
        "pillow_version": PIL.__version__,
        "numpy_version": np.__version__,
        "operating_system": platform.platform(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-count", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_utc = utc_now()
    started_wall = time.perf_counter()
    experiment_dir = args.experiment_dir.resolve()
    checkpoint_path = (args.checkpoint or (experiment_dir / "best_miou.pt")).resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    config_path = experiment_dir / "config.yaml"
    if not experiment_dir.is_dir() or not config_path.is_file() or not checkpoint_path.is_file():
        raise SystemExit("experiment directory, config.yaml, or checkpoint is missing")
    if not dataset_root.is_dir():
        raise SystemExit(f"dataset root does not exist: {dataset_root}")
    if output_dir == dataset_root or output_dir.is_relative_to(dataset_root):
        raise SystemExit("refusing to write submission outputs inside the official dataset")

    experiment_config = load_yaml(config_path)
    test_dir, test_paths, test_candidates = discover_test_directory(dataset_root)
    test_inspection = inspect_test_images(test_paths)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    model, load_info = strict_load_model(experiment_config, checkpoint_path, device)

    predictions_dir = output_dir / "predictions"
    visual_dir = output_dir / "visual_check"
    if predictions_dir.exists() and any(predictions_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"predictions directory is not empty; pass --overwrite only for an intentional rerun: {predictions_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    amp_enabled = bool(experiment_config.get("amp", True)) and device.type == "cuda"
    failures: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    global_counts = np.zeros(9, dtype=np.int64)
    total_inference_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for index, image_path in enumerate(test_paths, 1):
            try:
                with Image.open(image_path) as image_handle:
                    original_size = image_handle.size
                    image_array = np.asarray(image_handle.convert("RGB")).copy()
                pixel_values = normalize_to_tensor(image_array).unsqueeze(0).to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_started = time.perf_counter()
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(pixel_values=pixel_values).logits
                    if tuple(logits.shape[-2:]) != (original_size[1], original_size[0]):
                        logits = F.interpolate(
                            logits,
                            size=(original_size[1], original_size[0]),
                            mode="bilinear",
                            align_corners=False,
                        )
                prediction = logits.argmax(dim=1).add(1).to(torch.uint8).cpu().numpy()[0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds = time.perf_counter() - inference_started
                total_inference_seconds += inference_seconds
                if prediction.shape != (original_size[1], original_size[0]):
                    raise RuntimeError(f"prediction shape {prediction.shape} does not match input {original_size[::-1]}")
                unique_ids, counts = np.unique(prediction, return_counts=True)
                illegal = set(int(value) for value in unique_ids.tolist()) - OFFICIAL_IDS
                if illegal:
                    raise RuntimeError(f"illegal official IDs: {sorted(illegal)}")
                output_path = predictions_dir / image_path.name
                Image.fromarray(prediction.astype(np.uint8), mode="L").save(output_path, format="PNG")
                bincount = np.bincount(prediction.reshape(-1), minlength=9).astype(np.int64)
                global_counts += bincount
                total_pixels = int(prediction.size)
                record: dict[str, Any] = {
                    "record_type": "image",
                    "filename": image_path.name,
                    "width": int(prediction.shape[1]),
                    "height": int(prediction.shape[0]),
                    "mode": "L",
                    "dtype": str(prediction.dtype),
                    "min_id": int(prediction.min()),
                    "max_id": int(prediction.max()),
                    "unique_class_ids": "|".join(str(int(value)) for value in unique_ids.tolist()),
                    "unique_class_ids_list": [int(value) for value in unique_ids.tolist()],
                    "single_class": len(unique_ids) == 1,
                    "file_size_bytes": output_path.stat().st_size,
                    "inference_seconds": inference_seconds,
                }
                for class_id in range(9):
                    record[f"class_{class_id}_pixels"] = int(bincount[class_id])
                    record[f"class_{class_id}_ratio"] = float(bincount[class_id] / total_pixels)
                records.append(record)
                gpu_memory = (
                    f"allocated={torch.cuda.memory_allocated(device) / 1024**2:.1f}MB, "
                    f"reserved={torch.cuda.memory_reserved(device) / 1024**2:.1f}MB"
                    if device.type == "cuda"
                    else "CPU"
                )
                print(
                    f"[{index}/{len(test_paths)}] {image_path.name} input={original_size[0]}x{original_size[1]} "
                    f"output={prediction.shape[1]}x{prediction.shape[0]} {gpu_memory} "
                    f"avg_inference={total_inference_seconds / len(records):.4f}s remaining={len(test_paths) - index} error=None",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "filename": image_path.name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(f"[{index}/{len(test_paths)}] {image_path.name} ERROR: {failure['error_message']}", flush=True)

    if failures or len(records) != EXPECTED_TEST_COUNT:
        (output_dir / "prediction_failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Inference failed: successes={len(records)}, failures={len(failures)}. No qualified ZIP may be created.", flush=True)
        return 1

    write_statistics_csv(output_dir / "prediction_statistics.csv", records, global_counts, len(records))
    visual_paths = make_visual_checks(test_paths, predictions_dir, visual_dir, args.seed, args.visual_count)
    total_pixels = int(global_counts.sum())
    class_ratios = {str(class_id): float(global_counts[class_id] / total_pixels) for class_id in range(9)}
    absent_classes = [class_id for class_id in range(9) if global_counts[class_id] == 0]
    dominant_class_id = int(np.argmax(global_counts))
    dominant_ratio = float(global_counts[dominant_class_id] / total_pixels)
    finished_utc = utc_now()
    elapsed_seconds = time.perf_counter() - started_wall
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    run_config = {
        "experiment_id": experiment_config["experiment_id"],
        "quick_pilot": bool(experiment_config.get("quick_pilot", False)),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "experiment_config_path": str(config_path.resolve()),
        "test_images_path": str(test_dir),
        "test_image_candidates": test_candidates,
        "model_name": experiment_config["model_name"],
        "model_variant": experiment_config.get("model_variant"),
        "num_classes": 8,
        "input_mode": "RGB",
        "inference_size": [1024, 1024],
        "normalization": {
            "name": "ImageNet",
            "mean": [float(value) for value in IMAGENET_MEAN],
            "std": [float(value) for value in IMAGENET_STD],
        },
        "amp": amp_enabled,
        "TTA": False,
        "ensemble": False,
        "label_mapping": "prediction_official = prediction_train_id + 1; train 0-7 -> official 1-8",
        "output_directory": str(output_dir),
        "random_seed": args.seed,
        "execution_time": {
            "started_at_utc": started_utc,
            "finished_at_utc": finished_utc,
            "wall_seconds": elapsed_seconds,
            "total_model_inference_seconds": total_inference_seconds,
            "average_model_inference_seconds": total_inference_seconds / len(records),
        },
        "device": str(device),
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_load": load_info,
        "test_inspection": test_inspection,
        "prediction_summary": {
            "successful_predictions": len(records),
            "failed_predictions": len(failures),
            "official_class_pixel_counts": {str(class_id): int(global_counts[class_id]) for class_id in range(9)},
            "official_class_pixel_ratios": class_ratios,
            "absent_official_class_ids": absent_classes,
            "dominant_official_class_id": dominant_class_id,
            "dominant_pixel_ratio": dominant_ratio,
            "single_class_image_count": sum(bool(record["single_class"]) for record in records),
            "visual_check_count": len(visual_paths),
        },
    }
    (output_dir / "run_config.yaml").write_text(
        yaml.safe_dump(run_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment_info(device), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Inference completed: {len(records)} predictions, 0 failures", flush=True)
    print(f"Official class pixel ratios: {class_ratios}", flush=True)
    print(f"Absent official class IDs: {absent_classes}", flush=True)
    print(f"Dominant class: {dominant_class_id} ({dominant_ratio:.6%})", flush=True)
    print(f"Visual checks: {len(visual_paths)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
