#!/usr/bin/env python3
"""Freeze every SHBR input identity before any expensive inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import shbr_common as common


def stable_sha(path: Path) -> dict:
    before = path.stat()
    digest = common.sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {"path": str(path), "size": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}


def require_sha(path: Path, expected: str, label: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    receipt = stable_sha(path)
    if receipt["sha256"] != expected:
        raise RuntimeError(f"{label} SHA mismatch: expected={expected} actual={receipt['sha256']}")
    return receipt


def resolve_donor_config(configured: Path, checkpoint: Path) -> Path:
    if configured.is_file():
        return configured
    candidates = sorted(
        path
        for path in checkpoint.parent.rglob("*.py")
        if "vis_data" not in path.parts and "__pycache__" not in path.parts
    )
    if len(candidates) != 1:
        raise RuntimeError(f"donor config missing/ambiguous: configured={configured} candidates={candidates}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")
    settings = common.load_json(args.config)
    paths = common.ServerPaths.from_config(settings)

    required_directories = (
        paths.code,
        paths.src,
        paths.model_source,
        paths.source_images,
        paths.source_masks,
        paths.test2_images,
        paths.a05_target_cache,
    )
    missing = [str(path) for path in required_directories if not path.is_dir()]
    if missing:
        raise RuntimeError(f"missing required directories: {missing}")

    donor_config = resolve_donor_config(paths.donor_config, paths.donor_checkpoint)
    artifacts = {
        "a05": require_sha(paths.a05, common.EXPECTED_A05_SHA, "A05"),
        "armb": require_sha(paths.armb, common.EXPECTED_ARMB_SHA, "Arm B"),
        "armb_zip": require_sha(paths.armb_zip, common.EXPECTED_ARMB_ZIP_SHA, "Arm B ZIP"),
        "control_checkpoint": require_sha(
            paths.control_checkpoint, common.EXPECTED_CONTROL_CHECKPOINT_SHA, "xlogy CONTROL checkpoint"
        ),
        "control_zip": require_sha(paths.control_zip, common.EXPECTED_CONTROL_ZIP_SHA, "xlogy CONTROL ZIP"),
        "control_reference": stable_sha(paths.control_reference),
        "donor_checkpoint": stable_sha(paths.donor_checkpoint),
        "donor_config": stable_sha(donor_config),
        "b3_config": stable_sha(paths.config),
        "source_audit_split": stable_sha(paths.source_audit_split),
    }

    split = common.load_json(paths.source_audit_split)
    h300 = [str(item) for item in split.get("H300", [])]
    r1099 = [str(item) for item in split.get("R1099", [])]
    if len(h300) != 300 or len(r1099) != 1099 or set(h300) & set(r1099) or len(set(h300 + r1099)) != 1399:
        raise RuntimeError("source audit H300/R1099 identity contract failed")
    source_names = sorted(path.stem for path in paths.source_images.glob("*.png"))
    mask_names = sorted(path.stem for path in paths.source_masks.glob("*.png"))
    if len(source_names) != 6996 or source_names != mask_names:
        raise RuntimeError(f"official source image/mask identity failed: images={len(source_names)} masks={len(mask_names)}")
    if not set(h300 + r1099) <= set(source_names):
        raise RuntimeError("source audit split contains missing source IDs")
    train_ids = [item for item in paths.source_train_split.read_text(encoding="utf-8-sig").split() if item]
    if len(train_ids) != 5597 or set(train_ids) & set(h300 + r1099) or set(train_ids) | set(h300 + r1099) != set(source_names):
        raise RuntimeError("5597/1399 source partition contract failed")

    common.validate_exact_names(paths.test2_images, common.exact_test2_names(), "test2")
    target_cache_names = sorted(path.name for path in paths.a05_target_cache.glob("*.npy"))
    expected_cache_names = sorted(name.replace(".png", ".npy") for name in common.exact_test2_names())
    if target_cache_names != expected_cache_names:
        raise RuntimeError("A05 target cache identity mismatch")

    gt_encoding = settings["source_gt_encoding"]
    gt_smoke = []
    fixed_stems = (h300[:4] + r1099[:4])
    for stem in fixed_stems:
        with Image.open(paths.source_masks / f"{stem}.png") as image:
            raw = np.asarray(image).copy()
        mapped = common.canonical_gt_to_train(raw, gt_encoding)
        gt_smoke.append(
            {
                "stem": stem,
                "raw_unique": sorted(int(item) for item in np.unique(raw)),
                "mapped_unique": sorted(int(item) for item in np.unique(mapped)),
            }
        )

    allowed_gpus = settings.get("allowed_physical_gpus")
    if allowed_gpus != [2, 3]:
        raise RuntimeError(f"authorized GPU contract requires exactly physical GPU2/GPU3, got {allowed_gpus}")
    gpu = common.choose_idle_gpu(allowed_gpus)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "formula_version": common.FORMULA_VERSION,
        "config": str(args.config.resolve()),
        "config_sha256": common.sha256_file(args.config),
        "artifacts": artifacts,
        "resolved_donor_config": str(donor_config),
        "source": {
            "official_count": 6996,
            "train_count": len(train_ids),
            "audit_count": len(h300) + len(r1099),
            "H300_count": len(h300),
            "R1099_count": len(r1099),
            "train_ids_sha256": common.sha256_lines(train_ids),
            "H300_sha256": common.sha256_lines(h300),
            "R1099_sha256": common.sha256_lines(r1099),
            "gt_encoding": gt_encoding,
            "gt_smoke": gt_smoke,
        },
        "test2": {"count": 1300, "ordered_names_sha256": common.sha256_lines(common.exact_test2_names())},
        "selected_gpu": gpu,
        "no_process_was_stopped": True,
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    common.write_json(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
