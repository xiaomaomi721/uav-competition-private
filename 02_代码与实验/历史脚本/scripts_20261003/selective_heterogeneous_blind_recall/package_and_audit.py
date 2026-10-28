#!/usr/bin/env python3
"""Package one SHBR TREAT checkpoint and run full ZIP/behavior audits."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import shbr_common as common
from build_prediction_cache import build_b3


def audit_zip(path: Path) -> dict:
    expected = common.exact_test2_names()
    rows = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != expected or len(names) != len(set(names)):
            raise RuntimeError("submission ZIP identity/order/uniqueness failed")
        if archive.testzip() is not None:
            raise RuntimeError("submission ZIP CRC failed")
        for name in names:
            if "/" in name or "\\" in name:
                raise RuntimeError(f"nested ZIP member: {name}")
            payload = archive.read(name)
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                array = np.asarray(image)
                if image.mode != "L" or image.size != (1024, 1024) or array.dtype != np.uint8:
                    raise RuntimeError(f"PNG contract failed: {name}")
                if int(array.min()) < 1 or int(array.max()) > 8:
                    raise RuntimeError(f"label range failed: {name}")
            rows.append({"name": name, "sha256": hashlib.sha256(payload).hexdigest()})
    return {
        "count": len(rows),
        "numeric_order": True,
        "flat_root": True,
        "unique": True,
        "crc": True,
        "png_contract": True,
        "rows": rows,
    }


def read_mask(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with Image.open(io.BytesIO(archive.read(name))) as image:
        image.load()
        value = np.asarray(image)
        if image.mode != "L" or image.size != (1024, 1024) or value.dtype != np.uint8:
            raise RuntimeError(f"reference PNG contract failed: {name}")
        if int(value.min()) < 1 or int(value.max()) > 8:
            raise RuntimeError(f"reference label range failed: {name}")
        return value.astype(np.int16) - 1


def behavior(reference_path: Path, candidate_path: Path) -> dict:
    names = common.exact_test2_names()
    transition = np.zeros((8, 8), dtype=np.int64)
    reference_counts = np.zeros(8, dtype=np.int64)
    candidate_counts = np.zeros(8, dtype=np.int64)
    changed = 0
    total = 0
    identical_images = 0
    with zipfile.ZipFile(reference_path) as reference, zipfile.ZipFile(candidate_path) as candidate:
        if reference.namelist() != names or candidate.namelist() != names:
            raise RuntimeError("behavior ZIP members do not match canonical numeric order")
        for name in names:
            left = read_mask(reference, name)
            right = read_mask(candidate, name)
            moved = left != right
            count = int(moved.sum())
            identical_images += int(count == 0)
            changed += count
            total += left.size
            reference_counts += np.bincount(left.ravel(), minlength=8)[:8]
            candidate_counts += np.bincount(right.ravel(), minlength=8)[:8]
            if count:
                transition += np.bincount(
                    left[moved].astype(np.int64) * 8 + right[moved].astype(np.int64), minlength=64
                ).reshape(8, 8)
    reference_share = reference_counts / total
    candidate_share = candidate_counts / total
    return {
        "warning": "NO TEST2 GT: descriptive behavior, not accuracy evidence.",
        "reference": str(reference_path),
        "reference_sha256": common.sha256_file(reference_path),
        "candidate": str(candidate_path),
        "candidate_sha256": common.sha256_file(candidate_path),
        "total_pixels": total,
        "changed_pixels": changed,
        "changed_fraction": changed / total,
        "identical_images": identical_images,
        "class_share_delta": (candidate_share - reference_share).tolist(),
        "transition_matrix": transition.tolist(),
        "test2_ground_truth_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-health", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions = args.output_dir / "predictions"
    predictions.mkdir()
    settings = common.load_json(args.config)
    paths = common.ServerPaths.from_config(settings)
    preflight = common.load_json(args.preflight)
    gpu = preflight["selected_gpu"]
    gpu_receipt = common.verify_visible_gpu(int(gpu["index"]), gpu["uuid"])
    health = common.load_json(args.source_health)
    if health.get("status") != "PASS_SOURCE_HEALTH":
        raise RuntimeError(f"source health did not pass: {health.get('status')}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required = {
        "step": 2000,
        "recipe": "selective_heterogeneous_blind_recall",
        "formula_version": common.FORMULA_VERSION,
        "role": "treat",
        "source_coeff": 0.1,
        "target_coeff": 1.0,
        "beta": common.BETA,
        "donor_coefficient": common.BETA,
        "donor_denominator": "A05 selected count |M|",
    }
    mismatch = {key: (checkpoint.get(key), value) for key, value in required.items() if checkpoint.get(key) != value}
    if mismatch:
        raise RuntimeError(f"checkpoint metadata mismatch: {mismatch}")

    for path, expected, label in (
        (paths.control_zip, common.EXPECTED_CONTROL_ZIP_SHA, "CONTROL ZIP"),
        (paths.armb_zip, common.EXPECTED_ARMB_ZIP_SHA, "Arm B ZIP"),
    ):
        if common.sha256_file(path) != expected:
            raise RuntimeError(f"{label} SHA mismatch")
    model = build_b3(paths, args.checkpoint, args.device)
    for entry in (str(paths.code), str(paths.src)):
        if entry not in __import__("sys").path:
            __import__("sys").path.insert(0, entry)
    from src.transforms import normalize_to_tensor

    bias = torch.from_numpy(common.BSTAR).to(args.device)[:, None, None]
    names = common.exact_test2_names()
    with torch.inference_mode():
        for index, name in enumerate(names, 1):
            with Image.open(paths.test2_images / name) as image:
                pixels = normalize_to_tensor(np.asarray(image.convert("RGB")).copy()).unsqueeze(0).to(args.device)
            accumulator = None
            for view in range(8):
                transformed = common.d4_forward(pixels, view)
                with torch.amp.autocast("cuda", enabled=False):
                    logits = model(pixel_values=transformed).logits.float()
                canonical = common.d4_inverse(logits, view)[0]
                accumulator = canonical if accumulator is None else accumulator + canonical
            native = accumulator / 8.0
            upsampled = F.interpolate(
                (native + bias)[None], size=(1024, 1024), mode="bilinear", align_corners=False
            )[0]
            prediction = upsampled.argmax(0).byte().cpu().numpy() + 1
            Image.fromarray(prediction, mode="L").save(predictions / name)
            if index % 100 == 0:
                print(json.dumps({"packaged": index, "total": len(names)}), flush=True)

    zip_path = args.output_dir / "SHBR_TREAT_step2000_test2.zip"
    with zipfile.ZipFile(zip_path, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(predictions / name, name)
    zip_audit = audit_zip(zip_path)
    control_behavior = behavior(paths.control_zip, zip_path)
    armb_behavior = behavior(paths.armb_zip, zip_path)
    failures = []
    if control_behavior["changed_pixels"] == 0:
        failures.append("STOP_NO_EFFECT")
    if control_behavior["changed_fraction"] > 0.02:
        failures.append("changed_fraction>2percent")
    if max(abs(value) for value in control_behavior["class_share_delta"]) > 0.01:
        failures.append("class_share_delta>1pp")
    status = "READY_FOR_FINALIZATION" if not failures else "STOP_BEHAVIOR_OUTLIER"
    if failures == ["STOP_NO_EFFECT"]:
        status = "STOP_NO_EFFECT"
    manifest = {
        "schema_version": 1,
        "status": status,
        "failures": failures,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": common.sha256_file(args.checkpoint),
        "checkpoint_metadata": {key: checkpoint[key] for key in required},
        "zip": str(zip_path),
        "zip_sha256": common.sha256_file(zip_path),
        "zip_audit": zip_audit,
        "control_behavior": control_behavior,
        "armb_behavior": armb_behavior,
        "protocol": "single B3 checkpoint; D4 TTA8 FP32 native-logit mean; Bstar once; bilinear 1024; argmax+1",
        "single_model": True,
        "checkpoint_count": 1,
        "inference_ensemble": False,
        "teacher_used_at_candidate_inference": False,
        "gpu": gpu_receipt,
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    common.write_json(args.output_dir / "PACKAGE_MANIFEST.json", manifest)
    print(json.dumps({"status": status, "zip": str(zip_path), "zip_sha256": manifest["zip_sha256"], "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
