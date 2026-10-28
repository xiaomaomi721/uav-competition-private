#!/usr/bin/env python3
"""Package the single formal TCA checkpoint with the scored Arm B protocol."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import tca_common as common


def build_model(paths: common.ServerPaths, checkpoint: Path, device: str):
    for entry in (str(paths.root / "code"), str(paths.src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.environ["UAVSEG_MODEL_SOURCE"] = str(paths.model_source)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from src.models import build_model as make_model
    from src.utils import load_config

    cfg = load_config(str(paths.config))
    cfg["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0, "norm_eps": 1e-12, "start_step": 1}
    cfg.pop("mixstyle", None)
    cfg["pretrained_model_path"] = str(paths.model_source)
    model = make_model(cfg)
    model.load_state_dict(common.load_state(checkpoint), strict=True)
    return model.to(device).eval()


def audit_zip(path: Path) -> dict:
    expected = common.exact_test2_names()
    rows = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != expected:
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise RuntimeError(f"ZIP identity/order mismatch: missing={missing[:5]} extra={extra[:5]}")
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate ZIP member")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        for name in names:
            if "/" in name or "\\" in name:
                raise RuntimeError(f"nested ZIP member: {name}")
            payload = archive.read(name)
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                array = np.asarray(image)
                if image.mode != "L" or image.size != (1024, 1024) or array.dtype != np.uint8:
                    raise RuntimeError(
                        f"bad PNG contract {name}: mode={image.mode} size={image.size} dtype={array.dtype}"
                    )
                minimum = int(array.min())
                maximum = int(array.max())
                if minimum < 1 or maximum > 8:
                    raise RuntimeError(f"label range violation {name}: {minimum}..{maximum}")
                rows.append(
                    {
                        "name": name,
                        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                        "label_min": minimum,
                        "label_max": maximum,
                    }
                )
    return {
        "count": len(rows),
        "exact_numeric_order": True,
        "flat_root": True,
        "unique_names": True,
        "crc_pass": True,
        "all_png_contracts_pass": True,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/uav"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", default="TCA_TARGET_COMPLEMENT_ANCHOR")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    gpu_receipt = common.verify_gpu(args.physical_gpu, args.expected_gpu_uuid)
    paths = common.ServerPaths.from_root(args.root)
    common.validate_server_paths(paths)
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    checkpoint_obj = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_obj, dict):
        raise SystemExit("formal checkpoint must be a metadata dictionary")
    expected_metadata = {
        "step": 2000,
        "recipe": "target_complement_anchor",
        "formula_version": common.FORMULA_VERSION,
        "a05_sha256": common.EXPECTED_A05_SHA,
        "black_region_training_exclusion": False,
        "full_teacher_grid_partition": True,
        "source_coeff": 0.0,
        "target_coeff": 1.0,
    }
    mismatch = {
        key: (checkpoint_obj.get(key), expected)
        for key, expected in expected_metadata.items()
        if checkpoint_obj.get(key) != expected
    }
    if mismatch:
        raise SystemExit(f"checkpoint metadata mismatch: {mismatch}")
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=False)

    for entry in (str(paths.root / "code"), str(paths.src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from src.transforms import normalize_to_tensor

    model = build_model(paths, args.checkpoint, args.device)
    bias = torch.from_numpy(common.BSTAR).to(args.device)[:, None, None]
    names = common.exact_test2_names()
    started = time.time()

    with torch.inference_mode():
        for index, name in enumerate(names, 1):
            with Image.open(paths.test2 / name) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            pixels = normalize_to_tensor(rgb).unsqueeze(0).to(args.device)
            accumulator = None
            for view in range(8):
                transformed = common.d4_forward(pixels, view)
                with torch.amp.autocast("cuda", enabled=False):
                    logits = model(pixel_values=transformed).logits.float()
                canonical = common.d4_inverse(logits, view)[0]
                accumulator = canonical if accumulator is None else accumulator + canonical
            mean_raw = accumulator / 8.0
            upsampled = F.interpolate(
                (mean_raw + bias)[None],
                size=(1024, 1024),
                mode="bilinear",
                align_corners=False,
            )[0]
            prediction = upsampled.argmax(0).byte().cpu().numpy() + 1
            Image.fromarray(prediction, mode="L").save(prediction_dir / name)
            if index % 200 == 0:
                print(json.dumps({"packaged": index, "total": 1300, "elapsed_sec": round(time.time() - started, 1)}), flush=True)

    zip_path = args.output_dir / f"{args.tag}_test2.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(prediction_dir / name, name)
    zip_audit = audit_zip(zip_path)

    manifest = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": common.sha256_file(args.checkpoint),
        "checkpoint_metadata": {key: checkpoint_obj[key] for key in expected_metadata},
        "zip": str(zip_path),
        "zip_sha256": common.sha256_file(zip_path),
        "zip_audit": zip_audit,
        "protocol": "single checkpoint; D4 TTA8 FP32 native-grid mean; Bstar once; bilinear 1024; argmax+1",
        "package_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "tca_common_sha256": common.sha256_file(Path(common.__file__).resolve()),
        "config_sha256": common.sha256_file(paths.config),
        "gpu": gpu_receipt,
        "single_model": True,
        "checkpoint_count": 1,
        "ensemble": False,
        "teacher_used_for_candidate_inference": False,
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
        "runtime_sec": round(time.time() - started, 1),
    }
    common.write_json(args.output_dir / "PACKAGE_MANIFEST.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("checkpoint_sha256", "zip", "zip_sha256", "single_model")}, indent=2))


if __name__ == "__main__":
    main()
