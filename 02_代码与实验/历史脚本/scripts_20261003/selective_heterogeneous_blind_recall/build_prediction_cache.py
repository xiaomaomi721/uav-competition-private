#!/usr/bin/env python3
"""Build exact D4 native-grid prediction caches used by the SHBR gate."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import shbr_common as common


def build_b3(paths: common.ServerPaths, checkpoint: Path, device: str):
    for entry in (str(paths.code), str(paths.src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.environ["UAVSEG_MODEL_SOURCE"] = str(paths.model_source)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from src.models import build_model
    from src.utils import load_config

    config = load_config(str(paths.config))
    config["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0, "norm_eps": 1e-12, "start_step": 1}
    config.pop("mixstyle", None)
    config["pretrained_model_path"] = str(paths.model_source)
    model = build_model(config)
    model.load_state_dict(common.load_state(checkpoint), strict=True)
    return model.to(device).eval()


def b3_d4(model, image_path: Path, paths: common.ServerPaths, device: str) -> dict[str, np.ndarray]:
    for entry in (str(paths.code), str(paths.src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from src.transforms import normalize_to_tensor

    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    pixels = normalize_to_tensor(rgb).unsqueeze(0).to(device)
    accumulator = None
    with torch.inference_mode():
        for view in range(8):
            transformed = common.d4_forward(pixels, view)
            with torch.amp.autocast("cuda", enabled=False):
                logits = model(pixel_values=transformed).logits.float()
            canonical = common.d4_inverse(logits, view)[0]
            if tuple(canonical.shape) != (8, 256, 256):
                raise RuntimeError(f"unexpected B3 native shape for {image_path.name}: {tuple(canonical.shape)}")
            accumulator = canonical if accumulator is None else accumulator + canonical
    raw = accumulator / 8.0
    calibrated = raw + torch.from_numpy(common.BSTAR).to(device)[:, None, None]
    probabilities = calibrated.softmax(0)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError(f"non-finite B3 probabilities for {image_path.name}")
    confidence, prediction = probabilities.max(0)
    blind = torch.zeros_like(prediction, dtype=torch.bool)
    for class_id in common.BLIND_CLASSES:
        blind |= prediction == class_id
    selected = (confidence >= common.CONF_NORMAL) | (blind & (confidence >= common.CONF_BLIND))
    return {
        "probabilities": probabilities.cpu().numpy().astype(np.float32),
        "prediction": prediction.cpu().numpy().astype(np.uint8),
        "selected": selected.cpu().numpy().astype(np.uint8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--scope", choices=("source", "target"), required=True)
    parser.add_argument("--model", choices=("a05", "armb", "donor"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output exists: {args.output_dir}")
    settings = common.load_json(args.config)
    paths = common.ServerPaths.from_config(settings)
    preflight = common.load_json(args.preflight)
    gpu = preflight["selected_gpu"]
    gpu_receipt = common.verify_visible_gpu(int(gpu["index"]), gpu["uuid"])
    args.output_dir.mkdir(parents=True, exist_ok=False)

    split = common.load_json(paths.source_audit_split)
    if args.scope == "source":
        stems = [str(item) for item in split["H300"] + split["R1099"]]
        image_paths = [paths.source_images / f"{stem}.png" for stem in stems]
        names = stems
    else:
        names = [Path(name).stem for name in common.exact_test2_names()]
        image_paths = [paths.test2_images / f"{stem}.png" for stem in names]
    if len(set(names)) != len(names) or not all(path.is_file() for path in image_paths):
        raise RuntimeError(f"{args.scope} image identity failed")

    checkpoint_before = None
    if args.model in ("a05", "armb"):
        checkpoint = paths.a05 if args.model == "a05" else paths.armb
        checkpoint_before = common.sha256_file(checkpoint)
        expected = common.EXPECTED_A05_SHA if args.model == "a05" else common.EXPECTED_ARMB_SHA
        if checkpoint_before != expected:
            raise RuntimeError(f"{args.model} checkpoint SHA mismatch")
        predictor = build_b3(paths, checkpoint, args.device)
        predictor_receipt = {
            "family": "SegFormer-B3",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_before,
            "protocol": "D4 TTA8 FP32 native [8,256,256] raw-logit mean; Bstar once; softmax",
            "bstar": common.BSTAR.tolist(),
        }
    else:
        from donor_adapter import M2FDonorAdapter

        donor_config = Path(preflight["resolved_donor_config"])
        checkpoint_before = common.sha256_file(paths.donor_checkpoint)
        if checkpoint_before != preflight["artifacts"]["donor_checkpoint"]["sha256"]:
            raise RuntimeError("donor checkpoint changed after preflight")
        predictor = M2FDonorAdapter(
            donor_config,
            paths.donor_checkpoint,
            settings["donor"],
            device=args.device,
        )
        predictor_receipt = predictor.receipt()

    quantization_argmax_changes = 0
    total_pixels = 0
    started = time.time()
    for index, (name, image_path) in enumerate(zip(names, image_paths), 1):
        if args.model == "donor":
            value = predictor.infer_d4(image_path)
            probabilities = value["probabilities"]
            half = probabilities.astype(np.float16)
            quantization_argmax_changes += int((probabilities.argmax(0) != half.astype(np.float32).argmax(0)).sum())
            total_pixels += probabilities.shape[-2] * probabilities.shape[-1]
            np.savez_compressed(
                args.output_dir / f"{name}.npz",
                probabilities=half,
                prediction=value["prediction"],
                agreement=value["agreement"],
                margin=value["margin"].astype(np.float16),
            )
        else:
            value = b3_d4(predictor, image_path, paths, args.device)
            probabilities = value["probabilities"]
            half = probabilities.astype(np.float16)
            quantization_argmax_changes += int((probabilities.argmax(0) != half.astype(np.float32).argmax(0)).sum())
            total_pixels += probabilities.shape[-2] * probabilities.shape[-1]
            payload = {"prediction": value["prediction"]}
            if args.model == "a05":
                payload["probabilities"] = half
                payload["selected"] = value["selected"]
            np.savez_compressed(args.output_dir / f"{name}.npz", **payload)
        if index % 50 == 0 or index == len(names):
            print(json.dumps({"model": args.model, "scope": args.scope, "done": index, "total": len(names)}), flush=True)

    checkpoint_after = (
        common.sha256_file(paths.donor_checkpoint)
        if args.model == "donor"
        else common.sha256_file(paths.a05 if args.model == "a05" else paths.armb)
    )
    if checkpoint_after != checkpoint_before:
        raise RuntimeError(f"{args.model} checkpoint changed during inference")
    manifest = common.content_manifest(args.output_dir)
    metadata = {
        "schema_version": 1,
        "status": "COMPLETE",
        "scope": args.scope,
        "model": args.model,
        "count": len(names),
        "ordered_names_sha256": common.sha256_lines(names),
        "predictor": predictor_receipt,
        "checkpoint_stable": True,
        "cache": manifest,
        "stored_probabilities": "FP16" if args.model in ("a05", "donor") else None,
        "fp16_argmax_changes": quantization_argmax_changes,
        "fp16_argmax_change_fraction": quantization_argmax_changes / total_pixels if total_pixels else 0.0,
        "gpu": gpu_receipt,
        "script_sha256": common.sha256_file(Path(__file__).resolve()),
        "elapsed_sec": round(time.time() - started, 1),
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    common.write_json(args.output_dir / "META.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
