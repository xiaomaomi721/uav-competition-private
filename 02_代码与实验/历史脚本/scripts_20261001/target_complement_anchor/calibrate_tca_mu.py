#!/usr/bin/env python3
"""Freeze the single TCA complement weight from target-domain gradient ratios.

This script never updates model parameters.  It binds the chosen mu to the
A05 checkpoint, the full teacher-cache content manifest, the exact sampled
images, and the shared TCA implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import tca_common as common

CALIBRATION_SEED = common.SEED + 20000
SAMPLE_SEED = common.SEED + 30000


def build_model(paths: common.ServerPaths, device: str):
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
    model.load_state_dict(common.load_state(paths.a05), strict=True)
    model = model.to(device)
    trainable = common.configure_trainable_scope(model)
    return model, trainable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/uav"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--target-grad-ratio", type=float, default=0.10)
    parser.add_argument("--mu-min", type=float, default=0.02)
    parser.add_argument("--mu-max", type=float, default=0.15)
    args = parser.parse_args()

    if not 1 <= args.samples <= 1300:
        raise SystemExit("--samples must be in 1..1300")
    if not 0 < args.target_grad_ratio < 1:
        raise SystemExit("--target-grad-ratio must be in (0,1)")
    if not 0 < args.mu_min <= args.mu_max < 1:
        raise SystemExit("invalid mu bounds")

    gpu_receipt = common.verify_gpu(args.physical_gpu, args.expected_gpu_uuid)
    paths = common.ServerPaths.from_root(args.root)
    common.validate_server_paths(paths)
    cache_receipt = common.validate_teacher_cache(paths.teacher_cache, full_scan=True)
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    for entry in (str(paths.root / "code"), str(paths.src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from PIL import Image
    from src.transforms import normalize_to_tensor

    rng = np.random.default_rng(SAMPLE_SEED)
    indices = sorted(int(value) for value in rng.choice(np.arange(1, 1301), size=args.samples, replace=False))
    names = [f"test2_{index}.png" for index in indices]

    common.reset_seeds(CALIBRATION_SEED)
    model, trainable = build_model(paths, args.device)
    bias = torch.from_numpy(common.BSTAR).to(args.device)[None, :, None, None]

    main_norms: list[float] = []
    rest_norms: list[float] = []
    rows: list[dict] = []
    sampled_cache_sha: dict[str, str] = {}
    started = time.time()

    for sample_index, name in enumerate(names, 1):
        with Image.open(paths.test2 / name) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        pixels = normalize_to_tensor(rgb).unsqueeze(0).to(args.device)
        view = random.randrange(8)
        augmented = common.strong_photo(common.d4_forward(pixels, view))
        with torch.amp.autocast("cuda", enabled=True):
            student_logits = model(pixel_values=augmented).logits
        student_logits = common.d4_inverse(student_logits.float(), view) + bias

        cache_path = paths.teacher_cache / f"{Path(name).stem}.npy"
        sampled_cache_sha[name] = common.sha256_file(cache_path)
        teacher_raw = np.load(cache_path, allow_pickle=False)
        if teacher_raw.shape != (8, 256, 256) or teacher_raw.dtype != np.float32 or not np.isfinite(teacher_raw).all():
            raise RuntimeError(f"invalid teacher cache during calibration: {cache_path}")
        teacher_logits = torch.from_numpy(teacher_raw).to(args.device)[None] + bias
        teacher_probabilities = teacher_logits.softmax(1)
        confidence, predicted, selected, complement = common.teacher_masks(teacher_probabilities)
        kl = common.per_pixel_kl(teacher_probabilities, student_logits.log_softmax(1))
        main_loss = common.masked_mean(kl, selected, "selected")
        rest_loss = common.masked_mean(kl, complement, "complement")

        main_grads = torch.autograd.grad(main_loss, trainable, retain_graph=True, allow_unused=True)
        rest_grads = torch.autograd.grad(rest_loss, trainable, retain_graph=False, allow_unused=True)
        main_norm = common.global_grad_norm(main_grads)
        rest_norm = common.global_grad_norm(rest_grads)
        main_norms.append(main_norm)
        rest_norms.append(rest_norm)

        row = {
            "name": name,
            "view": view,
            "main_loss": float(main_loss.detach().cpu()),
            "rest_loss": float(rest_loss.detach().cpu()),
            "main_grad_norm": main_norm,
            "rest_grad_norm": rest_norm,
            "main_over_rest": main_norm / rest_norm,
            "selected_fraction": float(selected.float().mean().detach().cpu()),
            "complement_fraction": float(complement.float().mean().detach().cpu()),
            "full_teacher_grid_partition": bool(torch.all(selected ^ complement).item()),
            "mean_confidence": float(confidence.mean().detach().cpu()),
            "predicted_class_counts": torch.bincount(predicted.reshape(-1), minlength=8).cpu().tolist(),
        }
        rows.append(row)
        if sample_index % 8 == 0 or sample_index == len(names):
            print(json.dumps({"calibrated": sample_index, "total": len(names), "elapsed_sec": round(time.time() - started, 1)}), flush=True)

        del pixels, augmented, student_logits, teacher_logits, teacher_probabilities, kl
        torch.cuda.empty_cache()

    frozen = common.freeze_mu(
        main_norms,
        rest_norms,
        target_grad_ratio=args.target_grad_ratio,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
    )
    common_path = Path(common.__file__).resolve()
    receipt = {
        "schema_version": 1,
        "formula_version": common.FORMULA_VERSION,
        "black_region_training_exclusion": False,
        "full_teacher_grid_partition": True,
        "purpose": "freeze TCA complement KL coefficient without parameter updates",
        "parameter_updates_performed": False,
        "a05_path": str(paths.a05),
        "a05_sha256": common.EXPECTED_A05_SHA,
        "teacher_cache_path": str(paths.teacher_cache),
        "teacher_cache_content_manifest_sha256": cache_receipt["content_manifest_sha256"],
        "teacher_cache_validation": cache_receipt,
        "sample_seed": SAMPLE_SEED,
        "calibration_seed": CALIBRATION_SEED,
        "sample_count": len(names),
        "sample_names": names,
        "sampled_cache_sha256": sampled_cache_sha,
        "tca_common_path": str(common_path),
        "tca_common_sha256": common.sha256_file(common_path),
        "calibration_script_sha256": common.sha256_file(Path(__file__).resolve()),
        "gpu": gpu_receipt,
        "elapsed_sec": round(time.time() - started, 1),
        "gradient_freeze": frozen,
        "mu": frozen["mu"],
        "rows": rows,
        "platform_submission_performed": False,
        "test2_ground_truth_used": False,
    }
    receipt["receipt_payload_sha256"] = common.canonical_json_sha256(receipt)
    common.write_json(args.output_dir / "CALIBRATION_RECEIPT.json", receipt)
    common.write_json(
        args.output_dir / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "mu": frozen["mu"],
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "platform_submission_performed": False,
        },
    )
    print(json.dumps({"status": "COMPLETE", "mu": frozen["mu"], "mu_raw": frozen["mu_raw"], "clamped": frozen["clamped"]}, indent=2))


if __name__ == "__main__":
    main()
