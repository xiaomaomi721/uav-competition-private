#!/usr/bin/env python3
"""Source-GT collapse guard for frozen CONTROL versus SHBR TREAT.

This report is not a target-domain ranking signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import shbr_common as common
from build_prediction_cache import b3_d4, build_b3


def evaluate(checkpoint: Path, stems: list[str], paths: common.ServerPaths, encoding: str, device: str) -> dict:
    model = build_b3(paths, checkpoint, device)
    confusion = np.zeros((8, 8), dtype=np.int64)
    for index, stem in enumerate(stems, 1):
        prediction = b3_d4(model, paths.source_images / f"{stem}.png", paths, device)["prediction"]
        with Image.open(paths.source_masks / f"{stem}.png") as image:
            raw = np.asarray(image.resize((256, 256), resample=Image.Resampling.NEAREST)).copy()
        gt = common.canonical_gt_to_train(raw, encoding)
        confusion += common.confusion_matrix(gt, prediction)
        if index % 100 == 0:
            print(json.dumps({"checkpoint": checkpoint.name, "source_health": index, "total": len(stems)}), flush=True)
    per_class, miou = common.iou_from_confusion(confusion)
    del model
    torch.cuda.empty_cache()
    return {"miou": miou, "per_class_iou": per_class.tolist(), "confusion": confusion.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--treat-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output exists: {args.output}")
    settings = common.load_json(args.config)
    paths = common.ServerPaths.from_config(settings)
    preflight = common.load_json(args.preflight)
    gpu = preflight["selected_gpu"]
    gpu_receipt = common.verify_visible_gpu(int(gpu["index"]), gpu["uuid"])
    if common.sha256_file(paths.control_checkpoint) != common.EXPECTED_CONTROL_CHECKPOINT_SHA:
        raise RuntimeError("frozen CONTROL checkpoint SHA mismatch")
    checkpoint = torch.load(args.treat_checkpoint, map_location="cpu", weights_only=False)
    required = {
        "step": 2000,
        "recipe": "selective_heterogeneous_blind_recall",
        "formula_version": common.FORMULA_VERSION,
        "role": "treat",
        "source_coeff": 0.1,
        "target_coeff": 1.0,
        "beta": common.BETA,
    }
    mismatch = {key: (checkpoint.get(key), value) for key, value in required.items() if checkpoint.get(key) != value}
    if mismatch:
        raise RuntimeError(f"TREAT checkpoint metadata mismatch: {mismatch}")
    split = common.load_json(paths.source_audit_split)
    stems = [str(item) for item in split["H300"] + split["R1099"]]
    control = evaluate(paths.control_checkpoint, stems, paths, settings["source_gt_encoding"], args.device)
    treat = evaluate(args.treat_checkpoint, stems, paths, settings["source_gt_encoding"], args.device)
    control_iou = np.asarray(control["per_class_iou"], dtype=np.float64)
    treat_iou = np.asarray(treat["per_class_iou"], dtype=np.float64)
    delta = treat_iou - control_iou
    failures = []
    if treat["miou"] - control["miou"] < -0.05:
        failures.append("source_miou_drop>5pp")
    finite = delta[np.isfinite(delta)]
    if finite.size == 0 or finite.min() < -0.15:
        failures.append("source_class_iou_drop>15pp")
    confusion = np.asarray(treat["confusion"])
    if np.any(confusion.sum(0) == 0):
        failures.append("predicted_class_disappeared")
    report = {
        "schema_version": 1,
        "warning": "Source GT is used only as a catastrophic-collapse guard, never to rank target submissions.",
        "status": "PASS_SOURCE_HEALTH" if not failures else "STOP_COLLAPSE",
        "control_checkpoint_sha256": common.EXPECTED_CONTROL_CHECKPOINT_SHA,
        "treat_checkpoint_sha256": common.sha256_file(args.treat_checkpoint),
        "control": control,
        "treat": treat,
        "delta_miou": treat["miou"] - control["miou"],
        "per_class_delta_iou": delta.tolist(),
        "failures": failures,
        "gpu": gpu_receipt,
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    common.write_json(args.output, report)
    print(json.dumps({"status": report["status"], "delta_miou": report["delta_miou"], "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
