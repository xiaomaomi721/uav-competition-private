#!/usr/bin/env python3
"""Audit the fixed beta=0.25 gradient scale without updating parameters."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import shbr_common as common
from build_prediction_cache import build_b3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
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
    support = common.load_json(args.target_cache / "TARGET_SUPPORT.json")
    if support.get("status") != "PROCEED_TARGET_SUPPORT":
        raise RuntimeError(f"target support did not pass: {support.get('status')}")

    names = []
    for official_name in common.exact_test2_names():
        stem = Path(official_name).stem
        with np.load(args.target_cache / f"{stem}.npz", allow_pickle=False) as archive:
            if int(archive["gate"].sum()) > 0:
                names.append(official_name)
        if len(names) == 32:
            break
    if len(names) != 32:
        raise RuntimeError(f"need 32 gated images for fixed-beta audit, got {len(names)}")

    for entry in (str(paths.code), str(paths.src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from src.transforms import normalize_to_tensor

    common.reset_seeds(common.SEED)
    model = build_b3(paths, paths.a05, args.device)
    base, head = common.configure_trainable_scope(model)
    parameters = base + head
    bias = torch.from_numpy(common.BSTAR).to(args.device)[None, :, None, None]
    ratios = []
    rows = []
    for index, name in enumerate(names, 1):
        stem = Path(name).stem
        with Image.open(paths.test2_images / name) as image:
            pixels = normalize_to_tensor(np.asarray(image.convert("RGB")).copy()).unsqueeze(0).to(args.device)
        view = random.randrange(8)
        augmented = common.strong_photo(common.d4_forward(pixels, view))
        student = common.d4_inverse(model(pixel_values=augmented).logits.float(), view) + bias
        raw = np.load(paths.a05_target_cache / f"{stem}.npy", allow_pickle=False)
        teacher = (torch.from_numpy(raw).to(args.device)[None] + bias).softmax(1)
        _, _, selected = common.teacher_selected(teacher)
        with np.load(args.target_cache / f"{stem}.npz", allow_pickle=False) as archive:
            donor = torch.from_numpy(archive["donor_probabilities"].astype(np.float32)).to(args.device)[None]
            gate = torch.from_numpy(archive["gate"].astype(bool)).to(args.device)[None]
        student_log = student.log_softmax(1)
        a05_kl = common.per_pixel_kl(teacher, student_log)
        donor_kl = common.per_pixel_kl(donor, student_log)
        count = selected.sum()
        main = a05_kl[selected].sum() / count
        donor_term = common.BETA * donor_kl[gate].sum() / count
        main_grads = torch.autograd.grad(main, parameters, retain_graph=True, allow_unused=True)
        donor_grads = torch.autograd.grad(donor_term, parameters, allow_unused=True)
        main_norm = common.global_grad_norm(main_grads)
        donor_norm = common.global_grad_norm(donor_grads)
        ratio = donor_norm / main_norm
        if not np.isfinite(ratio):
            raise RuntimeError(f"non-finite fixed-beta gradient ratio for {name}")
        ratios.append(ratio)
        rows.append(
            {
                "name": name,
                "view": view,
                "selected_pixels": int(selected.sum().item()),
                "gate_pixels": int(gate.sum().item()),
                "main_grad_norm": main_norm,
                "beta_donor_grad_norm": donor_norm,
                "ratio": ratio,
            }
        )
        model.zero_grad(set_to_none=True)
        print(json.dumps({"beta_audit": index, "total": 32, "ratio": ratio}), flush=True)

    median = float(np.median(np.asarray(ratios, dtype=np.float64)))
    passed = 0.005 <= median <= 0.15
    receipt = {
        "schema_version": 1,
        "status": "PROCEED_TREATMENT_SCALE" if passed else "STOP_TREATMENT_SCALE",
        "formula_version": common.FORMULA_VERSION,
        "beta": common.BETA,
        "beta_was_adjusted": False,
        "image_selection": "first 32 numeric-order test2 images with non-empty frozen G",
        "image_names": names,
        "median_gradient_ratio": median,
        "allowed_interval": [0.005, 0.15],
        "ratios": ratios,
        "rows": rows,
        "target_cache_manifest": support["cache_manifest"],
        "gpu": gpu_receipt,
        "test2_ground_truth_used": False,
        "parameter_updates_performed": False,
        "platform_submission_performed": False,
    }
    unsigned = dict(receipt)
    receipt["receipt_payload_sha256"] = common.canonical_json_sha256(unsigned)
    common.write_json(args.output, receipt)
    print(json.dumps({"status": receipt["status"], "median_gradient_ratio": median}, indent=2))


if __name__ == "__main__":
    main()
