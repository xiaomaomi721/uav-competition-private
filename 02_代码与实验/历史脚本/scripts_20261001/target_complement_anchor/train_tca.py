#!/usr/bin/env python3
"""Train the single pre-registered Target Complement Anchor candidate.

No source images are loaded.  The original Arm B selected-mask KL remains the
main loss; a gradient-calibrated weak KL on the exact complement anchors the
student to A05 on target-domain images.
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


def build_model_and_config(paths: common.ServerPaths, device: str):
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
    common.configure_trainable_scope(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/uav"))
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    if not 1 <= args.max_steps <= 2000:
        raise SystemExit("--max-steps must be in 1..2000; step2000 is the only formal endpoint")
    gpu_receipt = common.verify_gpu(args.physical_gpu, args.expected_gpu_uuid)
    paths = common.ServerPaths.from_root(args.root)
    common.validate_server_paths(paths)
    cache_receipt = common.validate_teacher_cache(paths.teacher_cache, full_scan=True)
    if not args.calibration_receipt.is_file():
        raise SystemExit(f"missing calibration receipt: {args.calibration_receipt}")
    calibration = common.load_json(args.calibration_receipt)
    common_path = Path(common.__file__).resolve()
    mu = common.validate_calibration_receipt(
        calibration,
        common.EXPECTED_A05_SHA,
        cache_receipt["content_manifest_sha256"],
        common.sha256_file(common_path),
    )
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    for entry in (str(paths.root / "code"), str(paths.src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from PIL import Image
    from torch.utils.data import DataLoader
    from src.transforms import normalize_to_tensor

    common.reset_seeds(common.SEED)
    model = build_model_and_config(paths, args.device)
    base_parameters, head_parameters = common.split_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": common.BASE_LR},
            {"params": head_parameters, "lr": common.BASE_LR * common.HEAD_MULT},
        ],
        weight_decay=common.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    bias = torch.from_numpy(common.BSTAR).to(args.device)[None, :, None, None]

    class TargetDataset:
        def __init__(self) -> None:
            self.names = common.exact_test2_names()

        def __len__(self) -> int:
            return len(self.names)

        def __getitem__(self, index: int) -> dict:
            name = self.names[index]
            with Image.open(paths.test2 / name) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            return {
                "pixel_values": normalize_to_tensor(rgb),
                "name": name,
            }

    def collate(batch: list[dict]) -> dict:
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "name": [item["name"] for item in batch],
        }

    loader = DataLoader(
        TargetDataset(),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(common.SEED + 10002),
        persistent_workers=True,
    )
    iterator = iter(loader)
    history_tail: list[dict] = []
    amp_skips: list[int] = []
    sample_sequence: list[str] = []
    view_sequence: list[int] = []
    started = time.time()

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        for group_index, group in enumerate(optimizer.param_groups):
            base = common.BASE_LR if group_index == 0 else common.BASE_LR * common.HEAD_MULT
            group["lr"] = common.learning_rate(step, base)
        optimizer.zero_grad(set_to_none=True)

        name = batch["name"][0]
        pixels = batch["pixel_values"].to(args.device)
        view = random.randrange(8)
        augmented = common.strong_photo(common.d4_forward(pixels, view))
        with torch.amp.autocast("cuda", enabled=True):
            student_logits = model(pixel_values=augmented).logits
        student_logits = common.d4_inverse(student_logits.float(), view) + bias

        cache_path = paths.teacher_cache / f"{Path(name).stem}.npy"
        teacher_raw = np.load(cache_path, allow_pickle=False)
        if teacher_raw.shape != (8, 256, 256) or teacher_raw.dtype != np.float32 or not np.isfinite(teacher_raw).all():
            raise RuntimeError(f"invalid teacher cache during training: {cache_path}")
        teacher_logits = torch.from_numpy(teacher_raw).to(args.device)[None] + bias
        teacher_probabilities = teacher_logits.softmax(1)
        confidence, predicted, selected, complement = common.teacher_masks(teacher_probabilities)
        kl = common.per_pixel_kl(teacher_probabilities, student_logits.log_softmax(1))
        main_loss = common.masked_mean(kl, selected, "selected")
        rest_loss = common.masked_mean(kl, complement, "complement")
        ramp = common.ramp_coefficient(step)
        loss = ramp * (main_loss + mu * rest_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite total loss at step {step}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        stepped = scale_after >= scale_before
        if not stepped:
            amp_skips.append(step)

        row = {
            "step": step,
            "name": name,
            "view": view,
            "main_loss": float(main_loss.detach().cpu()),
            "rest_loss": float(rest_loss.detach().cpu()),
            "mu": mu,
            "ramp": ramp,
            "loss": float(loss.detach().cpu()),
            "selected_fraction": float(selected.float().mean().detach().cpu()),
            "complement_fraction": float(complement.float().mean().detach().cpu()),
            "full_teacher_grid_partition": bool(torch.all(selected ^ complement).item()),
            "mean_confidence": float(confidence.mean().detach().cpu()),
            "predicted_class_counts": torch.bincount(predicted.reshape(-1), minlength=8).cpu().tolist(),
            "base_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "amp_scale_before": scale_before,
            "amp_scale_after": scale_after,
            "stepped": stepped,
            "elapsed_sec": round(time.time() - started, 1),
        }
        history_tail.append(row)
        history_tail = history_tail[-50:]
        sample_sequence.append(name)
        view_sequence.append(view)
        if step % 50 == 0 or step == 1:
            print(json.dumps(row), flush=True)

        if step in (1000, 2000):
            checkpoint_path = args.output_dir / f"checkpoint_step{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "recipe": "target_complement_anchor",
                    "formula_version": common.FORMULA_VERSION,
                    "black_region_training_exclusion": False,
                    "full_teacher_grid_partition": True,
                    "a05_sha256": common.EXPECTED_A05_SHA,
                    "teacher_cache_content_manifest_sha256": cache_receipt["content_manifest_sha256"],
                    "calibration_receipt_payload_sha256": calibration["receipt_payload_sha256"],
                    "mu": mu,
                    "source_coeff": 0.0,
                    "target_coeff": 1.0,
                    "conf_normal": common.CONF_NORMAL,
                    "conf_blind": common.CONF_BLIND,
                    "blind_classes": list(common.BLIND_CLASSES),
                    "attempted_loops": step,
                    "successful_optimizer_updates": step - len(amp_skips),
                    "amp_skip_loops": list(amp_skips),
                },
                checkpoint_path,
            )
            print(f"saved {checkpoint_path}", flush=True)

    script_path = Path(__file__).resolve()
    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "formal_endpoint": args.max_steps == 2000,
        "attempted_loops": args.max_steps,
        "successful_optimizer_updates": args.max_steps - len(amp_skips),
        "amp_skip_loops": amp_skips,
        "seed": common.SEED,
        "recipe": "target_complement_anchor",
        "formula_version": common.FORMULA_VERSION,
        "black_region_training_exclusion": False,
        "full_teacher_grid_partition": True,
        "mu": mu,
        "source_coeff": 0.0,
        "target_coeff": 1.0,
        "a05_sha256": common.EXPECTED_A05_SHA,
        "teacher_cache_content_manifest_sha256": cache_receipt["content_manifest_sha256"],
        "calibration_receipt_path": str(args.calibration_receipt),
        "calibration_receipt_sha256": common.sha256_file(args.calibration_receipt),
        "calibration_receipt_payload_sha256": calibration["receipt_payload_sha256"],
        "tca_common_sha256": common.sha256_file(common_path),
        "train_script_sha256": common.sha256_file(script_path),
        "config_sha256": common.sha256_file(paths.config),
        "gpu": gpu_receipt,
        "sample_sequence_sha256": common.canonical_json_sha256({"samples": sample_sequence}),
        "view_sequence_sha256": common.canonical_json_sha256({"views": view_sequence}),
        "history_tail": history_tail,
        "elapsed_sec": round(time.time() - started, 1),
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    common.write_json(args.output_dir / "RUN_SUMMARY.json", summary)
    common.write_json(
        args.output_dir / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "formal_endpoint": summary["formal_endpoint"],
            "checkpoint_step2000_exists": (args.output_dir / "checkpoint_step2000.pt").is_file(),
            "platform_submission_performed": False,
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
