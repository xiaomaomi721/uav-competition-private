#!/usr/bin/env python3
"""Train SHBR CONTROL/TREAT with the Arm B recipe and one donor-only variable."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import shbr_common as common
from build_prediction_cache import build_b3


def reference_points(reference: dict) -> dict[int, dict]:
    candidates = []
    for key in ("replay_points", "logged_history", "history", "history_tail"):
        value = reference.get(key)
        if isinstance(value, list):
            candidates.extend(row for row in value if isinstance(row, dict))
    result = {}
    for row in candidates:
        if "step" not in row:
            continue
        name = row.get("name") or row.get("target_stem") or row.get("stem")
        if name is None:
            continue
        result[int(row["step"])] = {
            "name": f"{Path(str(name)).stem}.png",
            "view": int(row["view"]) if "view" in row else None,
            "mask_cov": row.get("mask_cov", row.get("selected_fraction")),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--beta-receipt", type=Path, required=True)
    parser.add_argument("--role", choices=("control", "treat"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=common.MAX_STEPS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= common.MAX_STEPS:
        raise SystemExit("max steps must be in 1..2000")
    if args.output_dir.exists():
        raise SystemExit(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = common.load_json(args.config)
    paths = common.ServerPaths.from_config(settings)
    preflight = common.load_json(args.preflight)
    gpu = preflight["selected_gpu"]
    gpu_receipt = common.verify_visible_gpu(int(gpu["index"]), gpu["uuid"])
    support = common.load_json(args.target_cache / "TARGET_SUPPORT.json")
    if support.get("status") != "PROCEED_TARGET_SUPPORT":
        raise RuntimeError(f"target support did not pass: {support.get('status')}")
    beta_receipt = common.load_json(args.beta_receipt)
    if beta_receipt.get("status") != "PROCEED_TREATMENT_SCALE" or beta_receipt.get("beta") != common.BETA:
        raise RuntimeError("fixed-beta receipt did not pass")

    reference = common.load_json(paths.control_reference)
    if int(reference.get("attempted_loops", reference.get("steps", -1))) != 2000:
        raise RuntimeError("frozen CONTROL reference has no 2000-loop identity")
    if int(reference.get("seed", common.SEED)) != common.SEED:
        raise RuntimeError("frozen CONTROL seed mismatch")
    points = reference_points(reference)
    if args.max_steps == common.MAX_STEPS and len(points) < 10:
        raise RuntimeError("STOP_MISSING_CONTROL: fewer than ten frozen CONTROL replay points")

    for entry in (str(paths.code), str(paths.src)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from src.dataset import UAVSegmentationDataset, segmentation_collate
    from src.rare_crop import RareClassCropper
    from src.transforms import TrainTransform, normalize_to_tensor
    from src.utils import load_config

    common.reset_seeds(common.SEED)
    config = load_config(str(paths.config))
    config["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0, "norm_eps": 1e-12, "start_step": 1}
    config.pop("mixstyle", None)
    config["pretrained_model_path"] = str(paths.model_source)
    model = build_b3(paths, paths.a05, args.device)
    base_parameters, head_parameters = common.configure_trainable_scope(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": common.BASE_LR},
            {"params": head_parameters, "lr": common.BASE_LR * common.HEAD_MULT},
        ],
        weight_decay=common.WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    bias = torch.from_numpy(common.BSTAR).to(args.device)[None, :, None, None]

    source_ids = [item for item in paths.source_train_split.read_text(encoding="utf-8-sig").split() if item]
    if len(source_ids) != 5597:
        raise RuntimeError(f"expected 5597 source IDs, got {len(source_ids)}")
    cropper = RareClassCropper(
        crop_size=1024,
        rare_probability=0.5,
        priority_ids=(7, 4, 6),
        max_offset_fraction=0.25,
    )
    source_dataset = UAVSegmentationDataset(
        paths.root / "datasets/训练集",
        source_ids,
        TrainTransform(crop_size=1024, config=config["transforms"], rare_cropper=cropper, ignore_index=255),
        maximum=None,
        seed=common.SEED,
        random_subset=False,
    )
    source_loader = DataLoader(
        source_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=segmentation_collate,
        generator=torch.Generator().manual_seed(common.SEED + 10001),
        persistent_workers=True,
    )

    class TargetDataset:
        names = common.exact_test2_names()

        def __len__(self):
            return len(self.names)

        def __getitem__(self, index):
            name = self.names[index]
            with Image.open(paths.test2_images / name) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            return {"pixel_values": normalize_to_tensor(rgb), "name": name}

    def target_collate(batch):
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "name": [item["name"] for item in batch],
        }

    target_loader = DataLoader(
        TargetDataset(),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=target_collate,
        generator=torch.Generator().manual_seed(common.SEED + 10002),
        persistent_workers=True,
    )
    source_iterator = iter(source_loader)
    target_iterator = iter(target_loader)
    trace_path = args.output_dir / "TRAIN_TRACE.jsonl"
    trace_handle = trace_path.open("x", encoding="utf-8", buffering=1)
    sample_digest = hashlib.sha256()
    view_digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    amp_skips = []
    replay_mismatches = []
    history_tail = []
    started = time.time()
    try:
        for step in range(1, args.max_steps + 1):
            try:
                source_batch = next(source_iterator)
            except StopIteration:
                source_iterator = iter(source_loader)
                source_batch = next(source_iterator)
            try:
                target_batch = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_batch = next(target_iterator)

            for group_index, group in enumerate(optimizer.param_groups):
                base_lr = common.BASE_LR if group_index == 0 else common.BASE_LR * common.HEAD_MULT
                group["lr"] = common.learning_rate(step, base_lr)
            optimizer.zero_grad(set_to_none=True)

            source_pixels = source_batch["pixel_values"].to(args.device)
            source_labels = source_batch["labels"].to(args.device)
            with torch.amp.autocast("cuda", enabled=True):
                source_logits = model(pixel_values=source_pixels).logits
            source_loss = F.cross_entropy(
                F.interpolate(source_logits.float(), size=source_labels.shape[-2:], mode="bilinear", align_corners=False),
                source_labels,
                ignore_index=common.IGNORE_INDEX,
            )

            name = target_batch["name"][0]
            stem = Path(name).stem
            target_pixels = target_batch["pixel_values"].to(args.device)
            view = random.randrange(8)
            augmented = common.strong_photo(common.d4_forward(target_pixels, view))
            with torch.amp.autocast("cuda", enabled=True):
                student_logits = model(pixel_values=augmented).logits
            student_logits = common.d4_inverse(student_logits.float(), view) + bias

            teacher_raw = np.load(paths.a05_target_cache / f"{stem}.npy", allow_pickle=False)
            if teacher_raw.shape != (8, 256, 256) or teacher_raw.dtype != np.float32 or not np.isfinite(teacher_raw).all():
                raise RuntimeError(f"invalid A05 teacher cache: {stem}")
            teacher = (torch.from_numpy(teacher_raw).to(args.device)[None] + bias).softmax(1)
            _, _, selected = common.teacher_selected(teacher)
            with np.load(args.target_cache / f"{stem}.npz", allow_pickle=False) as archive:
                donor = torch.from_numpy(archive["donor_probabilities"].astype(np.float32)).to(args.device)[None]
                gate = torch.from_numpy(archive["gate"].astype(bool)).to(args.device)[None]
            student_log = student_logits.log_softmax(1)
            a05_kl = common.per_pixel_kl(teacher, student_log)
            donor_kl = common.per_pixel_kl(donor, student_log)
            ramp = common.ramp_coefficient(step)
            losses = common.shbr_losses(source_loss, a05_kl, selected, donor_kl, gate, ramp, args.role)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            stepped = scale_after >= scale_before
            if not stepped:
                amp_skips.append(step)

            selected_fraction = float(selected.float().mean().detach().cpu())
            row = {
                "step": step,
                "role": args.role,
                "name": name,
                "view": view,
                "source_loss": float(source_loss.detach().cpu()),
                "main_loss": float(losses["main"].detach().cpu()),
                "donor_loss_normalized_by_M": float(losses["donor"].detach().cpu()),
                "donor_coefficient": float(losses["donor_coefficient"].detach().cpu()),
                "selected_fraction": selected_fraction,
                "gate_fraction": float(gate.float().mean().detach().cpu()),
                "ramp": ramp,
                "total_loss": float(losses["total"].detach().cpu()),
                "base_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
                "amp_scale_before": scale_before,
                "amp_scale_after": scale_after,
                "stepped": stepped,
                "elapsed_sec": round(time.time() - started, 1),
            }
            trace_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            history_tail.append(row)
            history_tail = history_tail[-50:]
            sample_digest.update(f"{step}\t{name}\n".encode("utf-8"))
            view_digest.update(f"{step}\t{view}\n".encode("utf-8"))
            labels_cpu = source_batch["labels"].numpy()
            source_digest.update(
                f"{step}\t{labels_cpu.shape}\t{int((labels_cpu != 255).sum())}\t{int(labels_cpu[labels_cpu != 255].sum())}\n".encode(
                    "utf-8"
                )
            )
            if step in points:
                expected = points[step]
                mismatch = {}
                if expected["name"] != name:
                    mismatch["name"] = [expected["name"], name]
                if expected["view"] is not None and expected["view"] != view:
                    mismatch["view"] = [expected["view"], view]
                if expected["mask_cov"] is not None and abs(float(expected["mask_cov"]) - selected_fraction) > 1e-9:
                    mismatch["mask_cov"] = [float(expected["mask_cov"]), selected_fraction]
                if mismatch:
                    replay_mismatches.append({"step": step, "mismatch": mismatch})
            if step % 50 == 0 or step == 1:
                print(json.dumps(row), flush=True)
    finally:
        trace_handle.close()

    if args.max_steps == common.MAX_STEPS and replay_mismatches:
        raise RuntimeError(f"STOP_REPLAY_MISMATCH: {replay_mismatches[:3]}")
    checkpoint_path = None
    if args.max_steps == common.MAX_STEPS:
        checkpoint_path = args.output_dir / "checkpoint_step2000.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "step": common.MAX_STEPS,
                "recipe": "selective_heterogeneous_blind_recall",
                "formula_version": common.FORMULA_VERSION,
                "role": args.role,
                "a05_sha256": common.EXPECTED_A05_SHA,
                "source_coeff": 0.1,
                "target_coeff": 1.0,
                "beta": common.BETA,
                "donor_coefficient": 0.0 if args.role == "control" else common.BETA,
                "donor_denominator": "A05 selected count |M|",
                "attempted_loops": args.max_steps,
                "successful_optimizer_updates": args.max_steps - len(amp_skips),
                "amp_skip_loops": amp_skips,
                "target_cache_manifest_sha256": support["cache_manifest"]["sha256"],
                "beta_receipt_payload_sha256": beta_receipt["receipt_payload_sha256"],
            },
            checkpoint_path,
        )

    summary = {
        "schema_version": 1,
        "status": "COMPLETE",
        "formal_endpoint": args.max_steps == common.MAX_STEPS,
        "role": args.role,
        "formula_version": common.FORMULA_VERSION,
        "attempted_loops": args.max_steps,
        "successful_optimizer_updates": args.max_steps - len(amp_skips),
        "amp_skip_loops": amp_skips,
        "seed": common.SEED,
        "source_coeff": 0.1,
        "target_coeff": 1.0,
        "beta": common.BETA,
        "donor_coefficient": 0.0 if args.role == "control" else common.BETA,
        "sample_sequence_sha256": sample_digest.hexdigest(),
        "view_sequence_sha256": view_digest.hexdigest(),
        "source_sequence_summary_sha256": source_digest.hexdigest(),
        "control_reference": str(paths.control_reference),
        "control_reference_sha256": common.sha256_file(paths.control_reference),
        "control_replay_points": len(points),
        "replay_mismatches": replay_mismatches,
        "target_cache_manifest": support["cache_manifest"],
        "beta_receipt_payload_sha256": beta_receipt["receipt_payload_sha256"],
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": common.sha256_file(checkpoint_path) if checkpoint_path else None,
        "trace_sha256": common.sha256_file(trace_path),
        "history_tail": history_tail,
        "gpu": gpu_receipt,
        "elapsed_sec": round(time.time() - started, 1),
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    common.write_json(args.output_dir / "RUN_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
