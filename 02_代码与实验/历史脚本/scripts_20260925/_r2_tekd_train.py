#!/usr/bin/env python3
"""Bounded TEKD: distill the scored theta278/D1 alpha-0.5 logit teacher into one B3.

Exactly four optimizer updates in smoke mode or 1000 in formal mode. No platform
submission, no target GT, no continuation, and no multi-model inference output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/home/jhy/rare_diag_20260915")
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
TEST2 = Path("/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images")
ATDA_CONFIG = ROOT / "atda_20260918/atda_s2s3.yaml"
D1_CKPT = ROOT / "round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt"
THETA_CACHE = ROOT / "round2_20260921/cache_278_test2_raw/scores"
D1_CACHE = ROOT / "round2_20260923/D1_2k_probe/raw_cache/scores"
THETA_META = THETA_CACHE.parent / "META.json"
D1_META = D1_CACHE.parent / "META.json"
EXPECTED_THETA_SHA = "5f8a092b92f1e5c68e7173131c946f50b1663ade1fd43fd71a2ce2d394f001dc"
EXPECTED_D1_SHA = "70f45afcddfb1711a0b8576f2e335ce5fa3f88df56c48724b761716e10f34b14"
EXPECTED_DATA_SHA = "c8784dba28f19b86e7043bced92e42f98b346ae3f3ea13326f51ef6af295b65a"
EXPECTED_GPU_UUID = "GPU-32853ccf-4c4f-1328-8d75-04efe9d5298f"
EXPECTED_GPU_BUS = "00000000:CA:00.0"
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
TRAIN_PREFIXES = ("segformer.stages.2.", "segformer.stages.3.")
HEAD_PREFIX = "decode_head."
BATCH_SIZE = 2
ACCUM_STEPS = 2
BASE_LR = 5e-6
HEAD_MULT = 0.05
WEIGHT_DECAY = 0.01
TEMPERATURE = 2.0
TARGET_WEIGHT = 0.5
RAMP_STEPS = 200
SEED = 20260925


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state(path: Path):
    import torch
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model", "student_model", "model_state_dict", "state_dict"):
            value = obj.get(key)
            if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
                return value, obj
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj, obj
    raise RuntimeError(f"no tensor state dict in {path}")


def assert_gpu3_binding() -> dict:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise SystemExit("TEKD requires physical GPU3 only")
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"],
        text=True,
    )
    rows = [x.strip() for x in query.splitlines() if x.strip()]
    match = [x for x in rows if x.startswith("3,")]
    if len(match) != 1 or EXPECTED_GPU_UUID not in match[0] or EXPECTED_GPU_BUS not in match[0]:
        raise SystemExit(f"physical GPU3 identity mismatch: {match}")
    return {"physical_index": 3, "uuid": EXPECTED_GPU_UUID, "pci_bus": EXPECTED_GPU_BUS, "row": match[0]}


def d4_forward(tensor, index: int):
    import torch
    rotation, mirror = VIEWS[index]
    if rotation:
        tensor = torch.rot90(tensor, rotation, (-2, -1))
    if mirror:
        tensor = tensor.flip((-1,))
    return tensor


def d4_inverse(tensor, index: int):
    import torch
    rotation, mirror = VIEWS[index]
    if mirror:
        tensor = tensor.flip((-1,))
    if rotation:
        tensor = torch.rot90(tensor, -rotation, (-2, -1))
    return tensor.contiguous()


def is_trainable(name: str) -> bool:
    return name.startswith(TRAIN_PREFIXES) or name.startswith(HEAD_PREFIX)


def apply_modes(model) -> None:
    model.eval()
    model.get_submodule("segformer.stages.2").train()
    model.get_submodule("segformer.stages.3").train()


def snapshot_frozen(model):
    return ({n: p.detach().cpu().clone() for n, p in model.named_parameters() if not is_trainable(n)} |
            {"buffer::" + n: b.detach().cpu().clone() for n, b in model.named_buffers()})


def frozen_diff(model, snapshot):
    import torch
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    bad = []
    for name, before in snapshot.items():
        current = buffers[name[8:]] if name.startswith("buffer::") else params[name]
        if not torch.equal(current.detach().cpu(), before):
            bad.append(name)
    return bad


def strong_photo(x):
    import torch
    import torch.nn.functional as F
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)[None, :, None, None]
    x = (x * std + mean).clamp(0, 1)
    brightness = 1 + random.uniform(-0.15, 0.15)
    contrast = 1 + random.uniform(-0.15, 0.15)
    saturation = 1 + random.uniform(-0.12, 0.12)
    gamma = random.uniform(0.9, 1.1)
    x = x * brightness
    center = x.mean((-2, -1), keepdim=True)
    x = (x - center) * contrast + center
    gray = x.mean(1, keepdim=True)
    x = (x - gray) * saturation + gray
    x = x.clamp(0, 1).pow(gamma)
    x = x + torch.randn_like(x) * random.uniform(0, 0.015)
    if random.random() < 0.25:
        x = F.avg_pool2d(x, 3, 1, 1)
    return (x.clamp(0, 1) - mean) / std


class TargetDataset:
    def __init__(self):
        self.names = [f"test2_{i}.png" for i in range(1, 1301)]
        if not all((TEST2 / n).is_file() for n in self.names):
            raise RuntimeError("test2 exact-name contract failed")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        from PIL import Image
        from src.transforms import normalize_to_tensor
        name = self.names[index]
        with Image.open(TEST2 / name) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        return {"pixel_values": normalize_to_tensor(rgb), "name": name}


def target_collate(batch):
    import torch
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "name": [x["name"] for x in batch],
    }


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def save_visualizations(model, source_dataset, out: Path, device: str) -> list[str]:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    vis = out / "prediction_visualizations"
    vis.mkdir()
    result = []
    model.eval()
    with torch.inference_mode():
        for index in range(8):
            sample = source_dataset[index]
            x = sample["pixel_values"].unsqueeze(0).to(device)
            y = sample["labels"]
            with torch.amp.autocast("cuda", enabled=False):
                z = model(pixel_values=x).logits.float()
            pred = F.interpolate(z, size=y.shape[-2:], mode="bilinear", align_corners=False)[0].argmax(0)
            path = vis / f"prediction_{index:02d}.png"
            Image.fromarray(pred.byte().cpu().numpy(), mode="L").save(path)
            result.append(str(path))
    apply_modes(model)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("smoke", "formal"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    expected_steps = 4 if args.mode == "smoke" else 1000
    if args.max_steps != expected_steps:
        raise SystemExit(f"{args.mode} requires exactly {expected_steps} optimizer updates")
    gpu_identity = assert_gpu3_binding()
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    if sha256_file(D1_CKPT) != EXPECTED_D1_SHA:
        raise SystemExit("D1 checkpoint SHA mismatch")
    cache_contracts = (
        ("old_theta278", THETA_META, EXPECTED_THETA_SHA),
        ("D1", D1_META, EXPECTED_D1_SHA),
    )
    for label, meta_path, expected_checkpoint_sha in cache_contracts:
        meta = json.loads(meta_path.read_text())
        if meta.get("checkpoint_sha256") != expected_checkpoint_sha:
            raise SystemExit(f"{label} cache checkpoint SHA mismatch")
        if meta.get("dataset_archive_sha256") != EXPECTED_DATA_SHA or meta.get("count") != 1300:
            raise SystemExit(f"{label} cache dataset/count mismatch")
        protocol = str(meta.get("protocol", ""))
        if "D4 TTA8" not in protocol or "no Bstar" not in protocol or "FP32 view mean" not in protocol:
            raise SystemExit(f"{label} cache protocol mismatch: {protocol}")
    for cache in (THETA_CACHE, D1_CACHE):
        names = sorted(x.name for x in cache.glob("test2_*.npy"))
        if len(names) != 1300:
            raise SystemExit(f"bad teacher cache count: {cache} => {len(names)}")
        for probe in ("test2_1.npy", "test2_1300.npy"):
            value = np.load(cache / probe, allow_pickle=False)
            if value.shape != (8, 256, 256) or not np.isfinite(value).all():
                raise SystemExit(f"bad teacher cache tensor: {cache / probe}")
    out.mkdir(parents=True)

    for path in (str(SNAP), str(ROOT / "round2_multiprobe_20260922"), "/home/jhy/gpu2_day_queue"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from src.dataset import UAVSegmentationDataset, segmentation_collate
    from src.models import build_model as make_model
    from src.rare_crop import RareClassCropper
    from src.transforms import TrainTransform
    from src.utils import load_config

    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg_model = load_config(str(CONFIG))
    cfg_model["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0, "norm_eps": 1e-12, "start_step": 1}
    cfg_model.pop("mixstyle", None)
    state, _ = load_state(D1_CKPT)
    model = make_model(cfg_model)
    model.load_state_dict(state, strict=True)
    model = model.to(args.device)

    base_params, head_params, train_names = [], [], []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_trainable(name))
        if name.startswith(TRAIN_PREFIXES):
            base_params.append(parameter)
            train_names.append(name)
        elif name.startswith(HEAD_PREFIX):
            head_params.append(parameter)
            train_names.append(name)
    frozen = snapshot_frozen(model)
    apply_modes(model)

    optimizer = torch.optim.AdamW(
        [
            {"name": "stage23", "params": base_params, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
            {"name": "head", "params": head_params, "lr": BASE_LR * HEAD_MULT, "weight_decay": WEIGHT_DECAY},
        ]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
    bstar = torch.from_numpy(BSTAR).to(args.device)[None, :, None, None]

    train_cfg = load_config(str(ATDA_CONFIG))
    source_ids = [x.strip() for x in (SNAP / "splits/train.txt").read_text().split() if x.strip()]
    if len(source_ids) != 5597:
        raise RuntimeError(f"source split count {len(source_ids)}")
    cropper = RareClassCropper(crop_size=1024, rare_probability=0.5, priority_ids=(7, 4, 6), max_offset_fraction=0.25)
    source = UAVSegmentationDataset(
        DATA,
        source_ids,
        TrainTransform(crop_size=1024, config=train_cfg["transforms"], rare_cropper=cropper, ignore_index=255),
        maximum=None,
        seed=seed,
        random_subset=False,
    )
    source_loader = DataLoader(
        source, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True, collate_fn=segmentation_collate,
        generator=torch.Generator().manual_seed(seed + 10001), persistent_workers=True,
    )
    target_loader = DataLoader(
        TargetDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
        drop_last=True, collate_fn=target_collate,
        generator=torch.Generator().manual_seed(seed + 10002), persistent_workers=True,
    )

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "max_optimizer_updates": expected_steps,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "effective_source_images_per_update": BATCH_SIZE * ACCUM_STEPS,
        "effective_target_images_per_update": BATCH_SIZE * ACCUM_STEPS,
        "seed": seed,
        "init_checkpoint": str(D1_CKPT),
        "init_sha256": EXPECTED_D1_SHA,
        "teacher": "0.5*old_theta278_native_D4_cache + 0.5*D1_native_D4_cache",
        "theta_cache": str(THETA_CACHE),
        "d1_cache": str(D1_CACHE),
        "temperature": TEMPERATURE,
        "target_weight": TARGET_WEIGHT,
        "ramp_steps": RAMP_STEPS,
        "bstar": BSTAR.tolist(),
        "trainable_prefixes": list(TRAIN_PREFIXES) + [HEAD_PREFIX],
        "cwfa_enabled": False,
        "gpu": gpu_identity,
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
        "automatic_continuation": False,
    }
    write_json(out / "RUN_MANIFEST.json", manifest)
    optimizer_manifest = {
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "groups": [
            {"name": "stage23", "lr": BASE_LR, "weight_decay": WEIGHT_DECAY, "parameters": sum(p.numel() for p in base_params)},
            {"name": "head", "lr": BASE_LR * HEAD_MULT, "weight_decay": WEIGHT_DECAY, "parameters": sum(p.numel() for p in head_params)},
        ],
        "trainable_names": train_names,
    }
    write_json(out / "OPTIMIZER_GROUPS.json", optimizer_manifest)

    columns = ["step", "microsteps", "source_ce", "target_kl", "target_lambda", "total_loss", "lr_stage23", "lr_head", "amp_stepped", "scale", "elapsed_sec"]
    log_file = (out / "train_log.csv").open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(log_file, fieldnames=columns)
    writer.writeheader()
    log_file.flush()

    source_iter, target_iter = iter(source_loader), iter(target_loader)
    amp_skips = []
    started = time.time()
    last_row = None
    for step in range(1, expected_steps + 1):
        lr_scale = 1.0 - (step - 1) / 10000.0
        optimizer.param_groups[0]["lr"] = BASE_LR * lr_scale
        optimizer.param_groups[1]["lr"] = BASE_LR * HEAD_MULT * lr_scale
        optimizer.zero_grad(set_to_none=True)
        source_sum = 0.0
        target_sum = 0.0
        loss_sum = 0.0
        target_lambda = TARGET_WEIGHT * min(1.0, step / RAMP_STEPS)

        for _ in range(ACCUM_STEPS):
            source_batch, source_iter = next_batch(source_iter, source_loader)
            target_batch, target_iter = next_batch(target_iter, target_loader)
            if set(target_batch) != {"pixel_values", "name"}:
                raise RuntimeError("target loader exposed unexpected keys")

            for item in range(BATCH_SIZE):
                sx = source_batch["pixel_values"][item:item + 1].to(args.device, non_blocking=True)
                sy = source_batch["labels"][item:item + 1].to(args.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=True):
                    source_logits = model(pixel_values=sx).logits
                source_ce = F.cross_entropy(
                    F.interpolate(source_logits.float(), size=sy.shape[-2:], mode="bilinear", align_corners=False),
                    sy, ignore_index=255,
                )

                name = target_batch["name"][item]
                stem = Path(name).stem
                tx = target_batch["pixel_values"][item:item + 1].to(args.device, non_blocking=True)
                view = random.randrange(8)
                augmented = strong_photo(d4_forward(tx, view))
                with torch.amp.autocast("cuda", enabled=True):
                    student_view = model(pixel_values=augmented).logits
                student = d4_inverse(student_view.float(), view)
                theta = torch.from_numpy(np.load(THETA_CACHE / f"{stem}.npy").astype(np.float32)).to(args.device)[None]
                d1 = torch.from_numpy(np.load(D1_CACHE / f"{stem}.npy").astype(np.float32)).to(args.device)[None]
                if tuple(student.shape) != (1, 8, 256, 256) or tuple(theta.shape) != tuple(student.shape) or tuple(d1.shape) != tuple(student.shape):
                    raise RuntimeError(f"native shape mismatch for {stem}: {student.shape}, {theta.shape}, {d1.shape}")
                teacher = 0.5 * (theta + d1)
                if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
                    raise RuntimeError(f"nonfinite target logits for {stem}")
                teacher_prob = ((teacher + bstar) / TEMPERATURE).softmax(1)
                student_log_prob = ((student + bstar) / TEMPERATURE).log_softmax(1)
                target_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(1).mean() * (TEMPERATURE ** 2)
                loss = source_ce + target_lambda * target_kl
                scaled_loss = loss / (ACCUM_STEPS * BATCH_SIZE)
                scaler.scale(scaled_loss).backward()
                source_sum += float(source_ce.detach())
                target_sum += float(target_kl.detach())
                loss_sum += float(loss.detach())
                del sx, sy, source_logits, tx, augmented, student_view, student, theta, d1, teacher, teacher_prob, student_log_prob

        scaler.unscale_(optimizer)
        for parameter in base_params + head_params:
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"nonfinite gradient at step {step}")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= old_scale
        if not stepped:
            amp_skips.append(step)
        apply_modes(model)
        bad = frozen_diff(model, frozen)
        if bad:
            raise RuntimeError(f"frozen drift at step {step}: {bad[:5]}")

        denominator = ACCUM_STEPS * BATCH_SIZE
        last_row = {
            "step": step,
            "microsteps": ACCUM_STEPS,
            "source_ce": source_sum / denominator,
            "target_kl": target_sum / denominator,
            "target_lambda": target_lambda,
            "total_loss": loss_sum / denominator,
            "lr_stage23": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "amp_stepped": int(stepped),
            "scale": scaler.get_scale(),
            "elapsed_sec": round(time.time() - started, 3),
        }
        writer.writerow(last_row)
        log_file.flush()
        if step == 1 or step % 25 == 0 or step == expected_steps:
            print(json.dumps(last_row), flush=True)

    log_file.close()
    if amp_skips:
        raise RuntimeError(f"AMP skipped optimizer updates: {amp_skips}")
    bad = frozen_diff(model, frozen)
    if bad:
        raise RuntimeError(f"frozen drift before checkpoint: {bad[:5]}")

    checkpoint = out / f"checkpoint_step{expected_steps}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": expected_steps,
            "TEKD": manifest,
        },
        checkpoint,
    )
    checkpoint_sha = sha256_file(checkpoint)
    reload_state, _ = load_state(checkpoint)
    reload_model = make_model(cfg_model)
    reload_model.load_state_dict(reload_state, strict=True)
    del reload_model

    visualizations = save_visualizations(model, source, out, args.device)
    result = {
        "status": "PASS",
        "mode": args.mode,
        "optimizer_updates": expected_steps,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "amp_skip_steps": amp_skips,
        "frozen_identical": True,
        "strict_reload": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "last_record": last_row,
        "prediction_visualizations": visualizations,
        "runtime_sec": round(time.time() - started, 3),
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
        "automatic_continuation": False,
    }
    write_json(out / "ENTRY_RESULT.json", result)
    write_json(out / "CHECKPOINT_AUDIT.json", {"checkpoint": str(checkpoint), "sha256": checkpoint_sha, "strict_reload": True, "frozen_identical": True})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
