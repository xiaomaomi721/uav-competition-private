#!/usr/bin/env python3
"""Exact single-model TEKD D4 inference, no-GT behavior audit, source guard, and ZIP.

Loads exactly one checkpoint. Teacher caches are used only for audit, never to
produce the candidate prediction. No platform submission helper exists here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/home/jhy/rare_diag_20260915")
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
TEST2 = Path("/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images")
DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
THETA_CACHE = ROOT / "round2_20260921/cache_278_test2_raw/scores"
D1_CACHE = ROOT / "round2_20260923/D1_2k_probe/raw_cache/scores"
D1_CKPT = ROOT / "round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt"
D1_PRED = ROOT / "round2_20260923/D1_2k_probe/predictions"
TEACHER_ZIP = ROOT / "d1_interp_20260925/D1_alpha050_test2.zip"
SOURCE_SPLIT = ROOT / "round2_multiprobe_20260922/00_inventory/SOURCE_AUDIT_SPLIT.json"
SOURCE_GT = ROOT / "b2_val_20260917/gt"
EXPECTED_GPU_UUID = "GPU-32853ccf-4c4f-1328-8d75-04efe9d5298f"
EXPECTED_GPU_BUS = "00000000:CA:00.0"
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]
TEMPERATURE = 2.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state(path: Path):
    import torch
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("model", "student_model", "model_state_dict", "state_dict"):
            value = obj.get(key)
            if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
                return value
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError(f"no tensor state dict in {path}")


def assert_gpu3_binding() -> dict:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise SystemExit("exact physical GPU3 binding required")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"], text=True
    ).splitlines()
    match = [x.strip() for x in rows if x.strip().startswith("3,")]
    if len(match) != 1 or EXPECTED_GPU_UUID not in match[0] or EXPECTED_GPU_BUS not in match[0]:
        raise SystemExit(f"GPU3 identity mismatch: {match}")
    return {"index": 3, "uuid": EXPECTED_GPU_UUID, "pci_bus": EXPECTED_GPU_BUS, "row": match[0]}


def d4_forward(tensor, index):
    import torch
    rotation, mirror = VIEWS[index]
    if rotation:
        tensor = torch.rot90(tensor, rotation, (-2, -1))
    if mirror:
        tensor = tensor.flip((-1,))
    return tensor


def d4_inverse(tensor, index):
    import torch
    rotation, mirror = VIEWS[index]
    if mirror:
        tensor = tensor.flip((-1,))
    if rotation:
        tensor = torch.rot90(tensor, -rotation, (-2, -1))
    return tensor.contiguous()


def build_model(checkpoint: Path, device: str):
    from src.models import build_model as make_model
    from src.utils import load_config
    cfg = load_config(str(CONFIG))
    cfg["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0, "norm_eps": 1e-12, "start_step": 1}
    cfg.pop("mixstyle", None)
    model = make_model(cfg)
    model.load_state_dict(load_state(checkpoint), strict=True)
    return model.to(device).eval()


def d4_mean(model, pixel_values):
    import torch
    accumulator = None
    with torch.inference_mode():
        for index in range(8):
            with torch.amp.autocast("cuda", enabled=False):
                logits = model(pixel_values=d4_forward(pixel_values, index)).logits.float()
            canonical = d4_inverse(logits, index)[0]
            if tuple(canonical.shape) != (8, 256, 256):
                raise RuntimeError(f"bad native shape: {canonical.shape}")
            accumulator = canonical if accumulator is None else accumulator + canonical
    return accumulator / 8.0


def update_transition(matrix, reference, candidate):
    matrix += np.bincount((reference.astype(np.int64) * 8 + candidate.astype(np.int64)).ravel(), minlength=64).reshape(8, 8)


def transition_rows(matrix):
    rows = []
    for source in range(8):
        for target in range(8):
            if source != target and matrix[source, target]:
                rows.append({"from": CLASS_NAMES[source], "to": CLASS_NAMES[target], "pixels": int(matrix[source, target])})
    return sorted(rows, key=lambda x: x["pixels"], reverse=True)


def iou_from_cm(cm):
    union = cm.sum(1) + cm.sum(0) - np.diag(cm)
    values = np.divide(np.diag(cm), union, out=np.zeros(8, dtype=np.float64), where=union > 0)
    return values, float(values.mean())


def eval_h300(checkpoint: Path, device: str):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.transforms import normalize_to_tensor
    names = json.loads(SOURCE_SPLIT.read_text())["H300"]
    model = build_model(checkpoint, device)
    cm = np.zeros((8, 8), np.int64)
    bias = torch.from_numpy(BSTAR).to(device)[:, None, None]
    start = time.time()
    for index, stem in enumerate(names, 1):
        with Image.open(DATA / f"train/images/{stem}.png") as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        gt = np.load(SOURCE_GT / f"{stem}.npy", allow_pickle=False)
        px = normalize_to_tensor(rgb).unsqueeze(0).to(device)
        raw = d4_mean(model, px)
        deployed = F.interpolate((raw + bias)[None], size=gt.shape, mode="bilinear", align_corners=False)[0]
        pred = deployed.argmax(0).byte().cpu().numpy()
        valid = gt != 255
        cm += np.bincount((gt[valid].astype(np.int64) * 8 + pred[valid]).ravel(), minlength=64).reshape(8, 8)
        if index % 50 == 0:
            print(f"H300 {checkpoint.name} {index}/300", flush=True)
    per_class, miou = iou_from_cm(cm)
    del model
    torch.cuda.empty_cache()
    return {"checkpoint": str(checkpoint), "miou": miou, "per_class_iou": per_class.tolist(), "confusion": cm.tolist(), "runtime_sec": round(time.time() - start, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    gpu = assert_gpu3_binding()
    checkpoint = Path(args.checkpoint)
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"output exists: {out}")
    out.mkdir(parents=True)
    scores = out / "raw_cache/scores"
    predictions = out / "predictions"
    scores.mkdir(parents=True)
    predictions.mkdir()

    for path in (str(SNAP), str(ROOT / "round2_multiprobe_20260922"), "/home/jhy/gpu2_day_queue"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.transforms import normalize_to_tensor

    model = build_model(checkpoint, args.device)
    bias = torch.from_numpy(BSTAR).to(args.device)[:, None, None]
    candidate_hist = np.zeros(8, np.int64)
    d1_hist = np.zeros(8, np.int64)
    teacher_hist = np.zeros(8, np.int64)
    matrix_d1 = np.zeros((8, 8), np.int64)
    matrix_teacher = np.zeros((8, 8), np.int64)
    per_image_d1 = []
    per_image_teacher = []
    kl_candidate_sum = 0.0
    kl_d1_sum = 0.0
    confidence_sum = 0.0
    entropy_sum = 0.0
    changed_teacher_conf_sum = 0.0
    changed_teacher_pixels = 0
    quant_diff = 0
    quant_pixels = 0
    start = time.time()
    names = [f"test2_{i}.png" for i in range(1, 1301)]
    sanity_indices = {1, 2, 3, 100, 500, 900, 1299, 1300}
    sanity_inputs = {}

    with zipfile.ZipFile(TEACHER_ZIP) as teacher_zip:
        for index, name in enumerate(names, 1):
            with Image.open(TEST2 / name) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            px = normalize_to_tensor(rgb).unsqueeze(0).to(args.device)
            raw = d4_mean(model, px)
            if not torch.isfinite(raw).all():
                raise RuntimeError(f"nonfinite TEKD raw logits: {name}")
            cached = raw.half().cpu().numpy()
            np.save(scores / f"{Path(name).stem}.npy", cached)
            quant_diff += int((raw.argmax(0) != raw.half().float().argmax(0)).sum())
            quant_pixels += raw.shape[-2] * raw.shape[-1]
            deployed = F.interpolate((raw + bias)[None], size=(1024, 1024), mode="bilinear", align_corners=False)[0]
            probability = deployed.softmax(0)
            confidence, pred_t = probability.max(0)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(0)
            candidate = pred_t.byte().cpu().numpy()
            Image.fromarray(candidate + 1, mode="L").save(predictions / name)

            d1 = np.asarray(Image.open(D1_PRED / name), dtype=np.uint8) - 1
            with teacher_zip.open(name) as file:
                teacher = np.asarray(Image.open(file), dtype=np.uint8) - 1
            update_transition(matrix_d1, d1, candidate)
            update_transition(matrix_teacher, teacher, candidate)
            per_image_d1.append(float(np.mean(d1 != candidate)))
            per_image_teacher.append(float(np.mean(teacher != candidate)))
            candidate_hist += np.bincount(candidate.ravel(), minlength=8)
            d1_hist += np.bincount(d1.ravel(), minlength=8)
            teacher_hist += np.bincount(teacher.ravel(), minlength=8)

            theta_native = torch.from_numpy(np.load(THETA_CACHE / f"{Path(name).stem}.npy").astype(np.float32)).to(args.device)
            d1_native = torch.from_numpy(np.load(D1_CACHE / f"{Path(name).stem}.npy").astype(np.float32)).to(args.device)
            teacher_native = 0.5 * (theta_native + d1_native)
            teacher_prob_native = ((teacher_native + bias) / TEMPERATURE).softmax(0)
            candidate_log_native = ((raw + bias) / TEMPERATURE).log_softmax(0)
            d1_log_native = ((d1_native + bias) / TEMPERATURE).log_softmax(0)
            kl_candidate = F.kl_div(candidate_log_native, teacher_prob_native, reduction="none").sum(0).mean() * TEMPERATURE ** 2
            kl_d1 = F.kl_div(d1_log_native, teacher_prob_native, reduction="none").sum(0).mean() * TEMPERATURE ** 2
            kl_candidate_sum += float(kl_candidate)
            kl_d1_sum += float(kl_d1)
            confidence_sum += float(confidence.mean())
            entropy_sum += float(entropy.mean())

            teacher_deployed = F.interpolate((teacher_native + bias)[None], size=(1024, 1024), mode="bilinear", align_corners=False)[0]
            teacher_conf = teacher_deployed.softmax(0).max(0).values
            changed = torch.from_numpy(teacher != candidate).to(args.device)
            if changed.any():
                changed_teacher_conf_sum += float(teacher_conf[changed].sum())
                changed_teacher_pixels += int(changed.sum())
            if index in sanity_indices:
                sanity_inputs[index] = rgb
            if index % 100 == 0:
                print(f"test2 {index}/1300 {time.time()-start:.0f}s", flush=True)

    # Fixed-eight live versus fp16 cache.
    live_sanity = []
    for index in sorted(sanity_indices):
        px = normalize_to_tensor(sanity_inputs[index]).unsqueeze(0).to(args.device)
        live = d4_mean(model, px)
        cached = torch.from_numpy(np.load(scores / f"test2_{index}.npy").astype(np.float32)).to(args.device)
        live_sanity.append({
            "image": f"test2_{index}.png",
            "max_abs": float((live - cached).abs().max()),
            "argmax_disagree_frac": float((live.argmax(0) != cached.argmax(0)).float().mean()),
        })

    package = out / "TEKD_D1_alpha050_T2_lam0p5_s1000_D4_Bstar_test2.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(predictions / name, name)
    with zipfile.ZipFile(package) as archive:
        if archive.namelist() != names or archive.testzip() is not None:
            raise RuntimeError("ZIP name/order/integrity failure")
        for name in names:
            with archive.open(name) as file:
                image = Image.open(file)
                image.load()
                values = np.asarray(image)
            if image.mode != "L" or image.size != (1024, 1024) or values.dtype != np.uint8 or values.min() < 1 or values.max() > 8:
                raise RuntimeError(f"bad packaged PNG {name}: {image.mode} {image.size} {values.dtype} {values.min()}..{values.max()}")

    per_d1 = np.asarray(per_image_d1)
    per_teacher = np.asarray(per_image_teacher)
    total = int(candidate_hist.sum())
    flip_d1 = int(matrix_d1.sum() - np.trace(matrix_d1))
    flip_teacher = int(matrix_teacher.sum() - np.trace(matrix_teacher))
    kl_candidate_mean = kl_candidate_sum / 1300
    kl_d1_mean = kl_d1_sum / 1300
    audit = {
        "single_model": True,
        "architecture": "SegFormer-B3",
        "checkpoint_count": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "tta": "D4_TTA8_native_single_scale",
        "multi_scale": False,
        "ensemble": False,
        "teacher_used_for_candidate_inference": False,
        "teacher_definition_for_training_and_audit": "0.5*old_theta278 + 0.5*D1 native raw D4 logits",
        "gpu": gpu,
        "count": 1300,
        "bstar": BSTAR.tolist(),
        "deployment_order": "raw D4 views -> inverse -> FP32 mean@256 -> Bstar once -> bilinear logits@1024 -> argmax -> +1",
        "candidate_teacher_kl_T2": kl_candidate_mean,
        "d1_teacher_kl_T2": kl_d1_mean,
        "teacher_kl_relative_reduction": (kl_d1_mean - kl_candidate_mean) / max(kl_d1_mean, 1e-12),
        "candidate_vs_d1": {
            "flip_rate": flip_d1 / total,
            "per_image": {"mean": float(per_d1.mean()), "median": float(np.median(per_d1)), "p90": float(np.quantile(per_d1, 0.9)), "p95": float(np.quantile(per_d1, 0.95)), "max": float(per_d1.max())},
            "transition_matrix": matrix_d1.tolist(),
            "top_transitions": transition_rows(matrix_d1)[:20],
        },
        "candidate_vs_teacher": {
            "flip_rate": flip_teacher / total,
            "per_image": {"mean": float(per_teacher.mean()), "median": float(np.median(per_teacher)), "p90": float(np.quantile(per_teacher, 0.9)), "p95": float(np.quantile(per_teacher, 0.95)), "max": float(per_teacher.max())},
            "transition_matrix": matrix_teacher.tolist(),
            "top_transitions": transition_rows(matrix_teacher)[:20],
            "changed_pixel_teacher_confidence_mean": changed_teacher_conf_sum / max(changed_teacher_pixels, 1),
        },
        "class_share_pct": {
            "candidate": (candidate_hist / total * 100).tolist(),
            "d1": (d1_hist / total * 100).tolist(),
            "teacher": (teacher_hist / total * 100).tolist(),
            "candidate_minus_teacher_pp": ((candidate_hist - teacher_hist) / total * 100).tolist(),
        },
        "candidate_mean_max_confidence": confidence_sum / 1300,
        "candidate_mean_entropy": entropy_sum / 1300,
        "fp16_quantization": {"pixels": quant_pixels, "argmax_disagree_pixels": quant_diff, "argmax_disagree_frac": quant_diff / quant_pixels},
        "live_vs_cache_sanity": live_sanity,
        "package": str(package),
        "package_sha256": sha256_file(package),
        "package_count": 1300,
        "package_format_pass": True,
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
        "runtime_sec_before_source_guard": round(time.time() - start, 1),
    }
    write_json(out / "TEST2_AUDIT.json", audit)
    print(json.dumps({k: audit[k] for k in ("candidate_teacher_kl_T2", "d1_teacher_kl_T2", "teacher_kl_relative_reduction", "candidate_vs_d1", "candidate_vs_teacher", "class_share_pct", "package_sha256")}, indent=2), flush=True)

    # Exact-D4 H300 collapse guard, using identical code for D1 and candidate.
    del model
    torch.cuda.empty_cache()
    candidate_h300 = eval_h300(checkpoint, args.device)
    d1_h300 = eval_h300(D1_CKPT, args.device)
    h300 = {
        "protocol": "exact D4 raw FP32 mean -> Bstar -> bilinear logits -> argmax; source collapse guard only",
        "candidate": candidate_h300,
        "d1": d1_h300,
        "delta_pp": 100 * (candidate_h300["miou"] - d1_h300["miou"]),
    }
    write_json(out / "H300_D4_GUARD.json", h300)

    hard_fail = []
    if audit["candidate_teacher_kl_T2"] >= audit["d1_teacher_kl_T2"]:
        hard_fail.append("teacher KL did not improve over D1")
    if audit["candidate_vs_teacher"]["flip_rate"] >= audit["candidate_vs_d1"]["flip_rate"]:
        hard_fail.append("candidate is not closer to teacher than to D1")
    if max(abs(x) for x in audit["class_share_pct"]["candidate_minus_teacher_pp"]) > 0.5:
        hard_fail.append("class-share delta versus teacher exceeds 0.5 pp")
    if audit["candidate_vs_d1"]["flip_rate"] > 0.03:
        hard_fail.append("candidate versus D1 flip exceeds 3%")
    if audit["candidate_vs_d1"]["per_image"]["max"] > 0.5:
        hard_fail.append("one image exceeds 50% change versus D1")
    if h300["delta_pp"] < -0.75:
        hard_fail.append("H300 D4 decline exceeds 0.75 pp")
    final = {
        "status": "PASS" if not hard_fail else "FAIL",
        "hard_failures": hard_fail,
        "checkpoint_sha256": audit["checkpoint_sha256"],
        "package_sha256": audit["package_sha256"],
        "package": str(package),
        "test2_audit": str(out / "TEST2_AUDIT.json"),
        "h300_guard": str(out / "H300_D4_GUARD.json"),
        "single_model": True,
        "platform_submission_performed": False,
        "test2_ground_truth_used": False,
        "total_runtime_sec": round(time.time() - start, 1),
    }
    write_json(out / "FINAL_RESULT.json", final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
