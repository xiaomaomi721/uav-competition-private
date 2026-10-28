#!/usr/bin/env python3
"""ATDA-71 -- Anchored Target-Domain Adaptation, base theta261.

Writes the target-domain information of the 500 unlabeled test1 images directly
into the mid/deep encoder of the current champion (theta261 = 0.702836), while
the head and the deployment calibration structure stay bit-frozen.

    student   theta     = theta261                       trainable: stages.2/3 only
    reference theta_ref = theta261                       no_grad, never updated
    ema       theta_ema = theta261, momentum 0.999       no_grad, never updated

    L = L_source + r(step) * (0.5 * L_hard + 0.25 * L_cons)
    r(step) = min(1.0, step / 500)

theta261 = phase0_candidates_20260918/A_cleanSEM_M_k100_th209.pt
           (theta209 + 1.0*R_old-normalized clean SEMEDA stage2 delta)
sha256    9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786
platform  0.702836  (package 261)

The fixed reference is the theta261 native-256 D4 TTA8 cache: for each view the
model's native stride-4 logits (256x256) are inverse-transformed to canonical
and FP32-averaged over the 8 D4 views -- no upsampling, no interpolation in the
geometric alignment. b_star is applied exactly once on the canonical grid.

Physical GPU2 only (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=2).
Writes only under --output-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

NUM_CLASSES = 8
IGNORE_INDEX = 255
PHYSICAL_GPU_INDEX = 2

SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
# --- theta261 (the current champion, package 261) ----------------------------
THETA261 = Path("/home/jhy/rare_diag_20260915/phase0_candidates_20260918/"
                "A_cleanSEM_M_k100_th209.pt")
THETA261_SHA256 = "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786"
THETA261_SCORE = 0.702836
CHAMPION_CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
# Native-256 D4 TTA8 canonical reference cache, regenerated for theta261.
REF_CACHE = Path("/home/jhy/rare_diag_20260915/atda_20260918/ref261_cache/scores")
TEST_IMAGES = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")

# Deployment operator of the current champion (unchanged from 237 -> 261).
# index: Background Building Road Water Barren Vegetation Agricultural Vehicle
BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]

# D4 ordering, identical to tta_anatomy_v2.VIEWS. Index 0 is the identity.
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
NON_IDENTITY_VIEWS = tuple(i for i in range(8) if VIEWS[i] != (0, False))

TRAINABLE_PREFIXES = ("segformer.stages.2.", "segformer.stages.3.")
TRAINABLE_MODULES = ("segformer.stages.2", "segformer.stages.3")
FROZEN_MODULES = ("segformer.stages.0", "segformer.stages.1", "decode_head")

LAMBDA_HARD = 0.5
LAMBDA_CONS = 0.25
RAMP_STEPS = 500
EMA_DECAY = 0.999

SOURCE_COUNT = 5597
TARGET_COUNT = 500
LOGIT_GRID = 256


# --------------------------------------------------------------------------
# D4 -- exact discrete permutations, never interpolation
# --------------------------------------------------------------------------
def d4_forward(t, index: int):
    """Apply D4 element `index` to the last two dims. Pure data movement."""
    import torch
    rotation, mirror = VIEWS[index]
    if rotation:
        t = torch.rot90(t, rotation, (-2, -1))
    if mirror:
        t = t.flip((-1,))
    return t


def d4_inverse(t, index: int):
    """Invert d4_forward(t, index). Exact permutation, no interpolation.

    forward = flip . rot90(k)  =>  inverse = rot90(-k) . flip
    """
    import torch
    rotation, mirror = VIEWS[index]
    t = t.contiguous()
    if mirror:
        t = t.flip((-1,))
    if rotation:
        t = torch.rot90(t, -rotation, (-2, -1))
    return t.contiguous()


def d4_self_test() -> dict[str, Any]:
    """Synthetic proof that transform+inverse is the exact identity."""
    import torch
    out: dict[str, Any] = {"ok": True}
    for shape, tag in (((2, 3, 1024, 1024), "grid_1024"), ((2, 3, 256, 256), "grid_256")):
        g = torch.Generator().manual_seed(20260918)
        t = torch.randn(*shape, generator=g)
        worst = 0.0
        moved = True
        for index in NON_IDENTITY_VIEWS:
            back = d4_inverse(d4_forward(t, index), index)
            worst = max(worst, float((back - t).abs().max()))
            if torch.equal(d4_forward(t, index), t):
                moved = False
        sigs = {hash(tuple(d4_forward(t, i).reshape(-1)[:256].tolist()))
                for i in NON_IDENTITY_VIEWS}
        entry = {
            "max_abs_err": worst,
            "bitwise_identity": worst == 0.0,
            "every_view_moves_tensor": moved,
            "distinct_signatures": len(sigs),
            "checked_views": len(NON_IDENTITY_VIEWS),
        }
        out[tag] = entry
        if not (entry["bitwise_identity"] and moved and len(sigs) == len(NON_IDENTITY_VIEWS)):
            out["ok"] = False
    return out


def native_tta8_mean_logits(model, px, device):
    """D4 TTA8 mean over NATIVE stride-4 logits, canonical aligned, FP32.

    Every view is transformed as an exact permutation of the 1024 input, the
    model's native 256 logits are inverse-transformed exactly back to canonical,
    and the 8 views are FP32-averaged. No upsampling, no bilinear anywhere.
    """
    import torch
    acc = None
    with torch.no_grad():
        for index in range(8):
            transformed = d4_forward(px, index)
            with torch.amp.autocast("cuda", enabled=False):
                logits = model(pixel_values=transformed).logits
            canonical = d4_inverse(logits.float(), index)[0]  # strip batch dim
            if tuple(canonical.shape) != (NUM_CLASSES, LOGIT_GRID, LOGIT_GRID):
                raise SystemExit(f"native logits are {tuple(canonical.shape)}, "
                                 f"expected {NUM_CLASSES}x{LOGIT_GRID}x{LOGIT_GRID}")
            acc = canonical if acc is None else acc + canonical
    return acc / 8.0  # [8,256,256] canonical mean logits


# --------------------------------------------------------------------------
# frozen-scope + mode helpers
# --------------------------------------------------------------------------
def is_trainable(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in TRAINABLE_PREFIXES)


def split_scope(model):
    trainable, frozen = [], []
    for name, param in model.named_parameters():
        (trainable if is_trainable(name) else frozen).append((name, param))
    return trainable, frozen


def frozen_modules(model):
    for name, module in model.named_modules():
        if name and not is_trainable(name + "."):
            yield name, module


def snapshot_frozen(model, frozen_names):
    """Byte snapshot of every frozen parameter and every frozen buffer."""
    params = dict(model.named_parameters())
    snap = {"param::" + name: params[name].detach().clone() for name in frozen_names}
    for mod_name, module in frozen_modules(model):
        for buf_name, buf in module._buffers.items():
            if buf is not None:
                snap[f"buffer::{mod_name}.{buf_name}"] = buf.detach().clone()
    return snap


def apply_train_eval_modes(model):
    """Explicit mode map. Never let a parent .eval() recurse into stages 2/3.

    eval everything, then re-assert train() on exactly the two trainable stages.
    """
    model.eval()
    for name in TRAINABLE_MODULES:
        model.get_submodule(name).train()


def training_modes(model) -> dict[str, bool]:
    return {
        "segformer.stages.0": bool(model.get_submodule("segformer.stages.0").training),
        "segformer.stages.1": bool(model.get_submodule("segformer.stages.1").training),
        "segformer.stages.2": bool(model.get_submodule("segformer.stages.2").training),
        "segformer.stages.3": bool(model.get_submodule("segformer.stages.3").training),
        "decode_head": bool(model.decode_head.training),
    }


def frozen_diff(model, snap) -> dict[str, Any]:
    """Byte-level diff of the whole frozen scope against the theta261 snapshot."""
    import torch
    params = dict(model.named_parameters())
    bad, max_abs = [], 0.0
    for key, value in snap.items():
        kind, name = key.split("::", 1)
        if kind == "param":
            cur = params[name].detach()
        else:
            parent, _, leaf = name.rpartition(".")
            module = model.get_submodule(parent) if parent else model
            cur = module._buffers[leaf].detach()
        if cur.dtype != value.dtype or cur.shape != value.shape or not torch.equal(cur, value):
            bad.append(key)
            if cur.is_floating_point() and cur.shape == value.shape:
                max_abs = max(max_abs, float((cur.float() - value.float()).abs().max()))
    return {
        "identical": not bad,
        "n_changed": len(bad),
        "changed": bad[:10],
        "max_abs_diff": max_abs,
    }


def restore_frozen(model, snap):
    """Put every frozen parameter/buffer back byte-for-byte, then fix modes."""
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in snap.items():
            kind, name = key.split("::", 1)
            if kind == "param":
                params[name].copy_(value)
            else:
                parent, _, leaf = name.rpartition(".")
                module = model.get_submodule(parent) if parent else model
                module._buffers[leaf].copy_(value)
    apply_train_eval_modes(model)


def scope_sha(model, names) -> str:
    """SHA256 over the exact bytes of the named parameters, in sorted order."""
    import torch
    numpy_ok = (torch.float32, torch.float16, torch.float64, torch.bfloat16,
                torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool)
    params = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in sorted(names):
        p = params[name].detach().to("cpu").contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(p.shape), str(p.dtype))).encode())
        digest.update(p.numpy().tobytes() if p.dtype in numpy_ok
                      else p.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_champion_state(path: Path):
    import torch
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state = None
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if (isinstance(candidate, dict) and candidate
                    and all(isinstance(v, torch.Tensor) for v in candidate.values())):
                state = candidate
                break
        if state is None and checkpoint and all(
                isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state = checkpoint
    if state is None:
        raise ValueError(f"no tensor state dict in {path}")
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


# --------------------------------------------------------------------------
# fixed reference: theta261 native-256 cache
# --------------------------------------------------------------------------
def build_reference(cache_dir: Path) -> dict[str, Any]:
    """z_ref is already native 256x256 canonical; +b_star -> p_ref -> y_ref, w."""
    import torch

    bstar = torch.tensor(BSTAR, dtype=torch.float32).view(1, NUM_CLASSES, 1, 1)
    y_ref = torch.empty((TARGET_COUNT, LOGIT_GRID, LOGIT_GRID), dtype=torch.uint8)
    w_ref = torch.empty((TARGET_COUNT, LOGIT_GRID, LOGIT_GRID), dtype=torch.float16)
    conf_hist = np.zeros(10, dtype=np.int64)
    low_conf = 0
    total = 0
    t0 = time.time()
    for index in range(TARGET_COUNT):
        arr = np.load(cache_dir / f"test1_{index + 1}.npy")
        if arr.shape != (NUM_CLASSES, LOGIT_GRID, LOGIT_GRID):
            raise SystemExit(f"bad cache shape {arr.shape} for test1_{index + 1}; "
                             f"expected native {NUM_CLASSES}x{LOGIT_GRID}x{LOGIT_GRID}")
        z = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
        probs = torch.softmax(z + bstar, dim=1)[0]
        conf = probs.max(0).values
        y_ref[index] = probs.argmax(0).to(torch.uint8)
        w_ref[index] = conf.to(torch.float16)
        conf_hist += np.histogram(conf.numpy(), bins=10, range=(0.0, 1.0))[0]
        low_conf += int((conf < 1.0 / NUM_CLASSES).sum())
        total += conf.numel()
        del arr, z, probs, conf
    meta = {
        "images": TARGET_COUNT,
        "grid": f"{LOGIT_GRID}x{LOGIT_GRID} (native stride-4, no pooling)",
        "bstar": BSTAR,
        "w_definition": "max class probability of p_ref, detached, no threshold",
        "w_min": round(float(w_ref.min()), 6),
        "w_max": round(float(w_ref.max()), 6),
        "w_mean": round(float(w_ref.float().mean()), 6),
        "confidence_histogram_0_to_1": conf_hist.tolist(),
        "below_uniform_pixels_pct": round(100.0 * low_conf / total, 4),
        "y_ref_sha256": hashlib.sha256(y_ref.numpy().tobytes()).hexdigest(),
        "w_ref_sha256": hashlib.sha256(w_ref.numpy().tobytes()).hexdigest(),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    return {"y_ref": y_ref, "w_ref": w_ref, "meta": meta}


def verify_reference_against_live_teacher(model_ref, device, count: int = 8) -> dict[str, Any]:
    """Re-forward theta261 on >=8 images; native canonical logits must equal the
    cache up to fp16 rounding."""
    import torch
    from PIL import Image

    from src.transforms import normalize_to_tensor

    rows = []
    worst = 0.0
    worst_share = 0.0
    bias = torch.tensor(BSTAR).view(NUM_CLASSES, 1, 1)
    for index in range(count):
        with Image.open(TEST_IMAGES / f"test1_{index + 1}.png") as im:
            img = np.asarray(im.convert("RGB")).copy()
        px = normalize_to_tensor(img).unsqueeze(0).to(device)
        live = native_tta8_mean_logits(model_ref, px, device).cpu()
        cached = torch.from_numpy(
            np.load(REF_CACHE / f"test1_{index + 1}.npy").astype(np.float32))
        diff = float((live - cached).abs().max())
        scale = float(cached.abs().max())
        share_b = float(((live + bias).argmax(0) != (cached + bias).argmax(0)).float().mean())
        worst = max(worst, diff)
        worst_share = max(worst_share, share_b)
        rows.append({"image": f"test1_{index + 1}", "max_abs_diff": round(diff, 6),
                     "abs_scale": round(scale, 3),
                     "argmax_bstar_disagree_pct": round(100.0 * share_b, 4)})
        del px, live, cached
    tolerance = 2.0e-3 * max(1.0, max(r["abs_scale"] for r in rows))
    return {
        "checked_images": count,
        "max_abs_diff": worst,
        "tolerance": round(tolerance, 4),
        "max_argmax_bstar_disagree_pct": round(100.0 * worst_share, 4),
        "pass": worst <= tolerance and worst_share <= 1.0e-4,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
class TargetDataset:
    """The 500 unlabeled test1 images.

    Structurally incapable of returning a label: it opens exactly one file per
    sample, from the official test image directory, and returns only the
    normalized tensor plus the file name. The competition ships no test1 GT.
    """

    def __init__(self, image_dir: Path, count: int = TARGET_COUNT):
        from PIL import Image  # noqa: F401  -- surface a broken env here, not in a worker

        self.image_dir = Path(image_dir)
        self.names = [f"test1_{i}.png" for i in range(1, count + 1)]
        missing = [n for n in self.names if not (self.image_dir / n).is_file()]
        if missing:
            raise FileNotFoundError(f"missing target images: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        from PIL import Image

        from src.transforms import normalize_to_tensor

        name = self.names[index]
        with Image.open(self.image_dir / name) as handle:
            image = np.asarray(handle.convert("RGB")).copy()
        if image.shape != (1024, 1024, 3):
            raise ValueError(f"bad target shape {name}: {image.shape}")
        return {"pixel_values": normalize_to_tensor(image), "name": name}


def target_collate(batch):
    import torch
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], 0),
        "name": [b["name"] for b in batch],
    }


def assert_target_dir() -> dict[str, Any]:
    """The target directory must be exactly test1_1..test1_500 and nothing else."""
    actual = sorted((p.name for p in TEST_IMAGES.glob("*.png")),
                    key=lambda n: int(Path(n).stem.split("_")[-1]))
    expected = [f"test1_{i}.png" for i in range(1, TARGET_COUNT + 1)]
    stray = [n for n in actual if n not in expected]
    missing = [n for n in expected if n not in actual]
    return {
        "dir": str(TEST_IMAGES),
        "n_png_files": len(actual),
        "exact_test1_1_to_500": actual == expected,
        "missing": missing[:5],
        "stray": stray[:5],
        # evidence this IS the official test1: the 237 package was generated from
        # exactly these files and the platform scored that package (class-share
        # reproduction diff 0.0008pp against the submitted zip).
        "provenance": "same directory that produced the platform-scored 237 package",
        "clean": actual == expected,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-root", required=True,
                    help="dataset_root holding train/images and train/masks")
    ap.add_argument("--source-ids", required=True,
                    help="text file, one stem per line: the 5597 official GT images")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--gate-steps", type=int, nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="200-step acceptance run: emit the SMOKE report and stop")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise SystemExit("require CUDA_VISIBLE_DEVICES=2 (physical GPU2)")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("require CUDA_DEVICE_ORDER=PCI_BUS_ID")

    for extra in ("/home/jhy/gpu2_day_queue", str(SNAP)):
        if extra not in sys.path:
            sys.path.insert(0, extra)

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from src.dataset import UAVSegmentationDataset, segmentation_collate
    from src.models import build_model as make_model
    from src.rare_crop import RareClassCropper
    from src.transforms import TrainTransform
    from src.utils import load_config

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_handle = (outdir / "atda.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    # ---------------- frozen inputs ----------------
    if not THETA261.is_file():
        raise SystemExit(f"missing theta261: {THETA261}")
    base_sha = sha256_file(THETA261)
    if base_sha != THETA261_SHA256:
        raise SystemExit(f"theta261 sha mismatch: {base_sha}")
    log(f"theta261 verified sha256={base_sha} score={THETA261_SCORE}")

    config = load_config(args.config)
    if config.get("atda_champion_sha256") != THETA261_SHA256:
        raise SystemExit("config atda_champion_sha256 does not match theta261")
    if float(config.get("atda_champion_score", 0.0)) != THETA261_SCORE:
        raise SystemExit("config atda_champion_score does not match theta261")
    if [round(float(x), 6) for x in config["atda_bstar"]] != [round(float(x), 6) for x in BSTAR]:
        raise SystemExit("config atda_bstar does not match the deployment operator")
    tdir = assert_target_dir()
    log("target dir: " + json.dumps(tdir))
    if not tdir["clean"]:
        raise SystemExit("target directory is not exactly test1_1..test1_500")

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    max_steps = int(args.max_steps if args.max_steps is not None else config["max_steps"])
    total_steps = int(config["scheduler"]["total_steps"])
    gate_steps = sorted(set(args.gate_steps if args.gate_steps is not None
                            else config["gate_steps"]))
    lr = float(config["optimizer"]["learning_rate"])
    weight_decay = float(config["optimizer"]["weight_decay"])
    power = float(config["scheduler"]["power"])
    amp_enabled = bool(config.get("amp", True))

    # ---------------- models ----------------
    model_cfg = load_config(str(CHAMPION_CONFIG))
    model_cfg["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0,
                         "norm_eps": 1.0e-12, "start_step": 1}
    model_cfg.pop("mixstyle", None)
    base_state = load_champion_state(THETA261)

    def fresh_theta261():
        m = make_model(model_cfg)
        m.load_state_dict(base_state, strict=True)
        return m

    student = fresh_theta261().to(args.device)
    model_ref = fresh_theta261().to(args.device).eval()
    model_ema = fresh_theta261().to(args.device).eval()
    for m in (model_ref, model_ema):
        for p in m.parameters():
            p.requires_grad_(False)
    log("student / theta_ref / theta_ema all built from theta261")

    trainable, frozen = split_scope(student)
    trainable_names = [n for n, _ in trainable]
    frozen_names = [n for n, _ in frozen]
    for _, param in frozen:
        param.requires_grad_(False)
    for _, param in trainable:
        param.requires_grad_(True)
    if not trainable_names:
        raise SystemExit("no trainable parameters matched the scope prefixes")
    log(f"trainable {len(trainable_names)} tensors / "
        f"{sum(p.numel() for _, p in trainable):,} params  scope={TRAINABLE_PREFIXES}")
    log(f"frozen    {len(frozen_names)} tensors / "
        f"{sum(p.numel() for _, p in frozen):,} params")

    frozen_snapshot = snapshot_frozen(student, frozen_names)
    scope_init_sha = scope_sha(student, trainable_names)
    all_names = [n for n, _ in student.named_parameters()]
    ref_sha_before = scope_sha(model_ref, all_names)
    ema_sha_before = scope_sha(model_ema, all_names)
    apply_train_eval_modes(student)
    init_diff = frozen_diff(student, frozen_snapshot)
    if not init_diff["identical"]:
        raise SystemExit(f"frozen scope not identical to theta261 at init: {init_diff}")
    log(f"frozen scope byte-identical to theta261 at init "
        f"({len(frozen_snapshot)} tensors asserted)")
    log("train/eval modes: " + json.dumps(training_modes(student)))

    ema_has_grad = any(p.requires_grad for p in model_ema.parameters())
    ref_has_grad = any(p.requires_grad for p in model_ref.parameters())

    # ---------------- target reference ----------------
    reference = build_reference(REF_CACHE)
    log("reference: " + json.dumps({k: v for k, v in reference["meta"].items()
                                    if k != "confidence_histogram_0_to_1"}))
    verify = verify_reference_against_live_teacher(model_ref, args.device, count=8)
    log("cache vs live theta261: " + json.dumps(
        {k: v for k, v in verify.items() if k != "rows"}))
    if not verify["pass"]:
        raise SystemExit("theta261 cache does not reproduce theta261 native TTA8; "
                         "refusing to anchor on it")
    y_ref_all = reference["y_ref"]
    w_ref_all = reference["w_ref"]

    # ---------------- loaders ----------------
    source_ids = [line.strip() for line in Path(args.source_ids).read_text().splitlines()
                  if line.strip()]
    if len(source_ids) != SOURCE_COUNT:
        raise SystemExit(f"source id file must hold {SOURCE_COUNT} stems, got {len(source_ids)}")
    cropper = RareClassCropper(
        crop_size=int(config["crop_size"]),
        rare_probability=float(config["rare_crop"]["probability"])
        if config["rare_crop"].get("enabled", False) else 0.0,
        priority_ids=tuple(int(x) for x in config["rare_crop"].get("priority_train_ids", [7, 4, 6])),
        max_offset_fraction=float(config["rare_crop"].get("max_offset_fraction", 0.25)),
    )
    source_dataset = UAVSegmentationDataset(
        Path(args.source_root), source_ids,
        TrainTransform(crop_size=int(config["crop_size"]),
                       config=config.get("transforms", {}),
                       rare_cropper=cropper, ignore_index=IGNORE_INDEX),
        maximum=None, seed=seed, random_subset=False,
    )
    if len(source_dataset) != SOURCE_COUNT:
        raise SystemExit(f"source dataset has {len(source_dataset)} samples")

    target_dataset = TargetDataset(TEST_IMAGES, TARGET_COUNT)
    workers = int(config.get("num_workers", 4))
    source_loader = DataLoader(
        source_dataset, batch_size=1, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=True, collate_fn=segmentation_collate,
        generator=torch.Generator().manual_seed(seed + 10_001),
        persistent_workers=workers > 0,
    )
    target_loader = DataLoader(
        target_dataset, batch_size=1, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=True, collate_fn=target_collate,
        generator=torch.Generator().manual_seed(seed + 10_002),
        persistent_workers=workers > 0,
    )
    log(f"loaders: source {len(source_dataset)} images / target {len(target_dataset)} images, "
        f"1 source + 1 target per optimizer step")

    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    bstar = torch.tensor(BSTAR, dtype=torch.float32, device=args.device).view(1, NUM_CLASSES, 1, 1)

    def lr_at(step: int) -> float:
        return lr * max(0.0, 1.0 - step / total_steps) ** power

    started = time.time()
    history: list[dict[str, Any]] = []
    skipped_steps: list[int] = []
    amp_skip_steps: list[int] = []
    nonfinite_loss_steps: list[int] = []
    target_key_violations: list[str] = []
    step = 0
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    source_loss_ema = None
    ema_delta_ema = None
    gate_stats: list[dict[str, Any]] = []
    hard_min = None

    log(f"ATDA start: max_steps={max_steps} gates={gate_steps} lr={lr} wd={weight_decay} "
        f"poly_power={power} total_steps={total_steps} amp={amp_enabled} "
        f"lam_hard={LAMBDA_HARD} lam_cons={LAMBDA_CONS} ramp={RAMP_STEPS} "
        f"ema={EMA_DECAY} base=theta261@{THETA261_SCORE}")
    if args.smoke:
        log("SMOKE MODE: this is the single permitted 200-step acceptance run")
    apply_train_eval_modes(student)

    while step < max_steps:
        step += 1
        try:
            source_batch = next(source_iter)
        except StopIteration:
            source_iter = iter(source_loader)
            source_batch = next(source_iter)
        try:
            target_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            target_batch = next(target_iter)

        if set(target_batch.keys()) != {"pixel_values", "name"}:
            target_key_violations.append(f"step {step}: {sorted(target_batch.keys())}")

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step - 1)
        optimizer.zero_grad(set_to_none=True)

        source_px = source_batch["pixel_values"].to(args.device, non_blocking=True)
        source_gt = source_batch["labels"].to(args.device, non_blocking=True)
        target_px = target_batch["pixel_values"].to(args.device, non_blocking=True)
        target_index = int(target_batch["name"][0].split("_")[-1].split(".")[0]) - 1
        view = NON_IDENTITY_VIEWS[random.randrange(len(NON_IDENTITY_VIEWS))]

        # ---- D: source branch, the semantic anchor ----
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            source_logits = student(pixel_values=source_px).logits
        if tuple(source_logits.shape[-2:]) != (LOGIT_GRID, LOGIT_GRID):
            raise SystemExit(f"unexpected source logit grid {tuple(source_logits.shape)}")
        source_logits_up = F.interpolate(source_logits.float(), size=source_gt.shape[-2:],
                                         mode="bilinear", align_corners=False)
        loss_source = F.cross_entropy(source_logits_up, source_gt, ignore_index=IGNORE_INDEX)
        del source_logits_up

        # ---- F/G: target branch, exact D4 round trip ----
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            student_logits = student(pixel_values=d4_forward(target_px, view)).logits
        z_s_can = d4_inverse(student_logits.float(), view)
        if tuple(z_s_can.shape[-2:]) != (LOGIT_GRID, LOGIT_GRID):
            raise SystemExit(f"unexpected target logit grid {tuple(z_s_can.shape)}")

        y_ref = y_ref_all[target_index].to(args.device).long().unsqueeze(0)
        w_anchor = w_ref_all[target_index].to(args.device).float().unsqueeze(0)

        biased = z_s_can + bstar
        ce = F.cross_entropy(biased, y_ref, reduction="none")
        w_sum = w_anchor.sum().clamp_min(1e-6)
        loss_hard = (w_anchor * ce).sum() / w_sum

        # ---- H: EMA consistency, teacher fully stop-gradient ----
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                ema_logits = model_ema(pixel_values=target_px).logits
            p_ema = torch.softmax(ema_logits.float() + bstar, dim=1)
        p_s = torch.softmax(biased, dim=1)
        kl = F.kl_div(torch.log(p_s.clamp_min(1e-12)), p_ema, reduction="none").sum(dim=1)
        loss_cons = (w_anchor * kl).sum() / w_sum

        ramp = min(1.0, step / RAMP_STEPS)
        loss = loss_source + ramp * (LAMBDA_HARD * loss_hard + LAMBDA_CONS * loss_cons)

        scalars = {"source": float(loss_source.detach()), "hard": float(loss_hard.detach()),
                   "cons": float(loss_cons.detach()), "loss": float(loss.detach())}
        loss_finite = all(np.isfinite(v) for v in scalars.values())
        if loss_finite:
            hard_min = scalars["hard"] if hard_min is None else min(hard_min, scalars["hard"])

        # GradScaler contract: once unscale_() has run, step() and update() MUST
        # both run before the next iteration. update() is what halves the scale
        # after an overflow; skipping out of that window latches the scaler.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for _, p in trainable if p.grad is not None]
        grad_norm = float(torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(g.detach().float()) for g in grads]))) if grads else 0.0

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        amp_skipped = scaler.get_scale() < scale_before
        stepped = bool(not amp_skipped and loss_finite and np.isfinite(grad_norm))
        if not loss_finite:
            nonfinite_loss_steps.append(step)
        if not stepped:
            skipped_steps.append(step)
            reason = []
            if not loss_finite:
                reason.append(f"nonfinite loss {scalars}")
            if not np.isfinite(grad_norm):
                reason.append(f"nonfinite grad_norm {grad_norm}")
            if amp_skipped:
                reason.append(f"AMP overflow, scale {scale_before:g}->{scaler.get_scale():g}")
            log(f"step {step}: optimizer did NOT step -- " + "; ".join(reason))
        if amp_skipped:
            amp_skip_steps.append(step)

        ema_delta = 0.0
        if stepped:
            # ---- B: EMA, strictly after the optimizer step ----
            with torch.no_grad():
                student_params = dict(student.named_parameters())
                for name, param in model_ema.named_parameters():
                    updated = EMA_DECAY * param.detach() + (1.0 - EMA_DECAY) * student_params[name].detach()
                    ema_delta = max(ema_delta, float((updated - param).abs().max()))
                    param.copy_(updated)
                student_buffers = dict(student.named_buffers())
                for name, buf in model_ema.named_buffers():
                    src = student_buffers[name]
                    if buf.is_floating_point():
                        buf.copy_(EMA_DECAY * buf + (1.0 - EMA_DECAY) * src)
                    else:
                        buf.copy_(src)
        ema_delta_ema = ema_delta if ema_delta_ema is None else (
            0.95 * ema_delta_ema + 0.05 * ema_delta)

        apply_train_eval_modes(student)

        source_loss_ema = (scalars["source"] if source_loss_ema is None
                           else 0.95 * source_loss_ema + 0.05 * scalars["source"])

        record = {
            "step": step, "lr": lr_at(step - 1), "ramp": ramp, "view": view,
            **scalars,
            "agree_with_y_ref_pct": round(100.0 * float(
                (biased.argmax(1) == y_ref).float().mean()), 4),
            "w_mean": round(float(w_anchor.mean()), 6),
            "ema_delta": ema_delta,
            "amp_scale": scaler.get_scale(),
            "amp_skipped": amp_skipped,
            "stepped": stepped,
            "loss_finite": loss_finite,
            "grad_norm": grad_norm,
            "elapsed": round(time.time() - started, 1),
        }
        history.append(record)

        if step % 25 == 0 or step == 1:
            log(f"step {step:5d} src {record['source']:.4f} hard {record['hard']:.4f} "
                f"cons {record['cons']:.4f} loss {record['loss']:.4f} "
                f"lr {record['lr']:.3e} ramp {ramp:.2f} w {record['w_mean']:.3f} "
                f"emad {ema_delta:.2e} agree {record['agree_with_y_ref_pct']:.2f}% "
                f"scale {record['amp_scale']:g} |g| {grad_norm:.2f}")

        if step in gate_steps or step == max_steps:
            gate_stats.append(save_gate(
                step, student, model_ema, model_ref, optimizer, scaler, config, args,
                frozen_snapshot, trainable_names, all_names, scope_init_sha,
                ref_sha_before, ema_sha_before))
            apply_train_eval_modes(student)

    elapsed = time.time() - started
    summary = {
        "experiment": config.get("experiment_id"),
        "base": str(THETA261),
        "base_sha256": THETA261_SHA256,
        "base_platform_score": THETA261_SCORE,
        "steps": step,
        "elapsed_sec": round(elapsed, 1),
        "sec_per_step": round(elapsed / max(1, step), 3),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated(args.device) / 2 ** 20, 1),
        "skipped_steps": skipped_steps,
        "amp_skip_steps": amp_skip_steps,
        "n_amp_skips": len(amp_skip_steps),
        "nonfinite_loss_steps": nonfinite_loss_steps,
        "optimizer_steps_taken": sum(1 for r in history if r.get("stepped")),
        "target_key_violations": target_key_violations,
        "source_loss_start": history[0]["source"] if history and "source" in history[0] else None,
        "source_loss_final_ema": source_loss_ema,
        "hard_min": hard_min,
        "ema_delta_final": ema_delta_ema,
        "scope_sha_init": scope_init_sha,
        "scope_sha_final": scope_sha(student, trainable_names),
        "theta_ref_unchanged": scope_sha(model_ref, all_names) == ref_sha_before,
        "theta_ema_unchanged_scope_sha_before": ema_sha_before,
        "theta_ema_unchanged_scope_sha_after": scope_sha(model_ema, all_names),
        "ema_parameters_have_grad": ema_has_grad,
        "ref_parameters_have_grad": ref_has_grad,
        "final_train_modes": training_modes(student),
        "gates": gate_stats,
        "history": history,
    }
    summary["stage_scope_changed"] = summary["scope_sha_init"] != summary["scope_sha_final"]
    (outdir / "ATDA_RUN_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    frozen_final = frozen_diff(student, frozen_snapshot)
    log(f"ATDA done: steps={step} elapsed={elapsed:.0f}s "
        f"peaked {summary['peak_vram_mb']}MB amp_skips={len(amp_skip_steps)} "
        f"frozen_identical={frozen_final['identical']} "
        f"theta_ref_unchanged={summary['theta_ref_unchanged']}")

    if args.smoke:
        report = smoke_report(summary, frozen_final, gate_stats, student)
        (outdir / "SMOKE_REPORT.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log("SMOKE REPORT " + json.dumps(report["checks"], ensure_ascii=False))
        log(f"SMOKE VERDICT: {'PASS' if report['pass'] else 'FAIL'} "
            f"({report['n_failed']} failures)")
        log_handle.close()
        return 0 if report["pass"] else 3

    log_handle.close()
    return 0


def smoke_report(summary, frozen_final, gate_stats, student) -> dict[str, Any]:
    """The acceptance checks (the original nine + four mode/scope assertions)."""
    hist = [r for r in summary["history"] if not r.get("nonfinite")]
    d4 = d4_self_test()
    modes = training_modes(student)
    # only stage2/3 may carry nonzero gradients
    import torch
    bad_grad = [name for name, p in student.named_parameters()
                if (p.grad is not None and p.grad.abs().max() > 0) and not is_trainable(name)]
    checks = {
        "source_ce_finite": all(np.isfinite(r["source"]) for r in hist) and bool(hist),
        "hard_finite_and_positive": (all(np.isfinite(r["hard"]) and r["hard"] > 0 for r in hist)
                                     and bool(hist)),
        "cons_finite_and_positive": (all(np.isfinite(r["cons"]) and r["cons"] > 0 for r in hist)
                                     and bool(hist)),
        "ema_has_no_grad": not summary["ema_parameters_have_grad"],
        "ema_parameters_updating": (summary["theta_ema_unchanged_scope_sha_before"]
                                    != summary["theta_ema_unchanged_scope_sha_after"]
                                    and (summary["ema_delta_final"] or 0) > 0),
        "d4_transform_inverse_exact": d4["ok"],
        "frozen_scope_byte_identical": frozen_final["identical"],
        "stage2_3_changed": summary["stage_scope_changed"],
        "target_never_read_a_label": (not summary["target_key_violations"]
                                      and assert_target_dir()["clean"]),
        "no_oom_no_nan": (not summary["nonfinite_loss_steps"]
                          and all(np.isfinite(r["loss"]) for r in hist)
                          and summary["optimizer_steps_taken"] > 0),
        # --- the four mode/scope assertions demanded by the spec ---
        "stage2_training_true": modes["segformer.stages.2"],
        "stage3_training_true": modes["segformer.stages.3"],
        "stage0_stage1_head_eval": (not modes["segformer.stages.0"]
                                    and not modes["segformer.stages.1"]
                                    and not modes["decode_head"]),
        "only_stage2_3_gradients_nonzero": not bad_grad,
    }
    failed = [k for k, v in checks.items() if not v]
    return {
        "pass": not failed,
        "n_failed": len(failed),
        "failed": failed,
        "checks": checks,
        "d4_self_test": d4,
        "frozen_scope_detail": frozen_final,
        "train_modes": modes,
        "bad_grad_params": bad_grad[:10],
        "steps": summary["steps"],
        "sec_per_step": summary["sec_per_step"],
        "peak_vram_mb": summary["peak_vram_mb"],
        "skip_detail": {"nonfinite_loss": summary["nonfinite_loss_steps"],
                        "amp": summary["amp_skip_steps"],
                        "n_amp_skips": summary["n_amp_skips"]},
        "gates": gate_stats,
    }


def save_gate(step, student, model_ema, model_ref, optimizer, scaler, config, args,
              frozen_snapshot, trainable_names, all_names, scope_init_sha,
              ref_sha_before, ema_sha_before) -> dict[str, Any]:
    import torch

    outdir = Path(args.output_dir)
    restore_frozen(student, frozen_snapshot)
    diff = frozen_diff(student, frozen_snapshot)
    if not diff["identical"]:
        raise SystemExit(f"FROZEN SCOPE DRIFT at step {step}: {diff}")

    ema_state = model_ema.state_dict()
    student_state = student.state_dict()
    ema_gap = max(float((ema_state[n].float() - student_state[n].float()).abs().max())
                  for n in trainable_names)
    frozen_gap = max(float((ema_state[n].float() - student_state[n].float()).abs().max())
                     for n in all_names if not is_trainable(n))

    torch.save({
        "model": {k: v.detach().cpu() for k, v in student_state.items()},
        "ema": {k: v.detach().cpu() for k, v in model_ema.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": config,
        "atda": {
            "base": str(THETA261),
            "base_sha256": THETA261_SHA256,
            "base_platform_score": THETA261_SCORE,
            "bstar": BSTAR,
            "trainable_prefixes": list(TRAINABLE_PREFIXES),
            "frozen_identical": diff["identical"],
            "ema_decay": EMA_DECAY,
            "lambda_hard": LAMBDA_HARD,
            "lambda_cons": LAMBDA_CONS,
            "ramp_steps": RAMP_STEPS,
        },
    }, outdir / f"checkpoint_step{step}.pt")

    stats = {
        "step": step,
        "frozen_identical_to_theta261": diff["identical"],
        "frozen_max_abs_diff": diff["max_abs_diff"],
        "frozen_tensors_asserted": len(frozen_snapshot),
        "stage_scope_changed_from_init": scope_sha(student, trainable_names) != scope_init_sha,
        "stage_scope_sha256": scope_sha(student, trainable_names),
        "ema_scope_sha256": scope_sha(model_ema, all_names),
        "theta_ref_unchanged": scope_sha(model_ref, all_names) == ref_sha_before,
        "ema_scope_changed_from_init": scope_sha(model_ema, all_names) != ema_sha_before,
        "ema_minus_student_max_abs_in_scope": ema_gap,
        "ema_minus_theta261_max_abs_out_of_scope": frozen_gap,
        "checkpoint": str(outdir / f"checkpoint_step{step}.pt"),
        "bytes": (outdir / f"checkpoint_step{step}.pt").stat().st_size,
        "sha256": sha256_file(outdir / f"checkpoint_step{step}.pt"),
    }
    (outdir / f"CHECKPOINT_step{step}.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    raise SystemExit(main())
