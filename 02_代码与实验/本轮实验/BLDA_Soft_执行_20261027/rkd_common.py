"""Reliable-KD 20261028 shared utilities (teacher-4090).

Adapted from BLDA_Soft common.py; D4 protocol identical to package_semifinal_bias.py
(verified: H75 source1399 D4-TTA8 = 77.5320 vs historical 77.5311488).
"""
from __future__ import annotations
import csv, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/home/jhy/aic2026_h76_20261026")
WORK = ROOT / "code" / "reliable_kd_20261028"
SRC = ROOT / "code" / "h76_3594"
sys.path.insert(0, str(SRC))
os.environ.setdefault("UAVSEG_MODEL_SOURCE", str(ROOT / "models" / "segformer-b3-ade512"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.transforms import normalize_to_tensor

BSTAR = [-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5]
BSTAR_T = torch.tensor(BSTAR, dtype=torch.float32)
SEED = 20261028
H75_SHA = "b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a"
H75 = ROOT / "models" / "H75.pt"
TARGET_MANIFEST = ROOT / "manifests" / "semifinal_3594.csv"
TARGET_IMAGES = ROOT / "data" / "test3" / "images"
SOURCE_DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
PRETRAINED = ROOT / "models" / "segformer-b3-ade512"
TRAIN_PREFIX = ("segformer.stages.2.", "segformer.stages.3.")
HEAD_PREFIX = "decode_head."
EXPECTED_GPU_INDEX = os.environ.get("EXPECTED_GPU_INDEX", "3")

torch.set_num_threads(4)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def guard():
    if (WORK / "STOP").exists():
        raise RuntimeError("STOP file present")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"], text=True)
    target = None
    for line in rows.splitlines():
        c = [x.strip() for x in line.split(",")]
        if c[0] == EXPECTED_GPU_INDEX:
            target = c[1]
    if target is None:
        raise RuntimeError(f"GPU {EXPECTED_GPU_INDEX} not visible")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    other = [l for l in apps.splitlines()
             if l.startswith(target) and str(os.getpid()) not in l]
    if other:
        raise RuntimeError("foreign process on target GPU: " + repr(other))


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cfg():
    from src.utils import load_config
    cfg = load_config(str(SRC / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}
    cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(PRETRAINED)
    cfg["gradient_checkpointing"] = False
    return cfg


def state_from(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "student_model", "model_state_dict", "state_dict"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def model_from(path=None, device="cuda:0"):
    from src.models import build_model
    path = Path(path or H75)
    if path == H75:
        assert sha256(path) == H75_SHA, "H75 SHA mismatch"
    m = build_model(get_cfg())
    m.load_state_dict(state_from(path), strict=True)
    return m.to(device)


def d4(x, v):
    k, flip = divmod(v, 2)
    if k:
        x = torch.rot90(x, k, (-2, -1))
    if flip:
        x = x.flip(-1)
    return x


def inv_d4(x, v):
    k, flip = divmod(v, 2)
    if flip:
        x = x.flip(-1)
    if k:
        x = torch.rot90(x, -k, (-2, -1))
    return x.contiguous()


def strong_photo(x, rng, generator):
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    x = (x * std + mean).clamp(0, 1)
    br = 1 + rng.uniform(-.15, .15); ct = 1 + rng.uniform(-.15, .15)
    sat = 1 + rng.uniform(-.12, .12); gam = rng.uniform(.9, 1.1)
    x = x * br
    mu = x.mean((-2, -1), keepdim=True); x = (x - mu) * ct + mu
    gray = x.mean(1, keepdim=True); x = (x - gray) * sat + gray
    x = x.clamp(0, 1).pow(gam)
    x = x + torch.randn(x.shape, device=x.device, dtype=x.dtype,
                        generator=generator) * rng.uniform(0, .015)
    if rng.random() < .25:
        x = F.avg_pool2d(x, 3, 1, 1)
    return (x.clamp(0, 1) - mean) / std


def target_rows():
    with open(TARGET_MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_rgb(path):
    with Image.open(path) as f:
        assert f.size == (1024, 1024), (str(path), f.size)
        return np.asarray(f.convert("RGB")).copy()


@torch.inference_mode()
def predict_raw_tta(model, x):
    """D4-TTA8 raw logits, native 256, FP32 mean. No Bstar."""
    acc = None
    for v in range(8):
        z = inv_d4(model(pixel_values=d4(x, v)).logits.float(), v)
        acc = z if acc is None else acc + z
    return acc / 8.0


def predict_raw_single(model, x):
    return model(pixel_values=x).logits.float()


def d4_stack_with_votes(model, x):
    """Return mean raw logits, per-view calibrated argmax stack.

    z_i aligned -> z_T = mean -> q = softmax(z_T + b)
    votes = number of views whose argmax(z_i + b) == argmax(z_T + b)
    """
    b = BSTAR_T.to(x.device)[None, :, None, None]
    zs = []
    for v in range(8):
        zs.append(inv_d4(model(pixel_values=d4(x, v)).logits.float(), v))
    zT = sum(zs) / 8.0
    cal = zT + b
    cal_arg = cal.argmax(1)
    votes = torch.zeros_like(cal_arg, dtype=torch.uint8)
    for z in zs:
        votes += ((z + b).argmax(1) == cal_arg).to(torch.uint8)
    q = cal.softmax(1)
    hn = -(q.clamp_min(1e-8).log() * q).sum(1) / np.log(8.0)
    raw_arg = zT.argmax(1)
    return zT, votes, hn.clamp(0.0, 1.0), (raw_arg != cal_arg)


def logits_to_png(raw, bias=True):
    if bias:
        raw = raw + BSTAR_T.to(raw.device)[None, :, None, None]
    return F.interpolate(raw, size=(1024, 1024), mode="bilinear",
                         align_corners=False).argmax(1)
