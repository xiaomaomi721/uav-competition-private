"""H75 TargetSSL pilot shared utilities (teacher-4090).

Frozen scope: stage2 / stage3 / decode_head are frozen; only stage0 / stage1 train.
Auxiliary heads are training-only and must be dropped before submission.
"""
from __future__ import annotations
import csv, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/home/jhy/aic2026_h76_20261026")
SRC = ROOT / "code" / "h76_3594"
WORK = Path("/home/jhy/aic2026_h76_20261026/code/H75_TargetSSL_Pilot_20261028")
sys.path.insert(0, str(SRC))
os.environ.setdefault("UAVSEG_MODEL_SOURCE", str(ROOT / "models" / "segformer-b3-ade512"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.transforms import normalize_to_tensor

H75 = ROOT / "models" / "H75.pt"
H75_SHA = "b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a"
BSTAR = [-.2, .1, .0709, .4, 3.6, .4, 2.1215, -.5]
SEED = 20261028
TARGET_MANIFEST = ROOT / "manifests" / "semifinal_3594.csv"
TARGET_IMAGES = ROOT / "data" / "test3" / "images"
SOURCE_DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
EXPECTED_GPU_INDEX = os.environ.get("EXPECTED_GPU_INDEX", "3")

STAGE01 = ("segformer.stages.0.", "segformer.stages.1.")
FROZEN = ("segformer.stages.2.", "segformer.stages.3.", "decode_head.")

torch.set_num_threads(4)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def guard():
    if (WORK / "STOP").exists():
        raise RuntimeError("STOP file present")
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    uuid = None
    for line in rows.splitlines():
        c = [x.strip() for x in line.split(",")]
        if c[0] == EXPECTED_GPU_INDEX:
            uuid = c[1]
    if uuid is None:
        raise RuntimeError(f"GPU {EXPECTED_GPU_INDEX} not visible")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    others = [l for l in apps.splitlines() if l.startswith(uuid) and str(os.getpid()) not in l]
    if others:
        raise RuntimeError("foreign process on target GPU: " + repr(others))


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cfg():
    from src.utils import load_config
    cfg = load_config(str(SRC / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}
    cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    cfg["gradient_checkpointing"] = False
    return cfg


def build_model(with_head=True):
    from src.models import build_model as bm
    m = bm(get_cfg())
    m.load_state_dict(torch.load(str(H75), map_location="cpu", weights_only=False).get("model"), strict=True)
    return m


def target_rows():
    with open(TARGET_MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_rgb(path):
    with Image.open(path) as f:
        assert f.size == (1024, 1024), (str(path), f.size)
        return np.asarray(f.convert("RGB")).copy()


def patch_mask(h=1024, w=1024, patch=32, ratio=0.4, seed=0):
    """Randomly select patch*grid cells until ~ratio of pixels are masked.

    Returns bool tensor [1,1,h,w] on CPU.
    """
    gh, gw = h // patch, w // patch
    n = gh * gw
    k = max(1, int(round(ratio * n)))
    g = np.random.default_rng(seed)
    idx = g.permutation(n)[:k]
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    m = m.reshape(gh, gw)
    m = np.kron(m, np.ones((patch, patch), dtype=bool))
    return torch.from_numpy(m)[None, None]


class RGBHead(torch.nn.Module):
    """stage1 (128x128x128) -> 1024x1024x3 via 1x1 conv + PixelShuffle(8)."""

    def __init__(self, in_ch=128, scale=8):
        super().__init__()
        self.scale = scale
        self.conv = torch.nn.Conv2d(in_ch, 3 * scale * scale, 1)
        self.ps = torch.nn.PixelShuffle(scale)

    def forward(self, f):
        return self.ps(self.conv(f))


def stage_features(model, x):
    out = model(pixel_values=x, output_hidden_states=True)
    return out.hidden_states, out.logits


def frozen_snapshot(model):
    snap = {}
    for n, p in model.named_parameters():
        if n.startswith(FROZEN):
            snap[n] = p.detach().clone()
    for n, b in model.named_buffers():
        if n.startswith(FROZEN):
            snap["buf:" + n] = b.detach().clone()
    return snap


def frozen_diff(model, snap):
    bad = []
    for n, p in model.named_parameters():
        if n.startswith(FROZEN) and not torch.equal(p.detach(), snap[n]):
            bad.append(n)
    for n, b in model.named_buffers():
        if n.startswith(FROZEN) and not torch.equal(b.detach(), snap["buf:" + n]):
            bad.append("buf:" + n)
    return bad
