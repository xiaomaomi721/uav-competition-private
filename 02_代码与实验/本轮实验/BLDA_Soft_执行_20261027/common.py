"""Shared, isolated H75 experiment utilities. Physical GPU1 only."""
from __future__ import annotations
import os
GPU_UUID = "GPU-9f3707b0-4a7d-0ba5-6c3b-463906c93bd7"
if os.environ.get("CUDA_VISIBLE_DEVICES", GPU_UUID) != GPU_UUID:
    raise RuntimeError("This experiment is restricted to new-server physical GPU1")
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import csv, hashlib, json, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
ROOT = Path(__file__).resolve().parent
ASSETS = json.loads((ROOT / "assets.json").read_text())
sys.path.insert(0, str(ROOT / "snapshot"))
os.environ["UAVSEG_MODEL_SOURCE"] = ASSETS["pretrained"]
from src.transforms import normalize_to_tensor
BSTAR = [-.2,.1,.0709,.4,3.6,.4,2.1215,-.5]
SEED = 20261027
PREFIX = ("segformer.stages.2.", "segformer.stages.3.", "decode_head.")
torch.set_num_threads(4)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()

def dump(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(p)

def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)

def guard():
    if (ROOT / "STOP").exists(): raise RuntimeError("User/agent STOP file present")
    rows = subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    if not any(line.strip().startswith("1,") and GPU_UUID in line for line in rows.splitlines()):
        raise RuntimeError("GPU1 identity changed")
    apps = subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_gpu_memory","--format=csv,noheader"],text=True)
    other = []
    for line in apps.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if cols and cols[0] == GPU_UUID and int(cols[1]) != os.getpid(): other.append(line)
    if other: raise RuntimeError("Other processes on GPU1; refusing to compete: " + repr(other))

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_cfg():
    from src.utils import load_config
    cfg = load_config(str(ROOT / "snapshot/config.yaml"))
    cfg["cwfa"] = {"enabled":False}
    cfg.pop("mixstyle",None)
    cfg["pretrained_model_path"] = ASSETS["pretrained"]
    cfg["gradient_checkpointing"] = False
    return cfg

def state_from(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "student_model", "model_state_dict", "state_dict"):
        if isinstance(obj,dict) and isinstance(obj.get(key),dict): return obj[key]
    return obj

def model_from(path=None):
    from src.models import build_model
    path = path or ASSETS["model"]
    if str(path) == ASSETS["model"]:
        assert sha256(path) == ASSETS["model_sha256"], "H75 hash changed"
    model = build_model(get_cfg())
    model.load_state_dict(state_from(path), strict=True)
    return model.to("cuda:0").eval()

def d4(x, v):
    k, flip = divmod(v,2)
    if k: x = torch.rot90(x,k,(-2,-1))
    if flip: x = x.flip(-1)
    return x

def inv_d4(x,v):
    k, flip = divmod(v,2)
    if flip: x = x.flip(-1)
    if k: x = torch.rot90(x,-k,(-2,-1))
    return x.contiguous()

def strong_photo(x, rng, generator):
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    x=(x*std+mean).clamp(0,1)
    br=1+rng.uniform(-.15,.15); ct=1+rng.uniform(-.15,.15)
    sat=1+rng.uniform(-.12,.12); gam=rng.uniform(.9,1.1)
    x=x*br
    mu=x.mean((-2,-1),keepdim=True); x=(x-mu)*ct+mu
    gray=x.mean(1,keepdim=True); x=(x-gray)*sat+gray
    x=x.clamp(0,1).pow(gam)
    x=x+torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)*rng.uniform(0,.015)
    if rng.random()<.25: x=F.avg_pool2d(x,3,1,1)
    return (x.clamp(0,1)-mean)/std

def target_rows():
    with open(ASSETS["target_manifest"],newline="") as f: return list(csv.DictReader(f))

def read_rgb(path):
    with Image.open(path) as f:
        assert f.size==(1024,1024), (str(path),f.size)
        return np.asarray(f.convert("RGB")).copy()

@torch.inference_mode()
def predict_raw(model, x, tta=True):
    if not tta: return model(pixel_values=x).logits.float()
    acc=None
    for v in range(8):
        z=inv_d4(model(pixel_values=d4(x,v)).logits.float(),v)
        acc=z if acc is None else acc+z
    return acc/8

def logits_prediction(raw, bias=True):
    if bias: raw=raw+raw.new_tensor(BSTAR)[None,:,None,None]
    return F.interpolate(raw,size=(1024,1024),mode="bilinear",align_corners=False).argmax(1)

