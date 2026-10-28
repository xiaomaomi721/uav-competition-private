#!/usr/bin/env python3
"""生成 test3 的 native-256 D4 TTA8 logits 缓存（ATDA 迁移前提）。
教师 = CTRL ckpt_005400（已收敛，TTA8 79.29%）。
复用 ATDA 的 native_tta8_mean_logits 逻辑（D4 TTA8, native 256, FP32 mean）。
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/home/jhy/aic2026_h76_20261026")
CODE = ROOT / "code" / "h76_3594"
sys.path.insert(0, str(CODE))
os.environ.setdefault("UAVSEG_MODEL_SOURCE", str(ROOT / "models" / "segformer-b3-ade512"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.models import build_model
from src.utils import load_config
from src.transforms import normalize_to_tensor

NUM_CLASSES = 8
LOGIT_GRID = 256
CKPT = ROOT / "experiments" / "ctrl" / "checkpoints" / "ckpt_005400.pt"
IMAGES_DIR = ROOT / "data" / "test3" / "images"


def d4_forward(x, view):
    k, hf = divmod(view, 2)
    if k:
        x = torch.rot90(x, k, (-2, -1))
    if hf:
        x = x.flip((-1,))
    return x


def d4_inverse(z, view):
    k, hf = divmod(view, 2)
    if hf:
        z = z.flip((-1,))
    if k:
        z = torch.rot90(z, -k, (-2, -1))
    return z.contiguous()


def native_tta8_mean_logits(model, px):
    acc = None
    with torch.no_grad():
        for index in range(8):
            transformed = d4_forward(px, index)
            with torch.amp.autocast("cuda", enabled=False):
                logits = model(pixel_values=transformed).logits
            canonical = d4_inverse(logits.float(), index)[0]
            acc = canonical if acc is None else acc + canonical
    return acc / 8.0


def main():
    import csv
    out = ROOT / "pseudo" / "teacher_test3_native256"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    cfg = load_config(str(CODE / "configs" / "base.yaml"))
    cfg["cwfa"] = {"enabled": False}; cfg["mixstyle"] = {"enabled": False}
    cfg["pretrained_model_path"] = str(ROOT / "models" / "segformer-b3-ade512")
    model = build_model(cfg)
    st = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(st.get("model", st), strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rows = list(csv.DictReader(open(ROOT / "manifests" / "semifinal_3594.csv")))
    names = [r["output_name"].replace(".png", "") for r in rows]
    t0 = time.time()
    for i, stem in enumerate(names):
        im = np.asarray(Image.open(IMAGES_DIR / f"{stem}.png").convert("RGB")).copy()
        px = normalize_to_tensor(im).unsqueeze(0).to(device)
        z = native_tta8_mean_logits(model, px)  # [8,256,256]
        np.save(out / f"{stem}.npy", z.cpu().numpy().astype(np.float16))
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(names)} elapsed={round(time.time()-t0,1)}s", flush=True)
    print("DONE", len(names), "files written to", out)


if __name__ == "__main__":
    main()
