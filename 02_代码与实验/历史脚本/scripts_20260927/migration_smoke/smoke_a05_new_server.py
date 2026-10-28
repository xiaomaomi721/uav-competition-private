#!/usr/bin/env python3
"""Smoke test: A05 inference on new server, compare test2_1.png to old server.

Goal: prove torch 2.11.0+cu128 reproduces the scored A05 output byte-exact.
Protocol identical to package_a05_weight.py: single checkpoint, D4 TTA8,
FP32 native-grid mean, Bstar once, bilinear 1024, argmax+1.
"""
import os, sys, hashlib
from pathlib import Path
import numpy as np

# --- new server paths ---
ROOT = Path('/data/uav')
SRC = ROOT / 'code/src'
CONFIG = ROOT / 'code/config.yaml'
MODEL_SOURCE = ROOT / 'models/segformer-b3-ade512'
A05 = ROOT / 'checkpoints/A05_WEIGHT_alpha05.pt'
TEST2 = ROOT / 'datasets/test_2/images'
OUT = ROOT / 'code/smoke_out'

os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['UAVSEG_MODEL_SOURCE'] = str(MODEL_SOURCE)
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / 'code'))

import torch
import torch.nn.functional as F
from PIL import Image

EXPECTED_A05_SHA = '023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134'
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def load_state(path):
    obj = torch.load(str(path), map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        for k in ('model', 'student_model', 'model_state_dict', 'state_dict'):
            v = obj.get(k)
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
                return v
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError('no state')

def main():
    from src.models import build_model as mm
    from src.utils import load_config
    from src.transforms import normalize_to_tensor

    assert sha256_file(A05) == EXPECTED_A05_SHA, 'A05 SHA mismatch'
    assert torch.cuda.is_available(), 'CUDA not available'
    print(f'torch {torch.__version__} cuda {torch.version.cuda} gpu {torch.cuda.get_device_name(0)}')

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    cfg['pretrained_model_path'] = str(MODEL_SOURCE)

    model = mm(cfg)
    model.load_state_dict(load_state(A05), strict=True)
    model = model.to('cuda:0').eval()
    bias = torch.from_numpy(BSTAR).to('cuda:0')[:, None, None]

    OUT.mkdir(parents=True, exist_ok=True)

    name = 'test2_1.png'
    with Image.open(TEST2 / name) as im:
        rgb = np.asarray(im.convert('RGB')).copy()
    px = normalize_to_tensor(rgb).unsqueeze(0).to('cuda:0')
    acc = None
    with torch.inference_mode():
        for r, hf in VIEWS:
            t = px
            if r: t = torch.rot90(t, r, (-2, -1))
            if hf: t = t.flip((-1,))
            logits = model(pixel_values=t).logits.float()
            if hf: logits = logits.flip((-1,))
            if r: logits = torch.rot90(logits, -r, (-2, -1))
            canon = logits[0].contiguous()
            acc = canon if acc is None else acc + canon
        raw = acc / 8.0
        dep = F.interpolate((raw + bias)[None], size=(1024, 1024), mode='bilinear', align_corners=False)[0]
        pred = dep.argmax(0).byte().cpu().numpy()

    out_path = OUT / name
    Image.fromarray(pred + 1, mode='L').save(out_path)
    out_sha = sha256_file(out_path)
    print(f'OUT_SHA {out_sha}')
    print(f'EXPECTED_SHA 81d840195fc932d7303f9d0423cce4e11e8ee637320a4cf0f0565d43fd9f1b5e')
    print('BYTE_EXACT_MATCH' if out_sha == '81d840195fc932d7303f9d0423cce4e11e8ee637320a4cf0f0565d43fd9f1b5e' else 'MISMATCH')

if __name__ == '__main__':
    main()
