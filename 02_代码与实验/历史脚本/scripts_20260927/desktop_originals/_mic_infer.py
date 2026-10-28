#!/usr/bin/env python3
"""MIC CONTROL/TREAT step8000 inference: D4 TTA8 + Bstar, generate test2 ZIP."""
from __future__ import annotations
import argparse, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
CONFIG = SNAP / 'stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml'
TEST2 = Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def d4_views(x):
    """D4 TTA8 views."""
    views = [x]
    # hflip
    views.append(torch.flip(x, dims=[-1]))
    # vflip
    views.append(torch.flip(x, dims=[-2]))
    # rot180
    views.append(torch.flip(x, dims=[-2, -1]))
    # rot90
    views.append(torch.rot90(x, k=1, dims=[-2, -1]))
    views.append(torch.rot90(x, k=-1, dims=[-2, -1]))
    # transpose
    views.append(x.transpose(-2, -1))
    views.append(torch.rot90(x.transpose(-2, -1), k=1, dims=[-2, -1]))
    return views

def undo_d4(logits_view, idx):
    """Undo D4 transform on logits."""
    if idx == 0: return logits_view
    if idx == 1: return torch.flip(logits_view, dims=[-1])
    if idx == 2: return torch.flip(logits_view, dims=[-2])
    if idx == 3: return torch.flip(logits_view, dims=[-2, -1])
    if idx == 4: return torch.rot90(logits_view, k=-1, dims=[-2, -1])
    if idx == 5: return torch.rot90(logits_view, k=1, dims=[-2, -1])
    if idx == 6: return logits_view.transpose(-2, -1)
    if idx == 7: return torch.rot90(logits_view.transpose(-2, -1), k=-1, dims=[-2, -1])
    return logits_view

def main():
    from src.models import build_model as make_model
    from src.utils import load_config
    from src.transforms import normalize_to_tensor

    for p in (str(SNAP), '/home/jhy/gpu2_day_queue'):
        if p not in __import__('sys').path: __import__('sys').path.insert(0, p)

    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--role', required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = a.device

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)

    model = make_model(cfg)
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    bstar = torch.from_numpy(BSTAR).to(device)[None, :, None, None]

    names = sorted([f for f in TEST2.iterdir() if f.suffix == '.png'])
    pred_dir = out / 'predictions'; pred_dir.mkdir(exist_ok=True)

    t0 = __import__('time').time()
    for i, img_path in enumerate(names, 1):
        with Image.open(img_path) as f:
            img = np.asarray(f.convert('RGB'))
        x = normalize_to_tensor(img).unsqueeze(0).to(device)
        views = d4_views(x)
        logits_sum = 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            for vi, v in enumerate(views):
                z = model(pixel_values=v).logits.float()
                z = undo_d4(z, vi)
                z = F.interpolate(z, size=(1024, 1024), mode='bilinear', align_corners=False)
                logits_sum = logits_sum + z
        logits_mean = logits_sum / 8.0 + bstar
        pred = logits_mean.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8) + 1
        Image.fromarray(pred, mode='L').save(pred_dir / f'test2_{i}.png')
        if i % 100 == 0:
            print(f'{a.role}: {i}/1300, {__import__("time").time()-t0:.0f}s', flush=True)

    # ZIP
    zip_path = out / f'MIC_{a.role.upper()}_S20260927_step8000_D4_TTA8_Bstar_test2.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(pred_dir.glob('*.png')):
            zf.write(p, p.name)

    ckpt_sha = sha256_file(a.checkpoint)
    zip_sha = sha256_file(zip_path)
    manifest = {
        'role': a.role,
        'checkpoint': str(a.checkpoint),
        'checkpoint_sha256': ckpt_sha,
        'zip_path': str(zip_path),
        'zip_sha256': zip_sha,
        'num_pngs': len(list(pred_dir.glob('*.png'))),
        'protocol': 'D4_TTA8_Bstar_FP32_mean',
        'elapsed_sec': round(__import__('time').time()-t0, 1),
        'platform_submission_performed': False,
    }
    (out / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
