#!/usr/bin/env python3
"""Multi-scale D4 TTA for alpha050 = theta278 + 0.5*(D1-theta278).
Scales: 0.75, 1.0, 1.25. Saves fused [8,256,256] fp16 logits.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
CONFIG = SNAP / 'stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml'
TEST = Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')
THETA278 = Path('/home/jhy/rare_diag_20260915/atda_20260918/cpu_candidates/f_s2p100_s3p125.pt')
D1_CKPT = Path('/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt')
ALPHA = 0.5
SCALES = [0.75, 1.0, 1.25]
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
OUTDIR = Path('/home/jhy/rare_diag_20260915/mstta_alpha050_20260925')

def build_model(device):
    import torch
    from src.utils import load_config
    from src.models import build_model as make_model
    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': 0.3, 'epsilon': 4.0, 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    # theta278 state
    ck278 = torch.load(str(THETA278), map_location='cpu', weights_only=False)
    sd278 = ck278.get('model', ck278) if isinstance(ck278, dict) else ck278
    # D1 state
    d1 = torch.load(str(D1_CKPT), map_location='cpu', weights_only=False)
    sd1 = d1.get('model', d1) if isinstance(d1, dict) else d1
    # Interpolate
    with torch.no_grad():
        for k in sd278:
            if k in sd1:
                sd278[k] = (1 - ALPHA) * sd278[k].float() + ALPHA * sd1[k].float()
    m = make_model(cfg)
    m.load_state_dict(sd278, strict=True)
    return m.to(device).eval()

def d4f(t, i):
    import torch
    k, hf = VIEWS[i]
    if k: t = torch.rot90(t, k, (-2, -1))
    if hf: t = t.flip((-1,))
    return t

def d4i(t, i):
    import torch
    k, hf = VIEWS[i]
    if hf: t = t.flip((-1,))
    if k: t = torch.rot90(t, -k, (-2, -1))
    return t.contiguous()

def forward_scale(model, px, scale, device):
    """px is normalized tensor [1,3,H,W] at 1024x1024. Resize, D4, inverse, resize to 256."""
    import torch
    import torch.nn.functional as F
    _, _, H, W = px.shape
    new_h, new_w = int(H * scale), int(W * scale)
    if scale != 1.0:
        px_s = F.interpolate(px, size=(new_h, new_w), mode='bilinear', align_corners=False)
    else:
        px_s = px
    outs = []
    with torch.inference_mode():
        for i in range(8):
            with torch.amp.autocast('cuda', enabled=False):
                z = model(pixel_values=d4f(px_s, i)).logits.float()
            z = d4i(z, i)[0]  # [8, h/4, w/4]
            # Resize to 256x256
            if z.shape[-2:] != (256, 256):
                z = F.interpolate(z.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)[0]
            outs.append(z)
    return torch.stack(outs, 0).mean(0)  # [8,256,256]

def main():
    import torch
    from PIL import Image
    from src.transforms import normalize_to_tensor
    device = 'cuda:0'
    for p in ('/home/jhy/gpu2_day_queue', str(SNAP)):
        if p not in sys.path: sys.path.insert(0, p)

    (OUTDIR / 'scores').mkdir(parents=True, exist_ok=False)
    names = sorted((p.name for p in TEST.glob('*.png')), key=lambda n: int(Path(n).stem.split('_')[-1]))
    assert names == [f'test2_{i}.png' for i in range(1, 1301)]
    print(f'Building alpha050 model...', flush=True)
    model = build_model(device)
    print(f'Model ready. Running {len(SCALES)} scales x D4 TTA8...', flush=True)
    t0 = time.time()
    for j, n in enumerate(names, 1):
        with Image.open(TEST / n) as im:
            img = np.asarray(im.convert('RGB')).copy()
        px = normalize_to_tensor(img).unsqueeze(0).to(device)
        scale_logits = []
        for sc in SCALES:
            z = forward_scale(model, px, sc, device)
            scale_logits.append(z)
        fused = torch.stack(scale_logits, 0).mean(0)  # [8,256,256]
        np.save(OUTDIR / 'scores' / f'{Path(n).stem}.npy', fused.half().cpu().numpy())
        del px, scale_logits, fused
        if j % 100 == 0:
            el = time.time() - t0
            print(f'{j}/1300 {el:.0f}s {el/j:.3f}s/img', flush=True)
    meta = {
        'scales': SCALES, 'alpha': ALPHA, 'count': 1300,
        'protocol': f'multi-scale {SCALES} x D4 TTA8, fused mean at 256x256, fp16',
        'runtime_sec': round(time.time() - t0, 1),
        'theta278': str(THETA278), 'd1': str(D1_CKPT),
    }
    (OUTDIR / 'META.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
