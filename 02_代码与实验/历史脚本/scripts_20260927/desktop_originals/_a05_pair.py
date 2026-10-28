#!/usr/bin/env python3
"""A05 Residual D1 Matched Pair: CONTROL vs TREAT.
CONTROL: target_coeff = 0 (but target branch still executes)
TREAT:   target_coeff = 0.5 * min(1, loop/500)
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROUND2 = Path('/home/jhy/rare_diag_20260915/round2_20260921')
SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
CONFIG = SNAP / 'stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml'
A05 = Path('/home/jhy/rare_diag_20260915/breakthrough_v2_20260926/03_weight_projection/A05_WEIGHT_alpha05.pt')
TEST2 = Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
EXPECTED_A05_SHA = '023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134'
TRAIN_PREFIX = ('segformer.stages.2.', 'segformer.stages.3.')
HEAD_PREFIX = 'decode_head.'
MAX_STEPS = 2000
BASE_LR = 1e-5
HEAD_MULT = 0.05  # head_lr = base_lr * 0.05 = 5e-7
WD = 0.01
RAMP = 500
CONF_AGREE = 0.98
SEED = 20260922

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def is_train(name):
    return name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX)

def strong_photo(x):
    mean = torch.tensor([.485,.456,.406], device=x.device, dtype=x.dtype)[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device=x.device, dtype=x.dtype)[None,:,None,None]
    x = (x*std+mean).clamp(0,1)
    br = 1+random.uniform(-.15,.15); ct = 1+random.uniform(-.15,.15)
    sat = 1+random.uniform(-.12,.12); gam = random.uniform(.9,1.1)
    x = x*br; mu = x.mean((-2,-1),keepdim=True); x = (x-mu)*ct+mu
    gray = x.mean(1,keepdim=True); x = (x-gray)*sat+gray
    x = x.clamp(0,1).pow(gam); x = x + torch.randn_like(x)*random.uniform(0,.015)
    if random.random() < .25: x = F.avg_pool2d(x,3,1,1)
    return (x.clamp(0,1)-mean)/std

def d4_forward(x, view):
    k, hf = divmod(view, 2)
    if k: x = torch.rot90(x, k, (-2,-1))
    if hf: x = x.flip((-1,))
    return x

def d4_inverse(z, view):
    k, hf = divmod(view, 2)
    if hf: z = z.flip((-1,))
    if k: z = torch.rot90(z, -k, (-2,-1))
    return z

def main():
    from torch.utils.data import DataLoader
    from src.models import build_model as make_model
    from src.utils import load_config
    from src.transforms import normalize_to_tensor
    from PIL import Image

    for p in (str(SNAP), '/home/jhy/gpu2_day_queue'):
        if p not in sys.path: sys.path.insert(0, p)

    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--role', choices=['control', 'treat'], required=True)
    ap.add_argument('--max-steps', type=int, default=MAX_STEPS)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
    device = a.device

    # Verify A05 SHA
    actual_sha = sha256_file(A05)
    if actual_sha != EXPECTED_A05_SHA:
        raise SystemExit(f'A05 SHA mismatch: expected {EXPECTED_A05_SHA}, got {actual_sha}')

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)

    model = make_model(cfg)
    ckpt = torch.load(str(A05), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt), strict=True)
    model = model.to(device)

    # Module mode: eval() but train stage2/3
    model.eval()
    model.get_submodule("segformer.stages.2").train()
    model.get_submodule("segformer.stages.3").train()

    for n, p in model.named_parameters():
        p.requires_grad_(is_train(n))

    base_params = [p for n,p in model.named_parameters() if n.startswith(TRAIN_PREFIX)]
    head_params = [p for n,p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([
        {'params': base_params, 'lr': BASE_LR},
        {'params': head_params, 'lr': BASE_LR * HEAD_MULT},
    ], weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    bstar = torch.from_numpy(BSTAR).to(device)[None,:,None,None]

    from src.dataset import UAVSegmentationDataset, segmentation_collate
    from src.rare_crop import RareClassCropper
    from src.transforms import TrainTransform
    DATA = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集')
    ids = [x.strip() for x in (SNAP/'splits/train.txt').read_text().split() if x.strip()]
    crop = RareClassCropper(crop_size=1024, rare_probability=.5, priority_ids=(7,4,6), max_offset_fraction=.25)
    src = UAVSegmentationDataset(DATA, ids, TrainTransform(crop_size=1024, config=cfg['transforms'], rare_cropper=crop, ignore_index=255), maximum=None, seed=SEED, random_subset=False)
    sl = DataLoader(src, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=segmentation_collate, generator=torch.Generator().manual_seed(SEED+10001), persistent_workers=True)

    class Target:
        def __init__(self):
            self.names = [f'test2_{i}.png' for i in range(1, 1301)]
        def __len__(self): return len(self.names)
        def __getitem__(self, i):
            n = self.names[i]
            with Image.open(TEST2/n) as f: x = np.asarray(f.convert('RGB')).copy()
            return {'pixel_values': normalize_to_tensor(x), 'name': n}
    def collate(b):
        return {'pixel_values': torch.stack([x['pixel_values'] for x in b]), 'name': [x['name'] for x in b]}
    tl = DataLoader(Target(), batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate, generator=torch.Generator().manual_seed(SEED+10002), persistent_workers=True)

    i_src = iter(sl); i_t = iter(tl)
    hist = []; t0 = time.time()
    amp_skips = []

    for step in range(1, a.max_steps+1):
        try: s = next(i_src)
        except StopIteration: i_src = iter(sl); s = next(i_src)
        for gi, g in enumerate(opt.param_groups):
            g['lr'] = (BASE_LR if gi == 0 else BASE_LR * HEAD_MULT) * (1 - (step-1)/10000)

        opt.zero_grad(set_to_none=True)
        sx = s['pixel_values'].to(device); sy = s['labels'].to(device)
        with torch.amp.autocast('cuda', enabled=True):
            sz = model(pixel_values=sx).logits
        lsrc = F.cross_entropy(F.interpolate(sz.float(), size=sy.shape[-2:], mode='bilinear', align_corners=False), sy, ignore_index=255)

        # Target branch (ALWAYS executed, even for CONTROL)
        try: t = next(i_t)
        except StopIteration: i_t = iter(tl); t = next(i_t)
        stem = Path(t['name'][0]).stem
        tx = t['pixel_values'].to(device)
        view = random.randrange(8)
        aug = strong_photo(d4_forward(tx, view))
        with torch.amp.autocast('cuda', enabled=True):
            uz = model(pixel_values=aug).logits
        z = d4_inverse(uz.float(), view) + bstar
        z261 = torch.from_numpy(np.load(ROUND2/f'cache_261_test2_raw/scores/{stem}.npy', allow_pickle=False).astype(np.float32)).to(device) + bstar
        z278 = torch.from_numpy(np.load(ROUND2/f'cache_278_test2_raw/scores/{stem}.npy', allow_pickle=False).astype(np.float32)).to(device) + bstar
        p278 = z278.softmax(1); c278, y278 = p278.max(1)
        y261 = z261.argmax(0) if z261.dim() == 3 else z261.argmax(1)
        if y261.dim() == 3: y261 = y261[0]
        agree = (y261 == y278)
        mh = agree & (c278 >= CONF_AGREE)

        # Compute hard CE (always, even for CONTROL)
        lh = lsrc.new_zeros(())
        if mh.any():
            lh = F.cross_entropy(z, y278, reduction='none')[mh].mean()

        # Unique difference: target coefficient
        ramp = min(1.0, step / RAMP)
        if a.role == 'control':
            coeff = 0.0
        else:
            coeff = 0.5 * ramp

        loss = lsrc + coeff * lh

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scale_before = scaler.get_scale()
        scaler.step(opt); scaler.update()
        stepped = scaler.get_scale() >= scale_before
        if not stepped:
            amp_skips.append(step)

        row = {'step': step, 'source': float(lsrc), 'hard_ce': float(lh),
               'coeff': coeff, 'loss': float(loss),
               'hard_cov': float(mh.float().mean()),
               'role': a.role, 'stepped': stepped,
               'elapsed': round(time.time()-t0, 1)}
        hist.append(row)
        if step % 50 == 0 or step == 1:
            print(json.dumps(row), flush=True)

        if step in (1000, 2000):
            path = out / f'checkpoint_step{step}.pt'
            torch.save({'model': model.state_dict(), 'step': step, 'role': a.role,
                        'a05_sha': EXPECTED_A05_SHA}, path)
            print(f"Saved {path}", flush=True)

    summary = {'steps': a.max_steps, 'elapsed_sec': round(time.time()-t0, 1),
               'role': a.role, 'seed': SEED, 'a05_sha': EXPECTED_A05_SHA,
               'attempted_loops': a.max_steps,
               'successful_optimizer_updates': a.max_steps - len(amp_skips),
               'amp_skip_loops': amp_skips,
               'history_tail': hist[-10:]}
    (out / 'RUN_SUMMARY.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
