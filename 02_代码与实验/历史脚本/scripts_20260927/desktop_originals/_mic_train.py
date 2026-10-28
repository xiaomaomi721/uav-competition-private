#!/usr/bin/env python3
"""A05 + MIC (Masked Image Consistency) paired experiment.
CONTROL: apply_mic_mask=False (student sees unmasked strong target)
TREAT:   apply_mic_mask=True  (student sees 70% block-masked strong target)
Both: EMA teacher on clean target, q-weighted target CE.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
CONFIG = SNAP / 'stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml'
A05 = Path('/home/jhy/rare_diag_20260915/breakthrough_v2_20260926/03_weight_projection/A05_WEIGHT_alpha05.pt')
TEST2 = Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')
BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
EXPECTED_A05_SHA = '023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134'
TRAIN_PREFIX = ('segformer.stages.2.', 'segformer.stages.3.')
HEAD_PREFIX = 'decode_head.'
BASE_LR = 1e-5
HEAD_MULT = 0.05
WD = 0.01
THRESH = 0.968
MASK_RATIO = 0.70
MASK_BLOCK = 64
EMA_ALPHA_MAX = 0.999

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def is_train(name):
    return name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX)

def strong_photo(x, rng_state):
    """Strong photo augmentation using provided RNG state."""
    mean = torch.tensor([.485,.456,.406], device=x.device, dtype=x.dtype)[None,:,None,None]
    std = torch.tensor([.229,.224,.225], device=x.device, dtype=x.dtype)[None,:,None,None]
    x = (x*std+mean).clamp(0,1)
    br = 1+rng_state.uniform(-.15,.15); ct = 1+rng_state.uniform(-.15,.15)
    sat = 1+rng_state.uniform(-.12,.12); gam = rng_state.uniform(.9,1.1)
    x = x*br; mu = x.mean((-2,-1),keepdim=True); x = (x-mu)*ct+mu
    gray = x.mean(1,keepdim=True); x = (x-gray)*sat+gray
    x = x.clamp(0,1).pow(gam)
    noise_sigma = rng_state.uniform(0,.015)
    x = x + torch.randn_like(x)*noise_sigma
    if rng_state.random() < .25: x = F.avg_pool2d(x,3,1,1)
    return (x.clamp(0,1)-mean)/std

def make_block_mask(b, c, h, w, rng):
    """Generate block mask: 16x16 blocks of 64px, Bernoulli(0.70) masked.
    Returns mask tensor (1,1,h,w) where 1=keep, 0=masked."""
    bh, bw = h // MASK_BLOCK, w // MASK_BLOCK  # 16x16
    block_mask = torch.ones(b, 1, bh, bw)
    for bi in range(bh):
        for bj in range(bw):
            if rng.random() < MASK_RATIO:
                block_mask[0, 0, bi, bj] = 0.0
    # Resize to full resolution
    mask = F.interpolate(block_mask, size=(h, w), mode='nearest')
    # Ensure not all masked/unmasked
    if mask.mean() < 0.05 or mask.mean() > 0.95:
        return make_block_mask(b, c, h, w, rng)
    return mask

def ema_update(teacher_model, student_model, alpha):
    with torch.no_grad():
        for tp, sp in zip(teacher_model.parameters(), student_model.parameters()):
            tp.mul_(alpha).add_(sp, alpha=1-alpha)
        # Buffers: float EMA, integer copy
        for tb, sb in zip(teacher_model.buffers(), student_model.buffers()):
            if tb.dtype.is_floating_point:
                tb.mul_(alpha).add_(sb, alpha=1-alpha)
            else:
                tb.copy_(sb)

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
    ap.add_argument('--max-steps', type=int, default=8000)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
    device = a.device

    actual_sha = sha256_file(A05)
    if actual_sha != EXPECTED_A05_SHA:
        raise SystemExit(f'A05 SHA mismatch: {actual_sha}')

    seed = 20260927
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)

    # Student
    student = make_model(cfg)
    ckpt = torch.load(str(A05), map_location='cpu', weights_only=False)
    student.load_state_dict(ckpt.get('model', ckpt), strict=True)
    student = student.to(device)

    # EMA teacher (same init)
    teacher = make_model(cfg)
    teacher.load_state_dict(ckpt.get('model', ckpt), strict=True)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Module mode
    student.eval()
    student.get_submodule("segformer.stages.2").train()
    student.get_submodule("segformer.stages.3").train()
    for n, p in student.named_parameters():
        p.requires_grad_(is_train(n))

    base_params = [p for n,p in student.named_parameters() if n.startswith(TRAIN_PREFIX)]
    head_params = [p for n,p in student.named_parameters() if n.startswith(HEAD_PREFIX)]
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
    src = UAVSegmentationDataset(DATA, ids, TrainTransform(crop_size=1024, config=cfg['transforms'], rare_cropper=crop, ignore_index=255), maximum=None, seed=seed, random_subset=False)
    sl = DataLoader(src, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=segmentation_collate, generator=torch.Generator().manual_seed(seed+10001), persistent_workers=True)

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
    tl = DataLoader(Target(), batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate, generator=torch.Generator().manual_seed(seed+10002), persistent_workers=True)

    i_src = iter(sl); i_t = iter(tl)
    # Independent RNG for strong_photo and mask
    sp_rng = random.Random(seed + 20001)
    mask_rng = random.Random(seed + 30001)

    hist = []; t0 = time.time()
    amp_skips = []
    apply_mask = (a.role == 'treat')

    for step in range(1, a.max_steps+1):
        try: s = next(i_src)
        except StopIteration: i_src = iter(sl); s = next(i_src)
        for gi, g in enumerate(opt.param_groups):
            g['lr'] = (BASE_LR if gi == 0 else BASE_LR * HEAD_MULT) * max(0, 1-(step-1)/10000)

        opt.zero_grad(set_to_none=True)
        sx = s['pixel_values'].to(device); sy = s['labels'].to(device)
        with torch.amp.autocast('cuda', enabled=True):
            sz = student(pixel_values=sx).logits
        lsrc = F.cross_entropy(F.interpolate(sz.float(), size=sy.shape[-2:], mode='bilinear', align_corners=False), sy, ignore_index=255)
        scaler.scale(lsrc).backward()

        # Target
        try: t = next(i_t)
        except StopIteration: i_t = iter(tl); t = next(i_t)
        tx = t['pixel_values'].to(device)

        # Clean teacher forward
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            z_teacher = teacher(pixel_values=tx).logits.float() + bstar
        p_teacher = z_teacher.softmax(1)
        conf, y_teacher = p_teacher.max(1)
        q = (conf >= THRESH).float().mean()

        # Strong photo on student input
        x_strong = strong_photo(tx, sp_rng)
        # Block mask
        mask = make_block_mask(1, 3, 1024, 1024, mask_rng).to(device)
        if apply_mask:
            x_student = x_strong * mask
        else:
            x_student = x_strong

        with torch.amp.autocast('cuda', enabled=True):
            z_student = student(pixel_values=x_student).logits.float() + bstar
        ce_map = F.cross_entropy(z_student, y_teacher, reduction='none')
        l_target_raw = ce_map.mean()
        l_target = q * l_target_raw

        scaler.scale(l_target).backward()
        scaler.unscale_(opt)
        scale_before = scaler.get_scale()
        scaler.step(opt); scaler.update()
        stepped = scaler.get_scale() >= scale_before
        if not stepped:
            amp_skips.append(step)
        else:
            # EMA update only on successful step
            alpha_u = min(1 - 1/(step+1), EMA_ALPHA_MAX)
            ema_update(teacher, student, alpha_u)

        total_loss = lsrc + l_target
        row = {'step': step, 'source': float(lsrc), 'target_raw': float(l_target_raw),
               'q': float(q), 'target_weighted': float(l_target),
               'mask_applied': apply_mask, 'loss': float(total_loss),
               'role': a.role, 'stepped': stepped,
               'mask_coverage': float(mask.mean()),
               'elapsed': round(time.time()-t0, 1)}
        hist.append(row)
        if step % 50 == 0 or step == 1:
            print(json.dumps(row), flush=True)

        if step in (2000, 4000, 8000):
            path = out / f'checkpoint_step{step}.pt'
            torch.save({'model': student.state_dict(), 'step': step, 'role': a.role,
                        'a05_sha': EXPECTED_A05_SHA, 'apply_mask': apply_mask}, path)
            print(f"Saved {path}", flush=True)

    summary = {
        'steps': a.max_steps, 'elapsed_sec': round(time.time()-t0, 1),
        'role': a.role, 'seed': seed, 'a05_sha': EXPECTED_A05_SHA,
        'apply_mic_mask': apply_mask,
        'attempted_loops': a.max_steps,
        'successful_optimizer_updates': a.max_steps - len(amp_skips),
        'amp_skip_loops': amp_skips,
        'history_tail': hist[-10:],
    }
    (out / 'RUN_SUMMARY.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
