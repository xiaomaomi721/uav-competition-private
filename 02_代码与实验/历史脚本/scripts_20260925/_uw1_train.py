#!/usr/bin/env python3
"""Alpha050 Uncertainty-Weighted Adaptation v1.
Start from alpha050 = theta278 + 0.5*(D1-theta278).
Online teacher forward on raw test2 images; student forward on augmented.
Pseudo-label = teacher argmax, weight = (1 - teacher_conf).
500 steps, source CE + ramp * weighted hard CE.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
from _r2_contracts import (BSTAR, CONFIG, DATA, OUT, ROUND2, SNAP, TEST2, THETA278,
                          d4_forward, d4_inverse, load_state, sha256_file, write_json)

TRAIN_PREFIX = ('segformer.stages.2.', 'segformer.stages.3.')
HEAD_PREFIX = 'decode_head.'
BASE_LR = 1e-5
HEAD_MULT = 0.05
WD = 0.01
RAMP = 250
LAMBDA_HARD = 0.5
CONF_THRESH = 0.98
GAMMA = 1.0  # weight = (1-conf)^gamma

D1_CKPT = Path('/home/jhy/rare_diag_20260915/round2_multiprobe_20260922/03_dart_fixed/D1/checkpoint_step2000.pt')
ALPHA = 0.5

def is_train(name):
    return name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX)

def mode_map(model):
    model.eval()
    model.get_submodule('segformer.stages.2').train()
    model.get_submodule('segformer.stages.3').train()

def build_model(device):
    from src.models import build_model as make_model
    from src.utils import load_config
    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': 0.3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    # Load theta278
    st278, _ = load_state(THETA278)
    m = make_model(cfg)
    m.load_state_dict(st278, strict=True)
    # Load D1 and interpolate to alpha050
    import torch
    d1_obj = torch.load(D1_CKPT, map_location='cpu', weights_only=False)
    d1_sd = d1_obj['model'] if 'model' in d1_obj else d1_obj
    with torch.no_grad():
        for k in st278:
            if k in d1_sd:
                st278[k] = (1 - ALPHA) * st278[k].float() + ALPHA * d1_sd[k].float()
    m.load_state_dict(st278, strict=True)
    return m.to(device)

class Target:
    def __init__(self):
        from PIL import Image
        self.names = [f'test2_{i}.png' for i in range(1, 1301)]
        assert all((TEST2 / n).is_file() for n in self.names)
    def __len__(self):
        return len(self.names)
    def __getitem__(self, i):
        from PIL import Image
        from src.transforms import normalize_to_tensor
        n = self.names[i]
        with Image.open(TEST2 / n) as f:
            x = np.asarray(f.convert('RGB')).copy()
        return {'pixel_values': normalize_to_tensor(x), 'name': n}

def collate(b):
    import torch
    return {'pixel_values': torch.stack([x['pixel_values'] for x in b]),
            'name': [x['name'] for x in b]}

def strong_photo(x):
    import torch, torch.nn.functional as F
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-steps', type=int, default=500)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    for p in (str(SNAP), '/home/jhy/gpu2_day_queue'):
        if p not in sys.path: sys.path.insert(0, p)
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from src.dataset import UAVSegmentationDataset, segmentation_collate
    from src.rare_crop import RareClassCropper
    from src.transforms import TrainTransform
    from src.utils import load_config

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=False)
    seed = 20260925; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    cfg = load_config('/home/jhy/rare_diag_20260915/atda_20260918/atda_s2s3.yaml')
    ids = [x.strip() for x in (SNAP/'splits/train.txt').read_text().split() if x.strip()]
    assert len(ids) == 5597
    crop = RareClassCropper(crop_size=1024, rare_probability=.5, priority_ids=(7,4,6), max_offset_fraction=.25)
    src = UAVSegmentationDataset(DATA, ids, TrainTransform(crop_size=1024, config=cfg['transforms'], rare_cropper=crop, ignore_index=255), maximum=None, seed=seed, random_subset=False)
    sl = DataLoader(src, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                    collate_fn=segmentation_collate, generator=torch.Generator().manual_seed(seed+10001), persistent_workers=True)
    tl = DataLoader(Target(), batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
                    collate_fn=collate, generator=torch.Generator().manual_seed(seed+10002), persistent_workers=True)

    model = build_model(a.device)
    base_params, head_params = [], []
    for n, p in model.named_parameters():
        p.requires_grad_(is_train(n))
        if n.startswith(TRAIN_PREFIX): base_params.append(p)
        elif n.startswith(HEAD_PREFIX): head_params.append(p)
    mode_map(model)
    opt = torch.optim.AdamW([{'params': base_params, 'lr': BASE_LR},
                              {'params': head_params, 'lr': BASE_LR*HEAD_MULT}], weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    bstar = torch.from_numpy(BSTAR).to(a.device)[None,:,None,None]

    i_src = iter(sl); i_t = iter(tl); hist = []; start = time.time()
    for step in range(1, a.max_steps+1):
        try: s = next(i_src)
        except StopIteration: i_src = iter(sl); s = next(i_src)
        for gi, g in enumerate(opt.param_groups):
            g['lr'] = (BASE_LR if gi == 0 else BASE_LR*HEAD_MULT) * (1 - (step-1)/10000)
        opt.zero_grad(set_to_none=True)
        sx = s['pixel_values'].to(a.device); sy = s['labels'].to(a.device)
        with torch.amp.autocast('cuda', enabled=True):
            sz = model(pixel_values=sx).logits
        lsrc = F.cross_entropy(F.interpolate(sz.float(), size=sy.shape[-2:], mode='bilinear', align_corners=False), sy, ignore_index=255)

        # Target: online teacher forward on raw image
        try: t = next(i_t)
        except StopIteration: i_t = iter(tl); t = next(i_t)
        tx = t['pixel_values'].to(a.device)
        # Teacher: raw image, no augment
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            tz = model(pixel_values=tx).logits.float()
        tz = F.interpolate(tz, size=(256,256), mode='bilinear', align_corners=False) + bstar
        p_t = tz.softmax(1); conf_t, pred_t = p_t.max(1)
        # Student: D4 view + strong photo augment
        view = random.randrange(8)
        aug = strong_photo(d4_forward(tx, view))
        with torch.amp.autocast('cuda', enabled=True):
            uz = model(pixel_values=aug).logits
        z = d4_inverse(uz.float(), view)
        z = F.interpolate(z, size=(256,256), mode='bilinear', align_corners=False) + bstar
        # Uncertainty-weighted hard CE
        mask = conf_t >= CONF_THRESH
        lh = lsrc.new_zeros(()); covh = 0.
        if mask.any():
            ce_per_pix = F.cross_entropy(z, pred_t, reduction='none')
            w = (1.0 - conf_t).clamp(min=0.0) ** GAMMA
            w_masked = w[mask]
            lh = (ce_per_pix[mask] * w_masked).sum() / w_masked.sum().clamp(min=1.0)
            covh = float(mask.float().mean())

        ramp = min(1., step / RAMP)
        loss = lsrc + ramp * LAMBDA_HARD * lh
        scaler.scale(loss).backward(); scaler.unscale_(opt); scale = scaler.get_scale()
        scaler.step(opt); scaler.update(); stepped = scaler.get_scale() >= scale
        mode_map(model)
        row = {'step': step, 'source': float(lsrc), 'hard': float(lh), 'loss': float(loss),
               'hard_coverage': covh, 'stepped': stepped, 'elapsed': round(time.time()-start, 1)}
        hist.append(row)
        if step % 50 == 0 or step == 1:
            print(json.dumps(row), flush=True)
        if step == a.max_steps:
            path = out / f'checkpoint_step{step}.pt'
            torch.save({'model': model.state_dict(), 'step': step,
                        'variant': 'UW1', 'alpha': ALPHA, 'gamma': GAMMA,
                        'conf_thresh': CONF_THRESH,
                        'init': 'alpha050',
                        'test2_ground_truth_used': False, 'platform_submission_performed': False}, path)
            write_json(out / f'CHECKPOINT_step{step}.json', {'path': str(path), 'sha256': sha256_file(path),
                                                              'step': step, 'last_record': hist[-1]})
    summary = {'variant': 'UW1', 'steps': a.max_steps, 'elapsed_sec': round(time.time()-start, 1),
               'alpha': ALPHA, 'gamma': GAMMA, 'conf_thresh': CONF_THRESH,
               'history_tail': hist[-20:]}
    write_json(out / 'UW1_RUN_SUMMARY.json', summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
