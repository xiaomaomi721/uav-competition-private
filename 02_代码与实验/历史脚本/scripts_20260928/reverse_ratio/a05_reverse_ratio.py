#!/usr/bin/env python3
"""Round-2 large-lever: reverse target/source ratio (de-tax the blind-recall signal).

Scientific question (from 2026-09-28 causal result):
    TREAT's blind-recall soft-KL gives net +0.5951pp, but source continuation
    costs -0.50pp, leaving only +0.092pp. The target gradient is 1-2 orders of
    magnitude SMALLER than the source gradient, so the source term dominates.

    Q: does REVERSING the ratio (source->0 or 0.1, target->1.0) recover the
       full +0.5951 signal?

Arms (all A05 init, blind-recall soft-KL, same seed/replay as the 09-28 pair):
    Arm A (pure target):  source_coeff=0.0,  target_coeff=1.0
    Arm B (light anchor): source_coeff=0.1,  target_coeff=1.0

Reference points already scored:
    CONTROL (S=1.0, T=0.0)  = 70.2720
    TREAT   (S=1.0, T=0.5)  = 70.8671
    A05                      = 70.775011

Blind-recall mask unchanged: (conf>=0.98) OR (blind_class AND conf>=0.50).
Teacher = A05 (same as TREAT), raw cache + Bstar calibration.
One endpoint: loop 2000. No offline gate. No packaging/submission.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# --- new server paths ---
ROOT = Path('/data/uav')
SRC = ROOT / 'code/src'
CONFIG = ROOT / 'code/config.yaml'
A05 = ROOT / 'checkpoints/A05_WEIGHT_alpha05.pt'
TEST2 = ROOT / 'datasets/test_2/images'
A05_CACHE = ROOT / 'teacher_cache/scores'
DATA = ROOT / 'datasets/训练集'
SPLIT = ROOT / 'code/splits/train.txt'
MODEL_SOURCE = ROOT / 'models/segformer-b3-ade512'

BSTAR = np.array([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], dtype=np.float32)
EXPECTED_A05_SHA = '023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134'
TRAIN_PREFIX = ('segformer.stages.2.', 'segformer.stages.3.')
HEAD_PREFIX = 'decode_head.'
MAX_STEPS = 2000
BASE_LR = 1e-5
HEAD_MULT = 0.05  # head_lr = 5e-7
WD = 0.01
RAMP = 500
CONF_NORMAL = 0.98
CONF_BLIND = 0.50
BLIND_CLASSES = (3, 4, 6)  # Water, Barren, Agricultural (train id 0-7)
SEED = 20260922


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def is_train(name):
    return name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX)


def strong_photo(x):
    mean = torch.tensor([.485, .456, .406], device=x.device, dtype=x.dtype)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=x.device, dtype=x.dtype)[None, :, None, None]
    x = (x * std + mean).clamp(0, 1)
    br = 1 + random.uniform(-.15, .15)
    ct = 1 + random.uniform(-.15, .15)
    sat = 1 + random.uniform(-.12, .12)
    gam = random.uniform(.9, 1.1)
    x = x * br
    mu = x.mean((-2, -1), keepdim=True)
    x = (x - mu) * ct + mu
    gray = x.mean(1, keepdim=True)
    x = (x - gray) * sat + gray
    x = x.clamp(0, 1).pow(gam)
    x = x + torch.randn_like(x) * random.uniform(0, .015)
    if random.random() < .25:
        x = F.avg_pool2d(x, 3, 1, 1)
    return (x.clamp(0, 1) - mean) / std


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


def verify_gpu1():
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise SystemExit('GPU1 required: CUDA_VISIBLE_DEVICES=1')
    import subprocess
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    m = [x for x in rows if x.strip().startswith('1,')]
    if not m:
        raise SystemExit(f'GPU1 not found: {rows}')
    import torch as _t
    if not _t.cuda.is_available() or _t.cuda.device_count() != 1:
        raise SystemExit('expected exactly one visible CUDA device')


def main():
    verify_gpu1()
    for p in (str(ROOT / 'code'), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ['UAVSEG_MODEL_SOURCE'] = str(MODEL_SOURCE)
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    from torch.utils.data import DataLoader
    from src.models import build_model as make_model
    from src.utils import load_config
    from src.transforms import normalize_to_tensor
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--source-coeff', type=float, required=True)
    ap.add_argument('--target-coeff', type=float, required=True)
    ap.add_argument('--max-steps', type=int, default=MAX_STEPS)
    a = ap.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    device = a.device

    if sha256_file(A05) != EXPECTED_A05_SHA:
        raise SystemExit('A05 SHA mismatch')

    cache_names = [f'test2_{i}.npy' for i in range(1, 1301)]
    missing = [n for n in cache_names if not (A05_CACHE / n).is_file()]
    if missing:
        raise SystemExit(f'A05 cache missing {len(missing)} files (first: {missing[:3]})')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    cfg['pretrained_model_path'] = str(MODEL_SOURCE)

    model = make_model(cfg)
    ckpt = torch.load(str(A05), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt), strict=True)
    model = model.to(device)

    model.eval()
    model.get_submodule('segformer.stages.2').train()
    model.get_submodule('segformer.stages.3').train()
    for n, p in model.named_parameters():
        p.requires_grad_(is_train(n))

    base_params = [p for n, p in model.named_parameters() if n.startswith(TRAIN_PREFIX)]
    head_params = [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([
        {'params': base_params, 'lr': BASE_LR},
        {'params': head_params, 'lr': BASE_LR * HEAD_MULT},
    ], weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    bstar = torch.from_numpy(BSTAR).to(device)[None, :, None, None]

    from src.dataset import UAVSegmentationDataset, segmentation_collate
    from src.rare_crop import RareClassCropper
    from src.transforms import TrainTransform
    ids = [x.strip() for x in SPLIT.read_text().split() if x.strip()]
    assert len(ids) == 5597, f'expected 5597 source ids, got {len(ids)}'
    crop = RareClassCropper(crop_size=1024, rare_probability=.5, priority_ids=(7, 4, 6), max_offset_fraction=.25)
    src = UAVSegmentationDataset(DATA, ids, TrainTransform(crop_size=1024, config=cfg['transforms'], rare_cropper=crop, ignore_index=255), maximum=None, seed=SEED, random_subset=False)
    sl = DataLoader(src, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=segmentation_collate, generator=torch.Generator().manual_seed(SEED + 10001), persistent_workers=True)

    class Target:
        def __init__(self):
            self.names = [f'test2_{i}.png' for i in range(1, 1301)]
        def __len__(self):
            return len(self.names)
        def __getitem__(self, i):
            n = self.names[i]
            with Image.open(TEST2 / n) as f:
                x = np.asarray(f.convert('RGB')).copy()
            return {'pixel_values': normalize_to_tensor(x), 'name': n}

    def collate(b):
        return {'pixel_values': torch.stack([x['pixel_values'] for x in b]), 'name': [x['name'] for x in b]}

    tl = DataLoader(Target(), batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate, generator=torch.Generator().manual_seed(SEED + 10002), persistent_workers=True)

    i_src = iter(sl)
    i_t = iter(tl)
    hist = []
    t0 = time.time()
    amp_skips = []

    for step in range(1, a.max_steps + 1):
        # source branch (skip if source_coeff == 0)
        if a.source_coeff > 0:
            try:
                s = next(i_src)
            except StopIteration:
                i_src = iter(sl)
                s = next(i_src)
            for gi, g in enumerate(opt.param_groups):
                g['lr'] = (BASE_LR if gi == 0 else BASE_LR * HEAD_MULT) * (1 - (step - 1) / 10000)
            opt.zero_grad(set_to_none=True)
            sx = s['pixel_values'].to(device)
            sy = s['labels'].to(device)
            with torch.amp.autocast('cuda', enabled=True):
                sz = model(pixel_values=sx).logits
            lsrc = F.cross_entropy(F.interpolate(sz.float(), size=sy.shape[-2:], mode='bilinear', align_corners=False), sy, ignore_index=255)
        else:
            for gi, g in enumerate(opt.param_groups):
                g['lr'] = (BASE_LR if gi == 0 else BASE_LR * HEAD_MULT) * (1 - (step - 1) / 10000)
            opt.zero_grad(set_to_none=True)
            lsrc = torch.zeros((), device=device)

        # target branch (always executes)
        try:
            t = next(i_t)
        except StopIteration:
            i_t = iter(tl)
            t = next(i_t)
        stem = Path(t['name'][0]).stem
        tx = t['pixel_values'].to(device)
        view = random.randrange(8)
        aug = strong_photo(d4_forward(tx, view))
        with torch.amp.autocast('cuda', enabled=True):
            uz = model(pixel_values=aug).logits
        z = d4_inverse(uz.float(), view) + bstar

        za = torch.from_numpy(np.load(A05_CACHE / f'{stem}.npy', allow_pickle=False).astype(np.float32)).to(device)
        if tuple(za.shape) != (8, 256, 256):
            raise SystemExit(f'A05 cache shape bad for {stem}: {tuple(za.shape)}')
        za = za[None] + bstar
        pa = za.softmax(1)
        conf, ya = pa.max(1)
        blind = (ya == 3) | (ya == 4) | (ya == 6)
        mask = (conf >= CONF_NORMAL) | (blind & (conf >= CONF_BLIND))
        logp_s = z.log_softmax(1)
        kl = (pa * (pa.log() - logp_s)).sum(1)
        lt = kl[mask].mean() if mask.any() else lsrc.new_zeros(())
        cov = float(mask.float().mean())
        cov_blind = float((mask & blind).float().mean()) if blind.any() else 0.0

        ramp = min(1.0, step / RAMP)
        sc = a.source_coeff
        tc = a.target_coeff * ramp
        loss = sc * lsrc + tc * lt

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        stepped = scaler.get_scale() >= scale_before
        if not stepped:
            amp_skips.append(step)

        row = {'step': step, 'source': float(lsrc), 'target_kl': float(lt),
               'source_coeff': sc, 'target_coeff': tc, 'loss': float(loss),
               'mask_cov': cov, 'mask_blind_cov': cov_blind,
               'stepped': stepped, 'elapsed': round(time.time() - t0, 1)}
        hist.append(row)
        if step % 50 == 0 or step == 1:
            print(json.dumps(row), flush=True)

        if step in (1000, 2000):
            path = out / f'checkpoint_step{step}.pt'
            torch.save({'model': model.state_dict(), 'step': step,
                        'a05_sha': EXPECTED_A05_SHA,
                        'target_recipe': 'soft_kl_blind_recall',
                        'source_coeff': sc, 'target_coeff': tc,
                        'conf_normal': CONF_NORMAL, 'conf_blind': CONF_BLIND,
                        'blind_classes': list(BLIND_CLASSES)}, path)
            print(f'Saved {path}', flush=True)

    summary = {'steps': a.max_steps, 'elapsed_sec': round(time.time() - t0, 1),
               'seed': SEED, 'a05_sha': EXPECTED_A05_SHA,
               'target_recipe': 'soft_kl_blind_recall',
               'source_coeff': a.source_coeff, 'target_coeff': a.target_coeff,
               'attempted_loops': a.max_steps,
               'successful_optimizer_updates': a.max_steps - len(amp_skips),
               'amp_skip_loops': amp_skips,
               'history_tail': hist[-10:]}
    (out / 'RUN_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
