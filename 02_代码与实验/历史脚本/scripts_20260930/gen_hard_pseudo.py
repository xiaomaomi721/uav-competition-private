#!/usr/bin/env python3
"""Generate hard pseudo-labels from Arm B teacher cache (pseudo500-style).

Arm B raw logits [8,256,256] + Bstar -> bilinear 1024 -> softmax -> conf/argmax
-> blind-recall mask -> hard label (argmax+1, official 1-8). Mask-out pixels = 0
(official ignore). Output 1024x1024 PNG, official 1-8, aligned to source GT size.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/data/uav')
CACHE_T1 = ROOT / 'teacher_cache/armb_test1/scores'
CACHE_T2 = ROOT / 'teacher_cache/armb_scores/scores'
OUT_T1 = ROOT / 'teacher_cache/armb_hard_pseudo/test1'
OUT_T2 = ROOT / 'teacher_cache/armb_hard_pseudo/test2'
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
CONF_NORMAL = 0.98
CONF_BLIND = 0.50
BLIND = (3, 4, 6)  # Water, Barren, Agricultural (train id 0-7)


def verify_gpu(gpu_id):
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu_id):
        raise SystemExit(f'GPU{gpu_id} required: CUDA_VISIBLE_DEVICES={gpu_id}')
    import subprocess
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    m = [x for x in rows if x.strip().startswith(f'{gpu_id},')]
    if not m:
        raise SystemExit(f'GPU{gpu_id} not found: {rows}')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('expected exactly one visible CUDA device')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    a = ap.parse_args()
    verify_gpu(a.gpu)
    bias = torch.from_numpy(BSTAR).cuda()[None, :, None, None]

    for cache_dir, out_dir, prefix, count in [
        (CACHE_T1, OUT_T1, 'test1', 500),
        (CACHE_T2, OUT_T2, 'test2', 1300),
    ]:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, count + 1):
            stem = f'{prefix}_{i}'
            za = torch.from_numpy(np.load(cache_dir / f'{stem}.npy').astype(np.float32)).cuda()
            if tuple(za.shape) != (8, 256, 256):
                raise SystemExit(f'{stem}: bad cache shape {tuple(za.shape)}')
            dep = F.interpolate((za[None] + bias), size=(1024, 1024), mode='bilinear', align_corners=False)[0]
            prob = dep.softmax(0)
            conf, ya = prob.max(0)
            blind = (ya == 3) | (ya == 4) | (ya == 6)
            mask = (conf >= CONF_NORMAL) | (blind & (conf >= CONF_BLIND))
            label = torch.where(mask, ya + 1, torch.zeros_like(ya)).byte().cpu().numpy()
            Image.fromarray(label, mode='L').save(out_dir / f'{stem}.png')
            if i % 200 == 0:
                print(f'{prefix} {i}/{count}', flush=True)

    print('DONE')


if __name__ == '__main__':
    main()
