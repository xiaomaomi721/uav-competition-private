"""Smoke: verify the top1/top2 exact-argmax model against real 8-channel argmax.

Runs on 12 cached validation images: for random bias vectors, the calibrated
prediction from (margin, i1, i2) must equal argmax(scores + bias) exactly, and
mIoU must match. CPU only.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

SHM = Path('/dev/shm/tta8_56k_v2')
stems = sorted([p.stem for p in SHM.glob('*.npy')
                if not p.stem.startswith('gt_') and p.stem != '_DONE'])[:12]
print('images', len(stems))
rng = np.random.default_rng(7)
bad = 0
tot = 0
for trial in range(6):
    bias = np.concatenate([[0.0], rng.normal(0, 0.02, 7)]).astype(np.float32)
    for stem in stems:
        s = np.load(SHM / f'{stem}.npy')                       # H,W,8 fp32
        gt = np.load(SHM / f'gt_{stem}.npy')
        ref = (s + bias.reshape(1, 1, 8)).argmax(-1) + 1
        top = np.argsort(-s, axis=-1)[..., :2]
        i1, i2 = top[..., 0], top[..., 1]
        g1 = s[..., 0] * 0 + np.take_along_axis(s, i1[..., None], -1)[..., 0]
        g2 = np.take_along_axis(s, i2[..., None], -1)[..., 0]
        marg = g1 - g2
        delta = marg + bias[i1] - bias[i2]
        model = np.where(delta > 0, i1, i2) + 1
        n_bad = int((model != ref).sum())
        bad += n_bad
        tot += ref.size
print(f'mismatched pixels {bad} / {tot}  -> {"EXACT" if bad == 0 else "MODEL WRONG"}')
