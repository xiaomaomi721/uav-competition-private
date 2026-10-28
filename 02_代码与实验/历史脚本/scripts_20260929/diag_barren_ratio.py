#!/usr/bin/env python3
"""Diagnostic 2: Barren prediction ratio, Arm B vs A05, on test2 (single-scale, fast).

Verifies the "double recall" hypothesis: Arm B already recalled Barren during
training (blind-recall mask), so with the SAME +3.6 Bstar bias at inference it
should output Barren on a larger fraction of test2 pixels than A05 does.
"""
import os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/uav')
SRC = ROOT / 'code/src'
CONFIG = ROOT / 'code/config.yaml'
MODEL_SOURCE = ROOT / 'models/segformer-b3-ade512'
A05 = ROOT / 'checkpoints/A05_WEIGHT_alpha05.pt'
ARMB = ROOT / 'runs/formal_armB/checkpoint_step2000.pt'
TEST2 = ROOT / 'datasets/test_2/images'
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)


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
    for p in (str(ROOT / 'code'), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ['UAVSEG_MODEL_SOURCE'] = str(MODEL_SOURCE)
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    from src.models import build_model as mm
    from src.utils import load_config
    from src.transforms import normalize_to_tensor
    from PIL import Image

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    cfg['pretrained_model_path'] = str(MODEL_SOURCE)

    names = sorted([f for f in os.listdir(TEST2) if f.endswith('.png')],
                   key=lambda x: int(x.split('_')[1].split('.')[0]))[:400]
    bias = torch.from_numpy(BSTAR).cuda()[None, :, None, None]

    results = {}
    for tag, ckpt in [('A05', A05), ('ArmB', ARMB)]:
        model = mm(cfg)
        model.load_state_dict(load_state(ckpt), strict=True)
        model = model.cuda().eval()
        per_class = np.zeros(8, dtype=np.int64)
        total = 0
        with torch.inference_mode():
            for n in names:
                rgb = np.asarray(Image.open(TEST2 / n).convert('RGB')).copy()
                px = normalize_to_tensor(rgb).unsqueeze(0).cuda()
                z = model(pixel_values=px).logits.float()
                dep = F.interpolate(z + bias, size=(1024, 1024), mode='bilinear', align_corners=False)[0]
                pred = dep.argmax(0).cpu().numpy()
                total += pred.size
                per_class += np.bincount(pred.ravel(), minlength=8)
        ratios = per_class / total
        print(f'{tag}: total={total}, 各类占比=', {i: round(float(ratios[i]), 4) for i in range(8)}, flush=True)
        print(f'{tag}: Barren(4)占比={round(float(ratios[4]), 4)}, Vehicle(7)占比={round(float(ratios[7]), 4)}', flush=True)
        results[tag] = {'barren': float(ratios[4]), 'vehicle': float(ratios[7])}

    print('=== 对比 ===')
    print(f"Barren: A05={results['A05']['barren']:.4f}  ArmB={results['ArmB']['barren']:.4f}  差值={results['ArmB']['barren']-results['A05']['barren']:.4f}")
    print(f"Vehicle: A05={results['A05']['vehicle']:.4f}  ArmB={results['ArmB']['vehicle']:.4f}")


if __name__ == '__main__':
    main()
