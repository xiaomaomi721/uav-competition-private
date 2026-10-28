#!/usr/bin/env python3
"""Evaluate UW1 checkpoint on H300 and R1099, compare to theta278 baseline."""
import json, sys, time
from pathlib import Path
import numpy as np
from _r2_contracts import BSTAR, CONFIG, DATA, OUT, SNAP, THETA278, load_state

UW1_CKPT = '/home/jhy/rare_diag_20260915/uw1_20260925/checkpoint_step500.pt'

def build(path, device):
    from src.models import build_model as make_model
    from src.utils import load_config
    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': 0.3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    obj = __import__('torch').load(path, map_location='cpu', weights_only=False)
    sd = obj['model'] if 'model' in obj else obj
    m = make_model(cfg); m.load_state_dict(sd, strict=True)
    return m.to(device).eval()

def cm_add(cm, p, g):
    v = g != 255
    cm += np.bincount(g[v].astype(np.int64)*8 + p[v], minlength=64).reshape(8,8)

def metric(cm):
    d = cm.sum(1) + cm.sum(0) - np.diag(cm)
    x = np.divide(np.diag(cm), d, out=np.zeros(8), where=d>0)
    return x, float(x.mean())

def eval_split(model, names, bstar, device):
    import torch, torch.nn.functional as F
    from PIL import Image
    from src.transforms import normalize_to_tensor
    cm = np.zeros((8,8), np.int64)
    with torch.inference_mode():
        for i, stem in enumerate(names, 1):
            with Image.open(DATA/f'train/images/{stem}.png') as f:
                im = np.asarray(f.convert('RGB')).copy()
            gt = np.load(f'/home/jhy/rare_diag_20260915/b2_val_20260917/gt/{stem}.npy', allow_pickle=False)
            px = normalize_to_tensor(im).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda', enabled=False):
                z = model(pixel_values=px).logits.float()
                z = F.interpolate(z, size=gt.shape, mode='bilinear', align_corners=False)
            p = (z + bstar).argmax(1)[0].cpu().numpy().astype(np.uint8)
            cm_add(cm, p, gt)
            if i % 100 == 0: print(f'  {i}/{len(names)}', flush=True)
    iou, miou = metric(cm)
    return iou, miou

def main():
    import torch
    device = 'cuda:0'
    for p in (str(SNAP), '/home/jhy/gpu2_day_queue'):
        if p not in sys.path: sys.path.insert(0, p)
    split = json.loads((OUT/'00_inventory/SOURCE_AUDIT_SPLIT.json').read_text())
    bstar = torch.from_numpy(BSTAR).to(device)[None,:,None,None]

    results = {}
    for tag, path in [('theta278', THETA278), ('UW1', UW1_CKPT)]:
        print(f'Evaluating {tag}...', flush=True)
        model = build(path, device)
        for split_name in ['H300', 'R1099']:
            names = split[split_name]
            iou, miou = eval_split(model, names, bstar, device)
            results[f'{tag}_{split_name}'] = {'miou': miou, 'per_class_iou': iou.tolist()}
            print(f'  {split_name} mIoU={miou:.6f}', flush=True)
        del model; torch.cuda.empty_cache()

    # Delta
    for sp in ['H300', 'R1099']:
        base = results[f'theta278_{sp}']['miou']
        uw1 = results[f'UW1_{sp}']['miou']
        print(f'{sp}: theta278={base:.6f} UW1={uw1:.6f} delta={uw1-base:+.6f}')
        base_iou = np.array(results[f'theta278_{sp}']['per_class_iou'])
        uw1_iou = np.array(results[f'UW1_{sp}']['per_class_iou'])
        for c in range(8):
            print(f'  class{c}: {base_iou[c]:.4f} -> {uw1_iou[c]:.4f} ({uw1_iou[c]-base_iou[c]:+.4f})')

    out = Path('/home/jhy/rare_diag_20260915/uw1_20260925/eval_results.json')
    out.write_text(json.dumps(results, indent=2))
    print(f'Saved to {out}')

if __name__ == '__main__':
    main()
