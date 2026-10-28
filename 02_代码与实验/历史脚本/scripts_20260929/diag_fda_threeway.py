#!/usr/bin/env python3
"""Three-way controlled diagnostic: does test2 appearance induce class confusion?

On a sample of SOURCE images WITH GT, apply three treatments and measure A05's
per-class IoU change vs the untreated original:
  A  test2-style FDA : replace source low-freq amplitude with a random test2 image's
  B  self-FDA       : replace source low-freq amplitude with ANOTHER source image's
                       (isolates the effect of the FFT manipulation itself)
  C  brightness     : multiply by test2/src brightness ratio (~0.932), plain dimming

If only A significantly degrades Barren/Agricultural IoU -> test2-specific
appearance triggers confusion -> FDA training is worth a submission slot.
If A/B/C degrade similarly -> any appearance perturbation, not test2-specific.
If all ~unchanged -> model is insensitive to appearance -> FDA direction is wrong.
"""
import os, sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/uav')
SRC = ROOT / 'code/src'
CONFIG = ROOT / 'code/config.yaml'
MODEL_SOURCE = ROOT / 'models/segformer-b3-ade512'
A05 = ROOT / 'checkpoints/A05_WEIGHT_alpha05.pt'
IMG = ROOT / 'datasets/训练集/train/images'
MSK = ROOT / 'datasets/训练集/train/masks'
TEST2 = ROOT / 'datasets/test_2/images'
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
CLASS_NAMES = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation', 'Agricultural', 'Vehicle']
NUM_CLASSES = 8
BETA = 0.05
N_SAMPLE = 300


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


def official_to_train(mask):
    out = np.full(mask.shape, 255, dtype=np.int64)
    for c in range(1, 9):
        out[mask == c] = c - 1
    return out


def fda(src_rgb, tgt_rgb, beta=BETA):
    """Replace src's central low-frequency amplitude with tgt's, keep src phase."""
    h, w = src_rgb.shape[:2]
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * beta)
    Y, X = np.ogrid[:h, :w]
    mask = ((Y - cy) ** 2 + (X - cx) ** 2) <= radius ** 2
    out = np.zeros_like(src_rgb)
    for c in range(3):
        fs = np.fft.fftshift(np.fft.fft2(src_rgb[..., c]))
        ft = np.fft.fftshift(np.fft.fft2(tgt_rgb[..., c]))
        amp_s = np.abs(fs)
        pha_s = np.angle(fs)
        amp_t = np.abs(ft)
        amp_new = np.where(mask, amp_t, amp_s)
        fnew = amp_new * np.exp(1j * pha_s)
        out[..., c] = np.fft.ifft2(np.fft.ifftshift(fnew)).real
    return np.clip(out, 0, 255)


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
    model = mm(cfg)
    model.load_state_dict(load_state(A05), strict=True)
    model = model.cuda().eval()
    bias = torch.from_numpy(BSTAR).cuda()[None, :, None, None]

    # sample source stems + GT, and a random test2 image per source for FDA-A
    all_stems = sorted([f[:-4] for f in os.listdir(IMG) if f.endswith('.png')])
    random.seed(0)
    stems = random.sample(all_stems, N_SAMPLE)
    t2_list = sorted([f for f in os.listdir(TEST2) if f.endswith('.png')])
    tgt_pairs = [random.choice(t2_list) for _ in stems]
    src_pairs = [random.choice(all_stems) for _ in stems]  # another source for self-FDA

    # brightness ratio from diagnostic-1: test2 L=97.32, src L=104.39
    dim = 97.32 / 104.39

    def predict(rgb):
        px = normalize_to_tensor(rgb).unsqueeze(0).cuda()
        with torch.inference_mode():
            z = model(pixel_values=px).logits.float()
        dep = F.interpolate(z + bias, size=(1024, 1024), mode='bilinear', align_corners=False)[0]
        return dep.argmax(0).cpu().numpy()

    def iou(gt, pred):
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        valid = gt != 255
        g = gt[valid]; p = pred[valid]
        cm = np.bincount(g * NUM_CLASSES + p, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)
        cmf = cm.astype(np.float64)
        tp = np.diag(cmf); fp = cmf.sum(0) - tp; fn = cmf.sum(1) - tp
        den = tp + fp + fn
        iou_v = np.where(den > 0, tp / np.maximum(den, 1e-12), np.nan)
        return iou_v, cm

    acc = {k: {'iou': np.zeros(NUM_CLASSES), 'cm': np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)} for k in ['orig', 'FDA_test2', 'FDA_self', 'bright']}

    for i, stem in enumerate(stems, 1):
        rgb = np.asarray(Image.open(IMG / f'{stem}.png').convert('RGB'), dtype=np.float32)
        gt = official_to_train(np.asarray(Image.open(MSK / f'{stem}.png')))
        # treatments
        t2 = np.asarray(Image.open(TEST2 / tgt_pairs[i - 1]).convert('RGB'), dtype=np.float32)
        s2 = np.asarray(Image.open(IMG / f'{src_pairs[i - 1]}.png').convert('RGB'), dtype=np.float32)
        variants = {
            'orig': rgb,
            'FDA_test2': fda(rgb, t2),
            'FDA_self': fda(rgb, s2),
            'bright': np.clip(rgb * dim, 0, 255),
        }
        for k, v in variants.items():
            pred = predict(v.astype(np.uint8))
            _, cm = iou(gt, pred)
            acc[k]['cm'] += cm
        if i % 50 == 0:
            print(f'{i}/{N_SAMPLE}', flush=True)

    # global confusion matrix -> per-class IoU
    def final_iou(cm):
        cmf = cm.astype(np.float64)
        tp = np.diag(cmf); fp = cmf.sum(0) - tp; fn = cmf.sum(1) - tp
        den = tp + fp + fn
        iou = np.where(den > 0, tp / np.maximum(den, 1e-12), np.nan)
        miou = float(np.nanmean(iou[den > 0])) if (den > 0).any() else float('nan')
        return iou, miou

    ious = {}
    print('\n=== 各组 mIoU（全局混淆矩阵）===')
    for k in acc:
        iou, miou = final_iou(acc[k]['cm'])
        ious[k] = iou
        print(f'{k:>10}  mIoU={miou:.4f}')

    print('\n=== 逐类 IoU ===')
    for c in range(NUM_CLASSES):
        row = CLASS_NAMES[c].ljust(12)
        for k in acc:
            v = ious[k][c]
            row += f'{v:>12.4f}' if not np.isnan(v) else f'{">":>6}{"nan":>6}'
        print(row)

    print('\n=== 相对 orig 的逐类 IoU 变化（百分点，正=变好）===')
    for c in range(NUM_CLASSES):
        base = ious['orig'][c]
        if np.isnan(base):
            continue
        row = CLASS_NAMES[c].ljust(12)
        for k in ['FDA_test2', 'FDA_self', 'bright']:
            d = (ious[k][c] - base) * 100 if not np.isnan(ious[k][c]) else float('nan')
            row += f'{d:>+12.2f}'
        print(row)


if __name__ == '__main__':
    main()
