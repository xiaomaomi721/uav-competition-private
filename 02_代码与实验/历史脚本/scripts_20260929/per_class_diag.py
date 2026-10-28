#!/usr/bin/env python3
"""Per-class IoU + confidence-bucket diagnosis for Arm B on local val (1399 imgs).

Answers three questions (2026-09-29 体检):
  Q1 (decisive): Is Vehicle's missed detection "confident misclassification"
     (leaked pixels have conf > 0.98, misread as Vegetation) or "low-conf
     miss" (conf < 0.98, recoverable by adding Vehicle to blind-recall)?
     -> per-class conf histogram split into CORRECT (pred==gt) vs WRONG (pred!=gt).
  Q2: Is Barren's low precision caused by over-recall? -> conf histogram of
     pixels predicted Barren but GT != Barren (false positives).
  Q3: per-class IoU of Arm B vs known alpha050 baseline.

Protocol identical to scored Arm B: D4 TTA8, FP32 view mean, Bstar once,
bilinear 1024, softmax, argmax. GT is official 1-8 mapped to train 0-7 (255=ignore).

Runs on new server GPU1. Read-only on datasets; writes one JSON report.
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
A05_SHA = '023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134'
ARMB_SHA = 'f186e06ddd531724f647757587e5afec1415fb672410534de9658be60e3c58b9'
CKPTS = {'a05': (A05, A05_SHA, 'A05 (70.775011)'), 'armb': (ARMB, ARMB_SHA, 'ArmB (71.045315)')}
IMG_DIR = ROOT / 'datasets/训练集/train/images'
MSK_DIR = ROOT / 'datasets/训练集/train/masks'
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
CLASS_NAMES = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation', 'Agricultural', 'Vehicle']
NUM_CLASSES = 8
NBINS = 10
CONF_EDGES = np.linspace(0.0, 1.0, NBINS + 1)  # 10 bins


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('expected exactly one visible CUDA device')


def official_to_train(official_mask):
    out = np.full(official_mask.shape, 255, dtype=np.int64)
    for c in range(1, 9):
        out[official_mask == c] = c - 1
    return out


def main():
    verify_gpu1()
    for p in (str(ROOT / 'code'), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ['UAVSEG_MODEL_SOURCE'] = str(MODEL_SOURCE)
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    from src.models import build_model as mm
    from src.utils import load_config
    from src.transforms import normalize_to_tensor
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument('--split-json', required=True, help='path to SOURCE_AUDIT_SPLIT.json')
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--checkpoint', default='armb', choices=list(CKPTS.keys()))
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()

    ckpt_path, ckpt_sha, ckpt_desc = CKPTS[a.checkpoint]
    if sha256_file(ckpt_path) != ckpt_sha:
        raise SystemExit(f'{a.checkpoint} SHA mismatch')

    split = json.loads(Path(a.split_json).read_text(encoding='utf-8'))
    stems = split['H300'] + split['R1099']
    assert len(stems) == 1399, f'expected 1399 val stems, got {len(stems)}'

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    cfg['pretrained_model_path'] = str(MODEL_SOURCE)
    model = mm(cfg)
    model.load_state_dict(load_state(ckpt_path), strict=True)
    model = model.to(a.device).eval()
    bias = torch.from_numpy(BSTAR).to(a.device)[:, None, None]

    # accumulators
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    conf_correct = np.zeros((NUM_CLASSES, NBINS), dtype=np.int64)  # pred==gt
    conf_wrong = np.zeros((NUM_CLASSES, NBINS), dtype=np.int64)    # pred!=gt
    barren_fp = np.zeros((NBINS,), dtype=np.int64)  # pred=Barren(4), gt!=4
    valid_pixels = 0
    ignore_pixels = 0

    t0 = time.time()
    with torch.inference_mode():
        for idx, stem in enumerate(stems, 1):
            with Image.open(IMG_DIR / f'{stem}.png') as im:
                rgb = np.asarray(im.convert('RGB')).copy()
            gt_off = np.asarray(Image.open(MSK_DIR / f'{stem}.png'))
            gt = official_to_train(gt_off)

            px = normalize_to_tensor(rgb).unsqueeze(0).to(a.device)
            acc = None
            for r, hf in VIEWS:
                t = px
                if r: t = torch.rot90(t, r, (-2, -1))
                if hf: t = t.flip((-1,))
                logits = model(pixel_values=t).logits.float()
                if hf: logits = logits.flip((-1,))
                if r: logits = torch.rot90(logits, -r, (-2, -1))
                canon = logits[0].contiguous()
                acc = canon if acc is None else acc + canon
            raw = acc / 8.0
            dep = F.interpolate((raw + bias)[None], size=(1024, 1024), mode='bilinear', align_corners=False)[0]
            prob = dep.softmax(0)
            conf, pred = prob.max(0)  # [1024,1024]
            conf_np = conf.cpu().numpy()
            pred_np = pred.cpu().numpy().astype(np.int64)

            valid = gt != 255
            ignore_pixels += int((~valid).sum())
            if not valid.any():
                continue
            gv = gt[valid]
            pv = pred_np[valid]
            cv = conf_np[valid]
            valid_pixels += int(valid.sum())

            # confusion matrix
            cm += np.bincount(gv * NUM_CLASSES + pv, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

            # conf buckets per gt class, split correct/wrong
            for c in range(NUM_CLASSES):
                m = gv == c
                if not m.any():
                    continue
                correct = pv[m] == c
                hist_c = np.histogram(cv[m][correct], bins=CONF_EDGES)[0] if correct.any() else np.zeros(NBINS, dtype=np.int64)
                hist_w = np.histogram(cv[m][~correct], bins=CONF_EDGES)[0] if (~correct).any() else np.zeros(NBINS, dtype=np.int64)
                conf_correct[c] += hist_c
                conf_wrong[c] += hist_w

            # Barren false positives: pred==4, gt!=4
            mfp = (pv == 4) & (gv != 4)
            if mfp.any():
                barren_fp += np.histogram(cv[mfp], bins=CONF_EDGES)[0]

            if idx % 200 == 0:
                print(f'{idx}/1399 {time.time()-t0:.0f}s', flush=True)

    # per-class IoU
    cmf = cm.astype(np.float64)
    tp = np.diag(cmf)
    fp = cmf.sum(axis=0) - tp
    fn = cmf.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / np.maximum(denom, 1e-12), np.nan)
    prec = np.where(cmf.sum(axis=0) > 0, tp / np.maximum(cmf.sum(axis=0), 1e-12), np.nan)
    rec = np.where(cmf.sum(axis=1) > 0, tp / np.maximum(cmf.sum(axis=1), 1e-12), np.nan)
    present = denom > 0
    miou = float(np.nanmean(iou[present])) if present.any() else float('nan')

    report = {
        'model': ckpt_desc,
        'n_images': len(stems),
        'valid_pixels': valid_pixels,
        'ignore_pixels': ignore_pixels,
        'mIoU': miou,
        'per_class_iou': {CLASS_NAMES[i]: (float(iou[i]) if present[i] else None) for i in range(NUM_CLASSES)},
        'per_class_precision': {CLASS_NAMES[i]: (float(prec[i]) if present[i] else None) for i in range(NUM_CLASSES)},
        'per_class_recall': {CLASS_NAMES[i]: (float(rec[i]) if present[i] else None) for i in range(NUM_CLASSES)},
        'conf_bins_edges': CONF_EDGES.tolist(),
        'conf_correct_by_gt': {CLASS_NAMES[i]: conf_correct[i].tolist() for i in range(NUM_CLASSES)},
        'conf_wrong_by_gt': {CLASS_NAMES[i]: conf_wrong[i].tolist() for i in range(NUM_CLASSES)},
        'barren_false_positive_conf': barren_fp.tolist(),
        'runtime_sec': round(time.time() - t0, 1),
    }
    Path(a.output_json).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
