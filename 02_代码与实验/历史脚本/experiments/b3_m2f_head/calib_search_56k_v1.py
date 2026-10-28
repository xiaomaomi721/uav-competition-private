"""M2F Run1b 56k native semantic-score calibration search.

Search runs in M2F's OWN score space (probability mixture), never Stage23 logits.
No training, no weight change, no multi-checkpoint fusion, no test1 packaging.
Scores are cached in RAM only; the disk has ~2.7 GB free.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/home/jhy/rare_diag_20260915/m2f_gpu_20260915')
BUNDLE = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
CKPT = ROOT / 'formal_run1b/scores/score_iter56000.pt'
MORPH = Path('/home/jhy/rare_diag_20260915/_error_morphology_v2_gt.py')
NAMES = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation',
         'Agricultural', 'Vehicle']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--val-split', required=True)
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('physical GPU2 / logical cuda:0 only')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation
    for p in (str(ROOT / 'py'), str(BUNDLE)):
        sys.path.insert(0, p)
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores
    from src.transforms import normalize_to_tensor

    stems = [Path(l.strip()).stem for l in Path(args.val_split).read_text().split() if l.strip()]
    if len(stems) != 1399 or len(set(stems)) != 1399:
        raise RuntimeError('expected 1399 unique validation stems')
    img_dir = Path(args.dataset_root) / 'train/images'
    mask_dir = Path(args.dataset_root) / 'train/masks'

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    encoder = AutoModelForSemanticSegmentation.from_pretrained(
        '/home/jhy/models/segformer-b3-ade512', num_labels=8,
        ignore_mismatched_sizes=True, local_files_only=True).segformer.to(args.device)
    head = B3M2FStyleHead(in_channels=(64, 128, 320, 512), num_classes=8,
                          hidden_dim=256, num_queries=100, nhead=8, num_layers=6,
                          dim_feedforward=1024, ignore_index=255).to(args.device)
    encoder.load_state_dict(ck['model'], strict=True)
    head.load_state_dict(ck['head'], strict=True)
    encoder.eval(); head.eval()
    del ck

    # ---------- Stage A: cache native scores in RAM (fp16), GT as uint8 ----------
    print('STAGE_A start', flush=True)
    t0 = time.time()
    cache, gts = [], []
    with torch.inference_mode():
        for i, stem in enumerate(stems, 1):
            img = np.asarray(Image.open(img_dir / f'{stem}.png').convert('RGB')).copy()
            gt = np.asarray(Image.open(mask_dir / f'{stem}.png')).copy()
            px = normalize_to_tensor(img).unsqueeze(0).to(args.device)
            with torch.amp.autocast('cuda', enabled=False):
                o = encoder(pixel_values=px, output_hidden_states=True)
                out = head(o.hidden_states)
                sc = semantic_scores(out['class_logits'].float(), out['mask_logits'].float())
            sc = F.interpolate(sc, size=gt.shape[:2], mode='bilinear', align_corners=False)
            if not torch.isfinite(sc).all():
                raise RuntimeError(f'nonfinite scores at {stem}')
            cache.append(sc[0].to(torch.float16).cpu())
            gts.append(torch.from_numpy(gt.astype(np.int64)))
            del px, o, out, sc
            if i % 200 == 0:
                print(f'  A {i}/1399 {time.time()-t0:.0f}s', flush=True)
    print(f'STAGE_A done {time.time()-t0:.0f}s  images={len(cache)}', flush=True)

    # ---------- scoring helper ----------
    def confusion(vec):
        """vec: 8 offsets, index i == official class i+1. Returns 9x9 int64."""
        bias = torch.tensor(vec, dtype=torch.float32, device=args.device).view(8, 1, 1)
        cm = torch.zeros(9, 9, dtype=torch.int64, device=args.device)
        with torch.inference_mode():
            for sc16, gt in zip(cache, gts):
                s = sc16.to(args.device, dtype=torch.float32) + bias
                pred = s.argmax(0).add(1).to(torch.int64)
                g = gt.to(args.device)
                v = g != 0
                idx = g[v] * 9 + pred[v]
                cm += torch.bincount(idx, minlength=81).view(9, 9)
        return cm.cpu().numpy()

    def mious(cm):
        sub = cm[1:, 1:].astype(np.float64)
        inter = np.diag(sub)
        union = sub.sum(1) + sub.sum(0) - inter
        present = union > 0
        iou = np.where(present, inter / np.maximum(union, 1e-12), np.nan)
        return float(np.nanmean(iou[present])), iou, sub

    zero = [0.0] * 8
    cm0 = confusion(zero)
    base_miou, base_iou, base_sub = mious(cm0)
    print(f'ZERO_VAL_MIOU {base_miou:.10f}', flush=True)

    # ---------- Stage B: native score scale ----------
    print('STAGE_B start', flush=True)
    samp = torch.stack([c.to(torch.float32) for c in cache[:120]])       # 120x8x1024x1024
    margin = samp.max(0).values - samp.topk(2, dim=0).values[1]
    scale = {'per_class_score_quantiles': {}, 'top1_minus_top2_margin_quantiles': {}}
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    for c in range(8):
        v = samp[:, c].flatten()
        scale['per_class_score_quantiles'][NAMES[c]] = {
            f'q{q}': float(torch.quantile(v, q)) for q in qs}
    mv = margin.flatten()
    scale['top1_minus_top2_margin_quantiles'] = {f'q{q}': float(torch.quantile(mv, q)) for q in qs}
    del samp, margin
    print('SCALE ' + json.dumps(scale['top1_minus_top2_margin_quantiles']), flush=True)

    # offset grid per class: half-width = (q99 - q01)/2 of that class score
    half = {}
    for c in range(8):
        q = scale['per_class_score_quantiles'][NAMES[c]]
        half[c] = max((q['q0.99'] - q['q0.01']) / 2.0, 1e-6)

    # ---------- Stage C: 1D sensitivity scan (Background fixed at 0) ----------
    print('STAGE_C start', flush=True)
    sens = {}
    for c in range(1, 8):
        grid = np.linspace(-half[c], half[c], 9)
        rows = []
        for off in grid:
            vec = list(zero); vec[c] = float(off)
            cm = confusion(vec)
            m, iou, sub = mious(cm)
            rows.append({'offset': float(off), 'overall_miou': m,
                         'class_iou': float(iou[c]), 'all_class_iou': [float(x) for x in iou]})
        sens[NAMES[c]] = rows
        best = max(rows, key=lambda r: r['overall_miou'])
        print(f'  SENS {NAMES[c]:13s} half={half[c]:.5f} best_off={best["offset"]:+.5f} '
              f'miou={best["overall_miou"]:.6f} ({best["overall_miou"]-base_miou:+.6f})', flush=True)
    (outdir / 'stage_scale_and_sensitivity.json').write_text(
        json.dumps({'zero_miou': base_miou, 'zero_per_class_iou': [float(x) for x in base_iou],
                    'scale': scale, 'sensitivity': sens}, indent=2, ensure_ascii=False))

    # ---------- Stage D: coordinate descent ----------
    print('STAGE_D start', flush=True)
    vec = list(zero)
    cur = base_miou
    trace = [{'round': 0, 'class': None, 'vector': list(vec), 'miou': cur}]
    for rnd in range(1, 5):
        improved = False
        for c in range(1, 8):
            span = half[c]
            for _ in range(3):                      # 3-level refinement per class
                cand_best, cand_off = cur, vec[c]
                for off in np.linspace(vec[c] - span, vec[c] + span, 9):
                    v = list(vec); v[c] = float(off)
                    m, _, _ = mious(confusion(v))
                    if m > cand_best + 1e-9:
                        cand_best, cand_off = m, float(off)
                if cand_best > cur + 1e-9:
                    vec[c], cur = cand_off, cand_best
                    improved = True
                span /= 3.0
            trace.append({'round': rnd, 'class': NAMES[c], 'vector': list(vec), 'miou': cur})
        print(f'  CD round {rnd}: miou={cur:.6f} ({cur-base_miou:+.6f}) vec='
              f'{[round(x,4) for x in vec]}', flush=True)
        if not improved:
            break
    cd_vec, cd_miou = list(vec), cur

    # ---------- Stage E: local joint random search ----------
    print('STAGE_E start', flush=True)
    rng = np.random.default_rng(20260916)
    best_vec, best_miou = list(cd_vec), cd_miou
    trials = []
    for t in range(200):
        v = list(cd_vec)
        k = int(rng.integers(1, 4))
        for c in rng.choice(range(1, 8), size=k, replace=False):
            v[int(c)] += float(rng.normal(0, half[int(c)] / 6.0))
        m, _, _ = mious(confusion(v))
        trials.append((m, v))
        if m > best_miou:
            best_miou, best_vec = m, list(v)
    trials.sort(key=lambda x: -x[0])
    print(f'  RS best={best_miou:.6f} ({best_miou-base_miou:+.6f})', flush=True)

    # ---------- Stage F: FP32 re-verification of top candidates ----------
    print('STAGE_F start', flush=True)
    cands = [('zero', zero)] + [(f'cd', cd_vec)] + \
            [(f'rs_{i}', v) for i, (_, v) in enumerate(trials[:8])] + \
            [(f'sens_{NAMES[c]}', [0.0 if j != c else max(sens[NAMES[c]], key=lambda r: r['overall_miou'])['offset'] for j in range(8)]) for c in range(1, 8)]
    # dedupe by rounded vector
    seen, uniq = set(), []
    for tag, v in cands:
        key = tuple(round(x, 6) for x in v)
        if key in seen:
            continue
        seen.add(key); uniq.append((tag, list(v)))
    uniq.sort(key=lambda kv: -mious(confusion(kv[1]))[0])
    top = uniq[:10]

    fp32 = {}
    with torch.inference_mode():
        for tag, v in top:
            bias = torch.tensor(v, dtype=torch.float32, device=args.device).view(8, 1, 1)
            cm = torch.zeros(9, 9, dtype=torch.int64, device=args.device)
            for stem in stems:
                img = np.asarray(Image.open(img_dir / f'{stem}.png').convert('RGB')).copy()
                gt = np.asarray(Image.open(mask_dir / f'{stem}.png')).copy()
                px = normalize_to_tensor(img).unsqueeze(0).to(args.device)
                with torch.amp.autocast('cuda', enabled=False):
                    o = encoder(pixel_values=px, output_hidden_states=True)
                    out = head(o.hidden_states)
                    sc = semantic_scores(out['class_logits'].float(), out['mask_logits'].float())
                sc = F.interpolate(sc, size=gt.shape[:2], mode='bilinear', align_corners=False)
                pred = (sc[0] + bias).argmax(0).add(1).to(torch.int64)
                g = torch.from_numpy(gt.astype(np.int64)).to(args.device)
                vv = g != 0
                cm += torch.bincount((g[vv] * 9 + pred[vv]), minlength=81).view(9, 9)
                del px, o, out, sc, pred
            m, iou, sub = mious(cm.cpu().numpy())
            fp32[tag] = {'vector': v, 'miou_fp32': m, 'per_class_iou_fp32': [float(x) for x in iou]}
            print(f'  FP32 {tag:18s} miou={m:.8f} ({m-base_miou:+.8f})', flush=True)

    result = {
        'schema': 'M2F_56K_CALIBRATION_SEARCH_V1',
        'checkpoint': str(CKPT), 'checkpoint_sha256': sha256(CKPT),
        'search_script_sha256': sha256(Path(__file__)),
        'score_space': 'M2F native semantic mixture (softmax(class)[...,:8] x sigmoid(mask))',
        'cache': {'location': 'RAM only (fp16)', 'dtype': 'float16', 'disk_written': False},
        'protocol': 'single view, no TTA, no autocast, bilinear align_corners=False, argmax+1',
        'zero_val_miou': base_miou, 'zero_per_class_iou': [float(x) for x in base_iou],
        'scale': scale, 'sensitivity': sens, 'coordinate_descent_trace': trace,
        'cd_vector': cd_vec, 'cd_miou_fp16cache': cd_miou,
        'random_search_best_vector': best_vec, 'random_search_best_miou_fp16cache': best_miou,
        'fp32_reverification': fp32,
        'note': 'offsets are fixed per-class additive constants on the native score; '
                'Background held at 0 (8 entries, 7 effective degrees of freedom)',
    }
    (outdir / 'CALIB_SEARCH_RESULT.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print('SEARCH_DONE', flush=True)


if __name__ == '__main__':
    main()
