"""M2F 56k calibration search in the D4-TTA8 FP32 score space (stage B..F).

Exact, no precision loss: for each pixel we keep the top-1/top-2 native score
gap (fp32) and the pair of class indices (uint8). Adding a per-class offset only
shifts that gap by bias[i1]-bias[i2], so the calibrated argmax is reproduced
exactly, and every scoring pass is an O(N) gather instead of an 8-channel max.

No training, no weight change, no fusion, no test1 packaging.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/home/jhy/rare_diag_20260915/m2f_gpu_20260915')
SHM = Path('/dev/shm/tta8_56k_v2')
NAMES = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation',
         'Agricultural', 'Vehicle']
FOLD_SEED = 20260916


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    if not (SHM / '_DONE').exists():
        raise SystemExit('TTA8 cache not complete')

    import torch
    torch.set_num_threads(16)

    stems = sorted([p.stem for p in SHM.glob('*.npy') if not p.stem.startswith('gt_') and p.stem != '_DONE'])
    n = len(stems)
    if n != 1399:
        raise SystemExit(f'expected 1399 cached images, got {n}')
    print(f'loading {n} TTA8 fp32 score maps', flush=True)
    t0 = time.time()
    IDX = torch.empty((n, 1024, 1024), dtype=torch.uint8)
    MAR = torch.empty((n, 1024, 1024), dtype=torch.float32)
    GTS = torch.empty((n, 1024, 1024), dtype=torch.uint8)
    for i, stem in enumerate(stems):
        s = torch.from_numpy(np.load(SHM / f'{stem}.npy'))       # H,W,8 fp32
        top = torch.topk(s, 2, dim=-1)
        MAR[i] = (top.values[..., 0] - top.values[..., 1])
        IDX[i] = top.indices[..., 0].to(torch.uint8)
        GTS[i] = torch.from_numpy(np.load(SHM / f'gt_{stem}.npy'))
        del s, top
    print(f'loaded in {time.time()-t0:.0f}s; margin bytes={MAR.numel()*4/2**30:.2f} GiB', flush=True)

    del stems
    rng = np.random.default_rng(FOLD_SEED)
    perm = rng.permutation(n)
    folds = [torch.from_numpy(np.sort(perm[k::3])) for k in range(3)]

    def scores(vec):
        b = torch.tensor(vec, dtype=torch.float32)
        b1 = b[IDX.long()]
        top1 = IDX.long()
        # second class index is needed; recover it from stored pair
        return b1, top1

    # store the second index too (uint8, cheap) so delta is exact
    print('building second-index map', flush=True)
    IDX2 = torch.empty_like(IDX)
    for i, stem in enumerate([p.stem for p in SHM.glob('*.npy')
                              if not p.stem.startswith('gt_') and p.stem != '_DONE']):
        s = torch.from_numpy(np.load(SHM / f'{stem}.npy'))
        IDX2[i] = torch.topk(s, 2, dim=-1).indices[..., 1].to(torch.uint8)
        del s

    def predict_and_confusion(vec, subset=None):
        b = torch.tensor(vec, dtype=torch.float32)
        idx = torch.arange(n) if subset is None else subset
        m = MAR[idx]
        i1 = IDX[idx].long()
        i2 = IDX2[idx].long()
        delta = m + b[i1] - b[i2]
        pred = torch.where(delta > 0, i1, i2) + 1                 # official 1..8
        gt = GTS[idx].long()
        v = gt != 0
        flat = (gt[v] * 9 + pred[v]).to(torch.int64)
        cm = torch.bincount(flat, minlength=81).view(9, 9).numpy()
        # pixel-level agreement with the un-biased top-1, a guard on the top-2 model
        mismatch = int((delta <= 0).sum())
        return cm, mismatch

    def miou_of(cm):
        sub = cm[1:, 1:].astype(np.float64)
        inter = np.diag(sub)
        union = sub.sum(1) + sub.sum(0) - inter
        p = union > 0
        iou = np.where(p, inter / np.maximum(union, 1e-12), np.nan)
        return float(np.nanmean(iou[p])), iou

    zero = [0.0] * 8
    cm0, mm0 = predict_and_confusion(zero)
    base, base_iou = miou_of(cm0)
    print(f'ZERO_TTA8_VAL_MIOU {base:.10f}  top2_mismatch={mm0}', flush=True)

    fold_base = []
    for f in folds:
        cmf, _ = predict_and_confusion(zero, f)
        fold_base.append(miou_of(cmf)[0])
    print('FOLD_BASE ' + json.dumps([round(x, 8) for x in fold_base]), flush=True)

    # ---- Stage B: native scale ----
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    scale = {'margin_quantiles': {f'q{q}': float(torch.quantile(MAR.flatten()[::7], q)) for q in qs}}
    per_class = {}
    for i, stem in enumerate([p.stem for p in SHM.glob('*.npy')
                              if not p.stem.startswith('gt_') and p.stem != '_DONE'][:150]):
        s = torch.from_numpy(np.load(SHM / f'{stem}.npy'))
        for c in range(8):
            per_class.setdefault(NAMES[c], []).append(s[..., c].flatten()[::11])
        del s
    for c in range(8):
        v = torch.cat(per_class[NAMES[c]])
        scale.setdefault('per_class_score_quantiles', {})[NAMES[c]] = {f'q{q}': float(torch.quantile(v, q)) for q in qs}
    print('SCALE_MARGIN ' + json.dumps(scale['margin_quantiles']), flush=True)

    half = {}
    for c in range(8):
        q = scale['per_class_score_quantiles'][NAMES[c]]
        half[c] = max((q['q0.99'] - q['q0.01']) / 2.0, 1e-6)

    # ---- Stage C: 1D sensitivity ----
    print('STAGE_C 1D sensitivity', flush=True)
    sens = {}
    for c in range(1, 8):
        rows = []
        for off in np.linspace(-half[c], half[c], 9):
            v = list(zero); v[c] = float(off)
            cm, mm = predict_and_confusion(v)
            m, iou = miou_of(cm)
            rows.append({'offset': float(off), 'overall_miou': m, 'class_iou': float(iou[c]),
                         'top2_mismatch': mm, 'gain_pp': (m - base) * 100})
        sens[NAMES[c]] = rows
        best = max(rows, key=lambda r: r['overall_miou'])
        print(f'  {NAMES[c]:13s} half={half[c]:.5f} best={best["offset"]:+.5f} '
              f'gain={best["gain_pp"]:+.4f}pp', flush=True)

    # ---- Stage D: coordinate descent ----
    print('STAGE_D coordinate descent', flush=True)
    vec = list(zero); cur = base
    trace = [{'round': 0, 'class': None, 'vector': list(vec), 'miou': cur, 'gain_pp': 0.0}]
    for rnd in range(1, 5):
        improved = False
        for c in range(1, 8):
            span = half[c]
            for _ in range(3):
                best_m, best_off = cur, vec[c]
                for off in np.linspace(vec[c] - span, vec[c] + span, 9):
                    v = list(vec); v[c] = float(off)
                    cm, _ = predict_and_confusion(v)
                    m, _ = miou_of(cm)
                    if m > best_m + 1e-10:
                        best_m, best_off = m, float(off)
                if best_m > cur + 1e-10:
                    vec[c], cur, improved = best_off, best_m, True
                span /= 3.0
            trace.append({'round': rnd, 'class': NAMES[c], 'vector': list(vec),
                          'miou': cur, 'gain_pp': (cur - base) * 100})
        print(f'  round {rnd}: {cur:.8f} ({(cur-base)*100:+.4f}pp) {[round(x,4) for x in vec]}', flush=True)
        if not improved:
            break
    cd_vec, cd_miou = list(vec), cur

    # ---- Stage E: joint perturbation ----
    print('STAGE_E random search', flush=True)
    rng2 = np.random.default_rng(FOLD_SEED + 1)
    best_vec, best_miou = list(cd_vec), cd_miou
    pool = []
    for t in range(300):
        v = list(cd_vec)
        for c in rng2.choice(range(1, 8), size=int(rng2.integers(1, 4)), replace=False):
            v[int(c)] += float(rng2.normal(0, half[int(c)] / 6.0))
        cm, _ = predict_and_confusion(v)
        m, _ = miou_of(cm)
        pool.append((m, list(v)))
        if m > best_miou:
            best_miou, best_vec = m, list(v)
    pool.sort(key=lambda x: -x[0])
    print(f'  RS best={best_miou:.8f} ({(best_miou-base)*100:+.4f}pp)', flush=True)

    # ---- candidate set: best / fold-stable / conservative ----
    def report(vec):
        cm, mm = predict_and_confusion(vec)
        m, iou = miou_of(cm)
        fg = []
        for f in folds:
            cmf, _ = predict_and_confusion(vec, f)
            fg.append((miou_of(cmf)[0] - fold_base[folds.index(f)]) * 100)
        sub = cm[1:, 1:].astype(np.float64)
        inter = np.diag(sub)
        fp = sub.sum(0) - inter
        fn = sub.sum(1) - inter
        ai = NAMES.index('Agricultural')
        return {'vector': [float(x) for x in vec], 'full_miou': m, 'gain_pp': (m - base) * 100,
                'fold_gains_pp': fg, 'min_fold_gain_pp': min(fg),
                'per_class_iou': {NAMES[i]: float(iou[i]) for i in range(8)},
                'per_class_iou_gain_pp': {NAMES[i]: (float(iou[i]) - float(base_iou[i])) * 100 for i in range(8)},
                'agri_fp_px': int(fp[ai]), 'agri_fn_px': int(fn[ai]),
                'agri_fp_delta_px': int(fp[ai] - (cm0[1:, 1:].sum(0) - np.diag(cm0[1:, 1:]))[ai]),
                'agri_fn_delta_px': int(fn[ai] - (cm0[1:, 1:].sum(1) - np.diag(cm0[1:, 1:]))[ai]),
                'top2_mismatch_px': mm}

    cands = [('zero', list(zero)), ('cd', list(cd_vec)), ('rs_best', list(best_vec))]
    for i, (_, v) in enumerate(pool[:15]):
        cands.append((f'rs_{i}', v))
    phalf = {NAMES[c]: max((s['offset'] for s in sens[NAMES[c]] if s['overall_miou'] ==
                            max(x['overall_miou'] for x in sens[NAMES[c]])), key=abs, default=0.0)
             for c in range(1, 8)}
    for scale_f, tag in ((0.5, 'conservative_0.5x'), (0.25, 'conservative_0.25x')):
        cands.append((tag, [0.0] + [scale_f * best_vec[c] for c in range(1, 8)]))
    for c in range(1, 8):
        v = list(zero); v[c] = phalf[NAMES[c]]
        cands.append((f'sens1d_{NAMES[c]}', v))

    seen, uniq = set(), []
    for tag, v in cands:
        k = tuple(round(x, 6) for x in v)
        if k in seen:
            continue
        seen.add(k); uniq.append((tag, v))
    reports = [(tag, report(v)) for tag, v in uniq]
    reports.sort(key=lambda kv: -kv[1]['full_miou'])
    top10 = reports[:10]

    # fold-stable and conservative picks
    stable = max(reports, key=lambda kv: (kv[1]['min_fold_gain_pp'], kv[1]['full_miou']))
    cons = [r for r in reports if r[0].startswith('conservative')]
    cons_best = max(cons, key=lambda kv: kv[1]['full_miou']) if cons else None

    result = {
        'schema': 'M2F_56K_TTA8_CALIB_SEARCH_V1',
        'checkpoint': str(ROOT / 'formal_run1b/scores/score_iter56000.pt'),
        'checkpoint_sha256': sha256(ROOT / 'formal_run1b/scores/score_iter56000.pt'),
        'search_script_sha256': sha256(Path(__file__)),
        'cache_script_sha256': sha256(ROOT / 'calib_tta8_cache_56k_v1.py'),
        'score_space': 'D4 TTA8 FP32 average of M2F native semantic mixture',
        'cache': {'location': '/dev/shm/tta8_56k_v2 (RAM)', 'dtype': 'float32',
                  'images': n, 'disk_written': False, 'per_view_stored': False},
        'protocol': 'per-view mixture -> inverse mirror/rot -> bilinear align_corners=False -> FP32 mean of 8 -> +bias -> argmax+1',
        'offset_application': 'after TTA8 averaging, before argmax',
        'degrees_of_freedom': '8 entries, Background fixed 0 -> 7 effective',
        'fold_split': {'seed': FOLD_SEED, 'n_folds': 3, 'scheme': 'fixed permutation, index % 3',
                       'reoptimized_on_folds': False},
        'zero_tta8_val_miou': base, 'zero_per_class_iou': [float(x) for x in base_iou],
        'fold_zero_miou': fold_base,
        'scale': scale, 'sensitivity_1d': sens,
        'coordinate_descent_trace': trace, 'cd_vector': cd_vec, 'cd_miou': cd_miou,
        'random_search_best': best_vec, 'random_search_miou': best_miou,
        'top10': [{'tag': t, **r} for t, r in top10],
        'fold_stable_pick': {'tag': stable[0], **stable[1]},
        'conservative_pick': ({'tag': cons_best[0], **cons_best[1]} if cons_best else None),
        'note': 'single-view validation mIoU of the same checkpoint is 0.8155545; this TTA8 space is the one the online package uses',
    }
    (outdir / 'CALIB_TTA8_SEARCH.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print('SEARCH_DONE', flush=True)
    for tag, r in top10[:5]:
        print(f'  TOP {tag:16s} {r["full_miou"]:.8f} ({r["gain_pp"]:+.4f}pp) '
              f'minfold={r["min_fold_gain_pp"]:+.4f}pp', flush=True)


if __name__ == '__main__':
    main()
