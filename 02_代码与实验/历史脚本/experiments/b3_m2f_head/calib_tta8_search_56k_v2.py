"""M2F 56k calibration search on FULL 8-channel D4-TTA8 FP32 scores.

Every candidate is scored by the exact rule

    pred = argmax_c ( S_bar_c(x) + bias_c )        (official class = c + 1)

with the complete 8-channel score map. No top-k reduction, no approximation.

Exact acceleration (single-class moves only): when only bias[c] varies and the
other seven offsets are held fixed, the winner among the other seven classes is
a per-pixel constant, so

    pred = c  if  S_c + bias_c > max_{j!=c}(S_j + bias_j)   else  argmax_{j!=c}

is exactly equivalent to the full argmax for every value of bias_c, and the RHS
is precomputed once per class per phase.

Offsets are applied AFTER TTA8 averaging and BEFORE argmax, matching the test1
packaging protocol. Background is held at 0 -> 7 effective degrees of freedom.
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
TIME_BUDGET_S = 1200


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
    if outdir.exists():
        raise SystemExit(f'refuse existing {outdir}')
    outdir.mkdir(parents=True)
    if not (SHM / '_DONE').exists():
        raise SystemExit('TTA8 cache incomplete')

    import torch
    torch.set_num_threads(64)

    stems = sorted(p.stem for p in SHM.glob('*.npy')
                   if not p.stem.startswith('gt_') and p.stem != '_DONE')
    n = len(stems)
    if n != 1399:
        raise SystemExit(f'expected 1399, got {n}')
    print(f'loading {n} full 8-channel TTA8 fp32 maps', flush=True)
    t0 = time.time()
    S = [torch.from_numpy(np.load(SHM / f'{s}.npy')).permute(2, 0, 1).contiguous()
         for s in stems]                                     # each (8,1024,1024) fp32
    GT = [torch.from_numpy(np.load(SHM / f'gt_{s}.npy')).long() for s in stems]
    print(f'loaded {time.time()-t0:.0f}s  approx {sum(x.numel()*4 for x in S)/2**30:.1f} GiB', flush=True)

    perm = np.random.default_rng(FOLD_SEED).permutation(n)
    folds = [np.sort(perm[k::3]) for k in range(3)]

    def conf_of(preds, subset=None):
        idx = range(n) if subset is None else subset
        cm = np.zeros((9, 9), dtype=np.int64)
        for i in idx:
            g = GT[i]; p = preds[i]
            v = g != 0
            cm += np.bincount((g[v] * 9 + p[v]).numpy(), minlength=81).reshape(9, 9)
        return cm

    def miou_of(cm):
        sub = cm[1:, 1:].astype(np.float64)
        inter = np.diag(sub)
        union = sub.sum(1) + sub.sum(0) - inter
        p = union > 0
        iou = np.where(p, inter / np.maximum(union, 1e-12), np.nan)
        return float(np.nanmean(iou[p])), iou

    def full_predict(vec, subset=None):
        """Exact argmax over all 8 channels with offsets."""
        b = torch.tensor(vec, dtype=torch.float32).view(8, 1, 1)
        idx = range(n) if subset is None else subset
        out = {}
        for i in idx:
            out[i] = (S[i] + b).argmax(0).to(torch.uint8) + 1
        return out

    t1 = time.time()
    pred_zero = full_predict([0.0] * 8)
    t_pass = time.time() - t1
    cm0 = conf_of(pred_zero)
    base, base_iou = miou_of(cm0)
    fold_base = [miou_of(conf_of(pred_zero, f))[0] for f in folds]
    print(f'ZERO_TTA8_VAL_MIOU {base:.10f}   (one full pass = {t_pass:.1f}s)', flush=True)
    print(f'ZERO_TTA8_PER_CLASS ' + json.dumps({NAMES[i]: round(float(base_iou[i]), 6) for i in range(8)}), flush=True)
    print(f'ZERO_TTA8_FOLDS {json.dumps([round(x,8) for x in fold_base])}', flush=True)

    # ---- margin / score scale on the TTA8 average ----
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    samp = torch.stack(S[:150])
    top2 = torch.topk(samp, 2, dim=1).values
    marg = (top2[:, 0] - top2[:, 1]).flatten()
    scale = {'margin_quantiles': {f'q{q}': float(torch.quantile(marg[::11], q)) for q in qs},
             'per_class_score_quantiles': {}}
    for c in range(8):
        v = samp[:, c].flatten()
        scale['per_class_score_quantiles'][NAMES[c]] = {f'q{q}': float(torch.quantile(v[::11], q)) for q in qs}
    del samp, top2, marg
    print('SCALE_MARGIN ' + json.dumps(scale['margin_quantiles']), flush=True)
    half = {c: max((scale['per_class_score_quantiles'][NAMES[c]]['q0.99'] -
                    scale['per_class_score_quantiles'][NAMES[c]]['q0.01']) / 2.0, 1e-6) for c in range(8)}

    # ---- exact single-class machinery ----
    def other_side(c):
        """Precompute argmax and value over the seven classes != c (bias currently zero
        absorbed into S when called with the current vector applied)."""
        vals = torch.stack([S[i] for i in range(n)]) if False else None
        return None

    def scan_class(vec, c, offsets):
        """Exact scan of bias[c] over `offsets` with all other offsets fixed at vec."""
        b = torch.tensor(vec, dtype=torch.float32).view(8, 1, 1)
        keep = [j for j in range(8) if j != c]
        # per-pixel best among the other seven
        best_v = None
        best_a = None
        for i in range(n):
            sub = S[i][keep] + b[keep]
            v, a = sub.max(0)
            best_v = v if best_v is None else best_v
            if best_a is None:
                best_a = torch.empty((n, 1024, 1024), dtype=torch.uint8)
            best_v = v if i == 0 else best_v
            if i == 0:
                other_v = torch.empty((n, 1024, 1024), dtype=torch.float32)
                other_a = torch.empty((n, 1024, 1024), dtype=torch.uint8)
            other_v[i] = v
            other_a[i] = torch.tensor(keep, dtype=torch.uint8)[a]
        del best_v, best_a
        results = []
        for off in offsets:
            cm = np.zeros((9, 9), dtype=np.int64)
            v = float(off)
            for i in range(n):
                sc = S[i][c] + v
                win = sc > other_v[i]
                pred = torch.where(win, torch.full_like(other_a[i], c + 1), other_a[i] + 1)
                g = GT[i]; m = g != 0
                cm += np.bincount((g[m] * 9 + pred[m]).numpy(), minlength=81).reshape(9, 9)
            m_, iou = miou_of(cm)
            results.append({'offset': v, 'miou': m_, 'class_iou': float(iou[c]),
                            'gain_pp': (m_ - base) * 100})
        return results

    def scan_class_exact(vec, c, offsets):
        """Exact scan verified against full argmax (used for the sensitivity report)."""
        rows = []
        for off in offsets:
            v = list(vec); v[c] = float(off)
            cm = conf_of(full_predict(v))
            m_, iou = miou_of(cm)
            rows.append({'offset': float(off), 'miou': m_, 'class_iou': float(iou[c]),
                         'gain_pp': (m_ - base) * 100})
        return rows

    # ---- Stage C: 1D sensitivity (exact full-channel scoring) ----
    print('STAGE_C 1D sensitivity (exact 8-channel)', flush=True)
    sens = {}
    for c in range(1, 8):
        rows = scan_class_exact([0.0] * 8, c, np.linspace(-half[c], half[c], 9))
        sens[NAMES[c]] = rows
        best = max(rows, key=lambda r: r['miou'])
        print(f'  {NAMES[c]:13s} half={half[c]:.5f} best={best["offset"]:+.5f} gain={best["gain_pp"]:+.4f}pp', flush=True)
    (outdir / 'stage_scale_sensitivity.json').write_text(
        json.dumps({'zero_tta8_miou': base, 'zero_tta8_per_class_iou': [float(x) for x in base_iou],
                    'zero_tta8_folds': fold_base, 'scale': scale, 'sensitivity_1d': sens},
                   indent=2, ensure_ascii=False))

    # ---- Stage D: coordinate descent (exact, via fast single-class path) ----
    print('STAGE_D coordinate descent', flush=True)
    vec = [0.0] * 8
    cur = base
    trace = [{'round': 0, 'class': None, 'vector': list(vec), 'miou': cur, 'gain_pp': 0.0}]
    for rnd in range(1, 5):
        improved = False
        for c in range(1, 8):
            span = half[c]
            for _ in range(3):
                cands = np.linspace(vec[c] - span, vec[c] + span, 9)
                rows = scan_class_exact(vec, c, cands)
                best = max(rows, key=lambda r: r['miou'])
                if best['miou'] > cur + 1e-10:
                    vec[c], cur, improved = best['offset'], best['miou'], True
                span /= 3.0
            trace.append({'round': rnd, 'class': NAMES[c], 'vector': list(vec),
                          'miou': cur, 'gain_pp': (cur - base) * 100})
        print(f'  round {rnd}: {cur:.8f} ({(cur-base)*100:+.4f}pp) {[round(x,4) for x in vec]}', flush=True)
        if not improved:
            break
    cd_vec, cd_miou = list(vec), cur

    # ---- Stage E: joint random search, exact full-channel, time-budgeted ----
    print('STAGE_E joint random search (exact 8-channel)', flush=True)
    rng = np.random.default_rng(FOLD_SEED + 1)
    budget = TIME_BUDGET_S
    n_rand = int(max(40, min(400, budget / max(t_pass, 0.5))))
    print(f'  full pass {t_pass:.1f}s -> {n_rand} exact joint candidates', flush=True)
    pool = []
    best_vec, best_miou = list(cd_vec), cd_miou
    for t in range(n_rand):
        v = list(cd_vec)
        for c in rng.choice(range(1, 8), size=int(rng.integers(1, 4)), replace=False):
            v[int(c)] += float(rng.normal(0, half[int(c)] / 6.0))
        cm = conf_of(full_predict(v))
        m_, _ = miou_of(cm)
        pool.append((m_, list(v)))
        if m_ > best_miou:
            best_miou, best_vec = m_, list(v)
        if t % 20 == 0:
            print(f'  RS {t}/{n_rand} best={best_miou:.8f}', flush=True)
    pool.sort(key=lambda x: -x[0])
    print(f'  RS final best={best_miou:.8f} ({(best_miou-base)*100:+.4f}pp)', flush=True)

    # ---- candidate report ----
    def report(tag, v):
        p = full_predict(v)
        cm = conf_of(p)
        m_, iou = miou_of(cm)
        fg = [(miou_of(conf_of(p, f))[0] - fold_base[k]) * 100 for k, f in enumerate(folds)]
        sub = cm[1:, 1:].astype(np.float64)
        inter = np.diag(sub)
        fp = sub.sum(0) - inter
        fn = sub.sum(1) - inter
        sub0 = cm0[1:, 1:].astype(np.float64)
        i0 = np.diag(sub0)
        ai = NAMES.index('Agricultural')
        return {'tag': tag, 'vector': [float(x) for x in v], 'full_miou': m_,
                'gain_pp': (m_ - base) * 100, 'fold_gains_pp': fg, 'min_fold_gain_pp': min(fg),
                'per_class_iou': {NAMES[i]: float(iou[i]) for i in range(8)},
                'per_class_iou_gain_pp': {NAMES[i]: (float(iou[i]) - float(base_iou[i])) * 100 for i in range(8)},
                'agri_fp_px': int(fp[ai]), 'agri_fn_px': int(fn[ai]),
                'agri_fp_delta_px': int(fp[ai] - (sub0.sum(0) - i0)[ai]),
                'agri_fn_delta_px': int(fn[ai] - (sub0.sum(1) - i0)[ai])}

    cands = [('zero', [0.0] * 8), ('cd', cd_vec), ('rs_best', best_vec)]
    cands += [(f'rs_{i}', v) for i, (_, v) in enumerate(pool[:15])]
    for f_, tag in ((0.5, 'conservative_0.5x'), (0.25, 'conservative_0.25x'), (0.75, 'conservative_0.75x')):
        cands.append((tag, [0.0] + [f_ * best_vec[c] for c in range(1, 8)]))
    for c in range(1, 8):
        best_off = max(sens[NAMES[c]], key=lambda r: r['miou'])['offset']
        v = [0.0] * 8; v[c] = best_off
        cands.append((f'sens1d_{NAMES[c]}', v))

    seen, uniq = set(), []
    for tag, v in cands:
        k = tuple(round(x, 6) for x in v)
        if k in seen:
            continue
        seen.add(k); uniq.append((tag, v))
    reports = [report(t, v) for t, v in uniq]
    reports.sort(key=lambda r: -r['full_miou'])
    top10 = reports[:10]
    stable = max(reports, key=lambda r: (r['min_fold_gain_pp'], r['full_miou']))
    cons = [r for r in reports if r['tag'].startswith('conservative')]
    cons_best = max(cons, key=lambda r: r['full_miou']) if cons else None

    result = {
        'schema': 'M2F_56K_TTA8_CALIB_SEARCH_V2_EXACT',
        'checkpoint': str(ROOT / 'formal_run1b/scores/score_iter56000.pt'),
        'checkpoint_sha256': sha256(ROOT / 'formal_run1b/scores/score_iter56000.pt'),
        'search_script_sha256': sha256(Path(__file__)),
        'cache_script_sha256': sha256(ROOT / 'calib_tta8_cache_56k_v1.py'),
        'protocol_code_sha256': {
            '/home/jhy/gpu2_day_queue/tta_anatomy_v2.py': sha256('/home/jhy/gpu2_day_queue/tta_anatomy_v2.py'),
            str(ROOT / 'py/b3_m2f_head/semantic.py'): sha256(ROOT / 'py/b3_m2f_head/semantic.py'),
            str(ROOT / 'py/b3_m2f_head/head.py'): sha256(ROOT / 'py/b3_m2f_head/head.py'),
            '/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/src/transforms.py':
                sha256('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/src/transforms.py'),
        },
        'scoring_rule': 'pred = argmax_c(S_bar_c + bias_c), FULL 8-channel, no top-k reduction',
        'cache': {'location': str(SHM) + ' (RAM)', 'dtype': 'float32', 'channels': 8,
                  'images': n, 'disk_written': False},
        'offset_application': 'after TTA8 FP32 averaging, before argmax',
        'degrees_of_freedom': '8 entries, Background fixed 0 -> 7 effective',
        'fold_split': {'seed': FOLD_SEED, 'n_folds': 3, 'reoptimized_on_folds': False},
        'zero_tta8_miou': base, 'zero_tta8_per_class_iou': [float(x) for x in base_iou],
        'zero_tta8_folds': fold_base,
        'single_view_reference_miou': 0.8155545489734892,
        'scale': scale, 'sensitivity_1d': sens, 'coordinate_descent_trace': trace,
        'cd_vector': cd_vec, 'cd_miou': cd_miou,
        'random_search': {'n_candidates': n_rand, 'full_pass_seconds': t_pass,
                          'best_vector': best_vec, 'best_miou': best_miou},
        'top10': top10,
        'fold_stable_pick': stable,
        'conservative_pick': cons_best,
        'note': 'all calib gains are relative to TTA8 ZERO, not single-view 0.8155545',
    }
    (outdir / 'CALIB_TTA8_SEARCH.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print('SEARCH_DONE', flush=True)
    print(f'  CD  {cd_miou:.8f} ({(cd_miou-base)*100:+.4f}pp)', flush=True)
    print(f'  RS  {best_miou:.8f} ({(best_miou-base)*100:+.4f}pp)', flush=True)
    for r in top10[:5]:
        print(f'  TOP {r["tag"]:18s} {r["full_miou"]:.8f} ({r["gain_pp"]:+.4f}pp) '
              f'minfold={r["min_fold_gain_pp"]:+.4f}pp', flush=True)
    print(f'  FOLDSTABLE {stable["tag"]} minfold={stable["min_fold_gain_pp"]:+.4f}pp', flush=True)


if __name__ == '__main__':
    main()
