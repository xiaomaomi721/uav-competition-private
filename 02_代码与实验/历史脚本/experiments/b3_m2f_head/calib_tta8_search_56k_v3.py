"""M2F 56k calibration search on FULL 8-channel D4-TTA8 FP32 scores. EXACT.

Scoring rule for every reported candidate:
    pred = argmax_c ( S_bar_c(x) + bias_c ),  full 8 channels, official class = c+1.

Exact single-class acceleration:
    When only bias[c] varies while the other seven offsets are fixed, the winner
    among those seven is a per-pixel constant (other_v, other_a). Then for any
    value of bias[c],
        pred = c  if  S_c + bias_c > other_v   else  argmax_{j!=c}
    which is mathematically identical to the full 8-channel argmax. other_a may
    point at ANY of the seven classes, so a class outside the original top-2 can
    still win -- this is exactly what a top-2 cache would have missed.

Background is held at 0 -> 7 effective degrees of freedom.
Offsets are applied after TTA8 FP32 averaging and before argmax.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
    t0 = time.time()
    S = [torch.from_numpy(np.load(SHM / f'{s}.npy')) for s in stems]      # (H,W,8) fp32
    GT = [torch.from_numpy(np.load(SHM / f'gt_{s}.npy')).long().reshape(-1) for s in stems]
    print(f'loaded {n} maps in {time.time()-t0:.0f}s', flush=True)
    H, W = S[0].shape[:2]

    perm = np.random.default_rng(FOLD_SEED).permutation(n)
    folds = [np.sort(perm[k::3]) for k in range(3)]
    fold_of = np.empty(n, dtype=np.int8)
    for k, f in enumerate(folds):
        fold_of[f] = k

    def conf_and_miou(pred_list, idx):
        """pred_list[i] = official ids 1..8 (H*W). Returns cm, miou, per-class iou."""
        tot = np.zeros((9, 9), dtype=np.int64)
        for i in idx:
            g = GT[i]
            m = g != 0
            tot += np.bincount((g[m] * 9 + pred_list[i][m]).numpy(), minlength=81).reshape(9, 9)
        sub = tot[1:, 1:].astype(np.float64)
        inter = np.diag(sub)
        union = sub.sum(1) + sub.sum(0) - inter
        p = union > 0
        iou = np.where(p, inter / np.maximum(union, 1e-12), np.nan)
        return tot, float(np.nanmean(iou[p])), iou

    def full_pred(vec):
        """EXACT full 8-channel argmax with offsets."""
        b = torch.tensor(vec, dtype=torch.float32).view(1, 1, 8)
        return [(t + b).argmax(-1).reshape(-1).to(torch.int64) + 1 for t in S]

    all_idx = list(range(n))
    t1 = time.time()
    p_zero = full_pred([0.0] * 8)
    t_pass = time.time() - t1
    cm0, base, base_iou = conf_and_miou(p_zero, all_idx)
    fold_base = [conf_and_miou(p_zero, fold_of == k)[1] for k in range(3)]
    print(f'ZERO_TTA8_VAL_MIOU {base:.10f}  (full pass {t_pass:.1f}s)', flush=True)
    print('ZERO_TTA8_PER_CLASS ' + json.dumps({NAMES[i]: round(float(base_iou[i]), 6) for i in range(8)}), flush=True)
    print('ZERO_TTA8_FOLDS ' + json.dumps([round(x, 8) for x in fold_base]), flush=True)

    # ---- scale from the TTA8 average ----
    qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    samp = torch.stack([t.reshape(-1, 8) for t in S[:150]])
    tv = torch.topk(samp, 2, dim=1).values
    marg = (tv[:, 0] - tv[:, 1])
    scale = {'margin_quantiles': {f'q{q}': float(torch.quantile(marg, q)) for q in qs},
             'per_class_score_quantiles': {}}
    for c in range(8):
        scale['per_class_score_quantiles'][NAMES[c]] = {
            f'q{q}': float(torch.quantile(samp[:, c], q)) for q in qs}
    del samp, tv, marg
    print('SCALE_MARGIN ' + json.dumps(scale['margin_quantiles']), flush=True)
    half = {c: max((scale['per_class_score_quantiles'][NAMES[c]]['q0.99'] -
                    scale['per_class_score_quantiles'][NAMES[c]]['q0.01']) / 2.0, 1e-6)
            for c in range(8)}

    def other_side(vec, c):
        """Per-pixel max and argmax over the seven classes != c (exact)."""
        keep = [j for j in range(8) if j != c]
        b = torch.tensor([vec[j] for j in keep], dtype=torch.float32).view(1, 1, 7)
        ov = torch.empty(n, H * W, dtype=torch.float32)
        oa = torch.empty(n, H * W, dtype=torch.uint8)
        kt = torch.tensor(keep, dtype=torch.int64)
        for i, t in enumerate(S):
            r = t.reshape(-1, 8)[:, keep] + b
            v, a = r.max(1)
            ov[i] = v
            oa[i] = kt[a]
            del r, v, a
        return ov, oa

    def scan_class(vec, c, offsets):
        """EXACT scan of bias[c]; other offsets fixed at vec."""
        ov, oa = other_side(vec, c)
        rows = []
        for off in offsets:
            preds = []
            for i, t in enumerate(S):
                sc = t.reshape(-1, 8)[:, c] + float(off)
                win = sc > ov[i]
                preds.append(torch.where(win, torch.full_like(oa[i], c + 1, dtype=torch.int64),
                                         oa[i].to(torch.int64) + 1))
            _, m, iou = conf_and_miou(preds, all_idx)
            rows.append({'offset': float(off), 'miou': m, 'class_iou': float(iou[c]),
                         'gain_pp': (m - base) * 100})
            del preds
        del ov, oa
        return rows

    # ---- self-check: fast path == full argmax on random vectors ----
    print('SELFCHECK fast single-class path vs full argmax', flush=True)
    rng = np.random.default_rng(11)
    ok = True
    for t in range(3):
        vec = [0.0] + list(rng.normal(0, 0.05, 7))
        c = int(rng.integers(1, 8))
        off = float(rng.normal(0, 0.05))
        ov, oa = other_side(vec, c)
        preds = []
        for i, tt in enumerate(S):
            sc = tt.reshape(-1, 8)[:, c] + off
            win = sc > ov[i]
            preds.append(torch.where(win, torch.full_like(oa[i], c + 1, dtype=torch.int64),
                                     oa[i].to(torch.int64) + 1))
        v2 = list(vec); v2[c] = off
        ref = full_pred(v2)
        mism = sum(int((preds[i] != ref[i]).sum()) for i in range(n))
        print(f'  trial {t}: c={NAMES[c]} off={off:+.4f} mismatched_px={mism}', flush=True)
        ok &= (mism == 0)
        del ov, oa, preds
    if not ok:
        raise SystemExit('FAST PATH NOT EXACT - aborting')
    print('SELFCHECK_PASS', flush=True)

    # ---- Stage C: 1D sensitivity ----
    print('STAGE_C 1D sensitivity', flush=True)
    sens = {}
    for c in range(1, 8):
        rows = scan_class([0.0] * 8, c, np.linspace(-half[c], half[c], 9))
        sens[NAMES[c]] = rows
        b = max(rows, key=lambda r: r['miou'])
        print(f'  {NAMES[c]:13s} half={half[c]:.5f} best={b["offset"]:+.5f} gain={b["gain_pp"]:+.4f}pp', flush=True)
    (outdir / 'stage_scale_sensitivity.json').write_text(json.dumps(
        {'zero_tta8_miou': base, 'zero_tta8_per_class_iou': [float(x) for x in base_iou],
         'zero_tta8_folds': fold_base, 'scale': scale, 'sensitivity_1d': sens},
        indent=2, ensure_ascii=False))

    # ---- Stage D: coordinate descent ----
    print('STAGE_D coordinate descent', flush=True)
    vec = [0.0] * 8
    cur = base
    trace = [{'round': 0, 'class': None, 'vector': list(vec), 'miou': cur, 'gain_pp': 0.0}]
    for rnd in range(1, 4):
        improved = False
        for c in range(1, 8):
            span = half[c]
            for _ in range(2):
                rows = scan_class(vec, c, np.linspace(vec[c] - span, vec[c] + span, 7))
                b = max(rows, key=lambda r: r['miou'])
                if b['miou'] > cur + 1e-10:
                    vec[c], cur, improved = b['offset'], b['miou'], True
                span /= 3.0
            trace.append({'round': rnd, 'class': NAMES[c], 'vector': list(vec),
                          'miou': cur, 'gain_pp': (cur - base) * 100})
        print(f'  round {rnd}: {cur:.8f} ({(cur-base)*100:+.4f}pp) {[round(x,4) for x in vec]}', flush=True)
        if not improved:
            break
    cd_vec, cd_miou = list(vec), cur

    # ---- Stage E: exact joint random search ----
    print('STAGE_E joint random search', flush=True)
    rng2 = np.random.default_rng(FOLD_SEED + 1)
    n_rand = 60
    pool = []
    best_vec, best_miou = list(cd_vec), cd_miou
    for t in range(n_rand):
        v = list(cd_vec)
        for c in rng2.choice(range(1, 8), size=int(rng2.integers(2, 5)), replace=False):
            v[int(c)] += float(rng2.normal(0, half[int(c)] / 8.0))
        _, m, _ = conf_and_miou(full_pred(v), all_idx)
        pool.append((m, list(v)))
        if m > best_miou:
            best_miou, best_vec = m, list(v)
        if t % 20 == 0:
            print(f'  RS {t}/{n_rand} best={best_miou:.8f}', flush=True)
    pool.sort(key=lambda x: -x[0])
    print(f'  RS best={best_miou:.8f} ({(best_miou-base)*100:+.4f}pp)', flush=True)

    # ---- candidate report ----
    def report(tag, v):
        p = full_pred(v)
        cm, m, iou = conf_and_miou(p, all_idx)
        fg = [(conf_and_miou(p, fold_of == k)[1] - fold_base[k]) * 100 for k in range(3)]
        sub = cm[1:, 1:].astype(np.float64); inter = np.diag(sub)
        fp = sub.sum(0) - inter; fn = sub.sum(1) - inter
        sub0 = cm0[1:, 1:].astype(np.float64); i0 = np.diag(sub0)
        ai = NAMES.index('Agricultural')
        return {'tag': tag, 'vector': [float(x) for x in v], 'full_miou': m,
                'gain_pp': (m - base) * 100, 'fold_gains_pp': fg, 'min_fold_gain_pp': min(fg),
                'per_class_iou': {NAMES[i]: float(iou[i]) for i in range(8)},
                'per_class_iou_gain_pp': {NAMES[i]: (float(iou[i]) - float(base_iou[i])) * 100 for i in range(8)},
                'agri_fp_px': int(fp[ai]), 'agri_fn_px': int(fn[ai]),
                'agri_fp_delta_px': int(fp[ai] - (sub0.sum(0) - i0)[ai]),
                'agri_fn_delta_px': int(fn[ai] - (sub0.sum(1) - i0)[ai])}

    cands = [('zero', [0.0] * 8), ('cd', cd_vec), ('rs_best', best_vec)]
    cands += [(f'rs_{i}', v) for i, (_, v) in enumerate(pool[:12])]
    for f_, tag in ((0.75, 'conservative_0.75x'), (0.5, 'conservative_0.5x'), (0.25, 'conservative_0.25x')):
        cands.append((tag, [0.0] + [f_ * best_vec[c] for c in range(1, 8)]))
    for c in range(1, 8):
        v = [0.0] * 8
        v[c] = max(sens[NAMES[c]], key=lambda r: r['miou'])['offset']
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
        'schema': 'M2F_56K_TTA8_CALIB_SEARCH_V3_EXACT',
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
        'scoring_rule': 'pred = argmax_c(S_bar_c + bias_c) over FULL 8 channels; no top-k reduction at any point',
        'cache': {'location': str(SHM) + ' (RAM, /dev/shm)', 'dtype': 'float32', 'channels': 8,
                  'images': n, 'disk_written': False},
        'offset_application': 'after TTA8 FP32 averaging, before argmax',
        'degrees_of_freedom': '8 entries, Background fixed 0 -> 7 effective',
        'fold_split': {'seed': FOLD_SEED, 'n_folds': 3, 'reoptimized_on_folds': False},
        'zero_tta8_miou': base, 'zero_tta8_per_class_iou': [float(x) for x in base_iou],
        'zero_tta8_folds': fold_base,
        'single_view_reference_miou': 0.8155545489734892,
        'scale': scale, 'sensitivity_1d': sens, 'coordinate_descent_trace': trace,
        'cd_vector': cd_vec, 'cd_miou': cd_miou,
        'random_search': {'n_candidates': n_rand, 'best_vector': best_vec, 'best_miou': best_miou},
        'top10': top10, 'fold_stable_pick': stable, 'conservative_pick': cons_best,
        'note': 'calibration gains are relative to TTA8 ZERO, never to single-view 0.8155545',
    }
    (outdir / 'CALIB_TTA8_SEARCH.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print('SEARCH_DONE', flush=True)
    print(f'  CD {cd_miou:.8f} ({(cd_miou-base)*100:+.4f}pp)   RS {best_miou:.8f} ({(best_miou-base)*100:+.4f}pp)', flush=True)
    for r in top10[:5]:
        print(f'  TOP {r["tag"]:18s} {r["full_miou"]:.8f} ({r["gain_pp"]:+.4f}pp) minfold={r["min_fold_gain_pp"]:+.4f}pp', flush=True)
    print(f'  FOLDSTABLE {stable["tag"]} {stable["gain_pp"]:+.4f}pp minfold={stable["min_fold_gain_pp"]:+.4f}pp', flush=True)


if __name__ == '__main__':
    main()
