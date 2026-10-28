#!/usr/bin/env python3
"""CPU-only, precision-gated evaluation of the original 45 calibration rows.

No expanded search, cache deletion, GPU, cache recomputation or report overwrite.
Run under the supervisor's nice -n 19 / ionice -c 3. This module independently
hides CUDA and pins numeric thread pools BEFORE importing NumPy.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _thread_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_key] = '2'

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
from phase_f_v2 import (Blocked, EXP41, LOCKED, TOLERANCE, VARIANTS, load_extra_bias,
                        report_pair, sha256_file, stable_record, strict_json,
                        valid_vector, validate_extra_bias, write_exclusive)

NUM = 8
EXPECTED_IMAGES = 1399
DOMAINS = ('D0', 'D1', 'D2')


def candidates():
    """45 pre-registered v1 rows, folded into unique vectors with alias provenance."""
    rows = [dict(name='A_FIXED8D_EXP41_STYLE', vec=list(EXP41), group='A_baseline_history'),
            dict(name='B_LOCKED8D_69_276229', vec=list(LOCKED), group='A_baseline_history')]
    def add(name, group, **values):
        vec = list(EXP41)
        for k, v in values.items():
            vec[{'bg': 0, 'br': 4, 'ag': 6, 'wa': 3}[k]] = v
        rows.append(dict(name=name, vec=vec, group=group))
    for br in (3.2, 3.4, 3.6, 3.8):
        add(f'B_barren_{br}', 'B_barren_only', br=br)
    for bg in (-.3, -.2, -.1):
        for br in (3.4, 3.6, 3.8):
            add(f'C_bg{bg}_barren{br}', 'C_bg_barren_2d', bg=bg, br=br)
    for bg in (-.3, -.2, -.1):
        for br in (3.4, 3.6, 3.8):
            for ag in (3.1, 3.3215, 3.5):
                add(f'D_bg{bg}_barren{br}_agri{ag}', 'D_bg_barren_agri_3d', bg=bg, br=br, ag=ag)
    for wa in (.6, .7, .8):
        add(f'E_water_{wa}', 'E_water_sensitivity', wa=wa)
    assert len(rows) == 45
    unique = {}
    for row in rows:
        key = tuple(row['vec'])
        if key not in unique:
            unique[key] = {**row, 'aliases': []}
        unique[key]['aliases'].append({'name': row['name'], 'group': row['group']})
    return list(unique.values())


def metrics_from_confusion(cm):
    cm = np.asarray(cm)
    if cm.shape != (8, 8) or not np.issubdtype(cm.dtype, np.integer) or np.any(cm < 0):
        raise ValueError('confusion matrix must be nonnegative integer 8x8')
    diag = np.diag(cm).astype(np.float64)
    row, col = cm.sum(1), cm.sum(0)
    def ratio(den):
        return [float(a / b) if b else None for a, b in zip(diag, den)]
    iou = ratio(row + col - diag)
    present = [v for v in iou if v is not None]
    return {'mIoU': sum(present) / len(present) if present else None,
            'per_class_iou': iou, 'per_class_precision': ratio(col),
            'per_class_recall': ratio(row), 'gt_pixels': row.tolist(),
            'pred_pixels': col.tolist(), 'valid_pixels': int(cm.sum()),
            'confusion_matrix': cm.tolist()}


def accumulate(cm, logits, bias, gt):
    pred = (logits + np.asarray(bias, dtype=np.float32).reshape(1, 1, 8)).argmax(-1)
    valid = (gt >= 0) & (gt < 8)
    encoded = gt[valid].astype(np.int64) * 8 + pred[valid]
    cm += np.bincount(encoded, minlength=64).reshape(8, 8)


def rank(records, baseline):
    """Same-domain zero bias is mandatory; unstable rows never enter top3."""
    ranked = []
    seen = set()
    for record in records:
        key = tuple(valid_vector(record['calibration_vector']))
        if key in seen:
            raise ValueError('duplicate candidate vector in ranking')
        seen.add(key)
        stable, gain, deltas = stable_record(record, baseline)
        r = {**record, 'stable': stable, 'full_val_delta': gain,
             'domain_deltas': deltas,
             'domain_delta_min': min(deltas.values()) if deltas else None,
             'domain_delta_spread': max(deltas.values()) - min(deltas.values()) if deltas else None}
        ranked.append(r)
    return sorted(ranked, key=lambda r: (not r['stable'],
                  -(r['domain_delta_min'] if r['domain_delta_min'] is not None else -math.inf),
                  -(r['full_val_delta'] if r['full_val_delta'] is not None else -math.inf), r['name']))


def read_split(path, expected=EXPECTED_IMAGES):
    stems = [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(stems) != expected or len(set(stems)) != expected:
        raise Blocked(f'validation split must contain exactly {expected} unique stems')
    if any(Path(s).name != s or '/' in s or '\\' in s or s in ('.', '..') or s.endswith('.png') for s in stems):
        raise Blocked('split must contain safe extensionless image stems')
    return stems


def read_domains(path, stems):
    """No fallback assignment: CSV must cover the exact split once, and only it."""
    mapping = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {'image_name', 'domain_id'} <= set(reader.fieldnames):
            raise Blocked('domain CSV needs image_name,domain_id')
        for row in reader:
            name = row['image_name']
            stem = name[:-4] if name.endswith('.png') else name
            if stem not in stems or stem in mapping:
                raise Blocked('domain CSV extra/duplicate image: ' + name)
            value = row['domain_id']
            if value not in ('0', '1', '2'):
                raise Blocked('domain CSV invalid domain: ' + value)
            mapping[stem] = int(value)
    if set(mapping) != set(stems) or set(mapping.values()) != {0, 1, 2}:
        raise Blocked('domain CSV missing images or empty domain')
    return mapping


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def _gate_artifact(report):
    if (report.get('schema_version') != 1 or report.get('status') != 'PASS' or
            report.get('payload_sha256') != _canonical_sha({k: v for k, v in report.items() if k != 'payload_sha256'})):
        raise ValueError('invalid immutable gate artifact payload/status/version')


def _sampled_precision(report, cache_dir, checkpoint_sha, config_sha, split_sha):
    """Authorized old-B fixed16 audit plus immutable full1399 integrity receipt."""
    _gate_artifact(report)
    if report.get('precision_status') != 'FP16_LOGIT_CACHE_PRECISION_PASS' or report.get('calibration_allowed') is not True or report.get('n') != 16 or report.get('n_images') != EXPECTED_IMAGES:
        raise ValueError('old-B sampled precision is not an authorized fixed16 PASS')
    expected_biases = {'no_bias': [0.] * 8, 'FIXED8D_EXP41_STYLE': EXP41, 'LOCKED8D_69_276229': LOCKED}
    if report.get('bias_vectors') != expected_biases:
        raise ValueError('sampled precision bias vectors differ')
    for key, expected in (('checkpoint_sha256', checkpoint_sha), ('config_sha256', config_sha), ('split_sha256', split_sha)):
        if report.get(key) != expected:
            raise ValueError('sampled precision source mismatch: ' + key)
    if Path(report['cache_dir']).resolve() != Path(cache_dir).resolve():
        raise ValueError('sampled precision cache path mismatch')
    artifacts = {}
    for path, digest in report.get('artifact_sha256', {}).items():
        if sha256_file(path) != digest:
            raise ValueError('precision prerequisite artifact SHA mismatch: ' + path)
        if Path(path).suffix == '.json':
            child = strict_json(path)
            _gate_artifact(child)
            gate = child.get('gate')
            if gate in artifacts:
                raise ValueError('duplicate prerequisite gate artifact')
            artifacts[gate] = child
    if not {'integrity', 'select', 'benchmark'} <= set(artifacts):
        raise ValueError('sampled precision lacks integrity/frozen sample/benchmark bindings')
    integrity, sample = artifacts['integrity'], artifacts['select']
    for key in ('checkpoint_sha256', 'config_sha256', 'split_sha256', 'image_list_sha256', 'cache_dir', 'n_images'):
        if integrity.get(key) != report.get(key):
            raise ValueError('integrity/precision identity mismatch: ' + key)
    inventory = integrity.get('cache_inventory', {})
    if len(inventory) != EXPECTED_IMAGES or integrity.get('cache_inventory_sha256') != _canonical_sha(inventory):
        raise ValueError('integrity inventory is not exact1399 SHA-bound')
    samples = sample.get('samples', [])
    sample_stems = [s['stem'] for s in samples]
    if (sample.get('frozen_before_gpu') is not True or sample.get('seed') != 42 or sample.get('n') != 16 or
            sample.get('split_sha256') != split_sha or sample.get('bias_vectors') != expected_biases or
            len(samples) != 16 or len(set(sample_stems)) != 16 or report.get('sample_stems') != sample_stems or
            any(s not in inventory for s in sample_stems) or
            [sum(s['domain'] == d for s in samples) for d in DOMAINS] != [6, 5, 5]):
        raise ValueError('invalid frozen fixed16 membership/domains')
    thresholds = report.get('thresholds', {})
    if thresholds != sample.get('thresholds'):
        raise ValueError('frozen precision thresholds changed')
    for field in ('disagreement', 'aggregate_iou_delta'):
        value = thresholds.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= TOLERANCE:
            raise ValueError('invalid frozen precision threshold')
    if thresholds.get('logit_error') != 1e-3:
        raise ValueError('invalid frozen logit threshold')
    for group in ('old_cache_combined_drift', 'v2_self_quantization'):
        rows = report.get('comparisons', {}).get(group, {})
        if set(rows) != set(expected_biases):
            raise ValueError('missing precision comparison bias')
        for row in rows.values():
            if row.get('status') != 'PASS' or [r['stem'] for r in row.get('images', [])] != sample_stems:
                raise ValueError('precision comparison failed or sample drifted')
            for field, threshold in (('max_per_image_disagreement', thresholds['disagreement']),
                                      ('max_iou_delta', thresholds['aggregate_iou_delta']),
                                      ('miou_delta', thresholds['aggregate_iou_delta'])):
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < threshold:
                    raise ValueError('sample precision comparison exceeds threshold')
            if any(r.get('meaningful_disagreements') != 0 for r in row['images']):
                raise ValueError('meaningful precision disagreements')
    return {**report, 'cache_file_sha256': {s + '.npy': row['sha256'] for s, row in inventory.items()},
            'precision_sample_assignments_sha256': sample['assignments_sha256']}


def precision_gate(audit_path, cache_dir, checkpoint_sha, config_sha, split_sha):
    """Bind actual precision PASS to this source, not an unrelated model audit."""
    try:
        report = strict_json(audit_path)
        if report.get('gate') == 'precision':
            return _sampled_precision(report, cache_dir, checkpoint_sha, config_sha, split_sha)
        audit = report.get('precision_audit', report)
        if audit.get('status') not in ('PASS', 'FP16_LOGIT_CACHE_PRECISION_PASS'):
            raise ValueError('precision status is not PASS')
        if audit.get('downstream_cache_use_allowed') is not True or report.get('n_images') != EXPECTED_IMAGES or report.get('split') != 'val':
            raise ValueError('intrinsic precision report does not authorize full validation cache use')
        for key, expected in (('checkpoint_sha256', checkpoint_sha), ('config_sha256', config_sha),
                              ('split_sha256', split_sha)):
            # Top-level binding is required; if nested binding exists it must agree.
            if report.get(key) != expected or (key in audit and audit[key] != expected):
                raise ValueError('precision source mismatch: ' + key)
        bound_cache = report.get('cache_dir')
        if not bound_cache or Path(bound_cache).resolve() != Path(cache_dir).resolve():
            raise ValueError('precision cache directory mismatch')
        if 'cache_dir' in audit and Path(audit['cache_dir']).resolve() != Path(cache_dir).resolve():
            raise ValueError('nested precision cache directory mismatch')
        if audit.get('n_images') != EXPECTED_IMAGES or audit.get('disk_reload_n', 0) < 32:
            raise ValueError('precision does not cover all 1399 images and 32 disk reloads')
        for field in ('argmax_disagreement_max', 'miou_delta_max'):
            value = audit.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < TOLERANCE:
                raise ValueError('precision threshold failed: ' + field)
        expected_variants = {'no_bias': [0.] * 8, 'FIXED8D_EXP41_STYLE': EXP41,
                             'LOCKED8D_69_276229': LOCKED}
        variants = audit.get('variants', {})
        if set(variants) != set(expected_variants) or audit.get('reference_logits_n') != EXPECTED_IMAGES:
            raise ValueError('precision must compare original fp32 for all three allowed biases')
        for name, vector in expected_variants.items():
            row = variants[name]
            if valid_vector(row.get('bias')) != vector or row.get('n_images') != EXPECTED_IMAGES:
                raise ValueError('precision variant identity/count mismatch: ' + name)
            for field in ('argmax_disagreement_max', 'miou_delta_max', 'aggregate_miou_delta'):
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < TOLERANCE:
                    raise ValueError('precision variant threshold failed: ' + name + '.' + field)
        return report
    except Exception as exc:
        raise Blocked('BLOCKED_PRECISION: BLOCKED_PRECISE_CACHE_REQUIRED: ' + str(exc)) from exc


def cache_manifest(cache_dir, stems):
    expected = {s + '.npy' for s in stems}
    actual = {p.name for p in Path(cache_dir).glob('*.npy')}
    if expected != actual:
        raise Blocked(f'cache coverage mismatch missing={len(expected-actual)} extra={len(actual-expected)}')
    hashes = {name: sha256_file(Path(cache_dir) / name) for name in sorted(expected)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return hashes, digest


def run(args):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.model_tag) or args.model_tag in ('.', '..'):
        raise Blocked('unsafe model tag')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = 'CALIBRATION_CONSTRAINED_REPORT_' + args.model_tag
    if any((args.out_dir / (stem + ext)).exists() for ext in ('.json', '.md', '.started.json')):
        raise Blocked('CONFLICTING_REPORT: partial/existing run preserved')
    write_exclusive(args.out_dir / (stem + '.started.json'), json.dumps({'model_tag': args.model_tag, 'status': 'STARTED'}) + '\n')
    result = {'schema': 'constrained_calibration_v2', 'status': 'RUNNING', 'model_tag': args.model_tag,
              'checkpoint_sha256': args.source_checkpoint_sha256, 'config_sha256': args.source_config_sha256,
              'cache_dir': str(args.cache_dir.resolve()), 'precision_status': 'NOT_EVALUATED',
              'n_images_used': 0, 'top3': [], 'all_candidates': [],
              'raw_candidate_count': 45, 'unique_candidate_count': len(candidates()),
              'stable_tolerance': TOLERANCE, 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
              'threads': 2, 'historical_attribution': 'NEITHER vector is asserted to be exact 69.281290 bias; identity unresolved.'}
    start = time.monotonic()
    try:
        for sha in (args.source_checkpoint_sha256, args.source_config_sha256):
            if not re.fullmatch('[0-9a-f]{64}', sha):
                raise Blocked('source identity must be lowercase SHA256')
        result['split_sha256'] = sha256_file(args.split_file)
        result['domain_csv_sha256'] = sha256_file(args.domain_csv)
        audit = precision_gate(args.precision_audit, args.cache_dir, args.source_checkpoint_sha256,
                               args.source_config_sha256, result['split_sha256'])
        result['precision_status'] = 'PASS'
        if audit.get('precision_sample_assignments_sha256', result['domain_csv_sha256']) != result['domain_csv_sha256']:
            raise Blocked('BLOCKED_PRECISION: frozen sample domain assignments mismatch')
        result['precision_audit_kind'] = 'frozen_fixed16_old_B' if audit.get('gate') == 'precision' else 'full1399_new_C'
        result['precision_audit_path'] = str(args.precision_audit.resolve())
        result['precision_audit_sha256'] = sha256_file(args.precision_audit)
        if audit.get('weight_content_sha256'):
            result['weight_content_sha256'] = audit['weight_content_sha256']
        stems = read_split(args.split_file)
        domains = read_domains(args.domain_csv, stems)
        result['domain_counts'] = {f'D{d}': list(domains.values()).count(d) for d in range(3)}
        hashes, manifest_sha = cache_manifest(args.cache_dir, stems)
        if audit.get('cache_file_sha256') != hashes:
            raise Blocked('BLOCKED_PRECISION: audited cache content manifest missing or mismatched')
        result['cache_manifest_sha256'] = manifest_sha
        # Every mask must exist before consuming any cache; preserve full identity.
        mask_hashes = {s + '.png': sha256_file(args.mask_dir / (s + '.png')) for s in stems}
        result['mask_manifest_sha256'] = hashlib.sha256(json.dumps(mask_hashes, sort_keys=True,
                                                       separators=(',', ':')).encode()).hexdigest()
        cs = candidates()
        vectors = [[0.] * 8] + [c['vec'] for c in cs]
        cms = np.zeros((len(vectors), 4, 8, 8), dtype=np.int64)
        from PIL import Image
        for i, s in enumerate(stems):
            path = args.cache_dir / (s + '.npy')
            if sha256_file(path) != hashes[path.name]:
                raise Blocked('cache changed during calibration: ' + s)
            lg = np.load(path, allow_pickle=False)
            if lg.dtype != np.float16 or lg.ndim != 3 or lg.shape[-1] != 8 or not np.isfinite(lg).all():
                raise Blocked('invalid fp16 HWC8 cache: ' + s)
            lg = lg.astype(np.float32)
            mask_path = args.mask_dir / (s + '.png')
            if sha256_file(mask_path) != mask_hashes[mask_path.name]:
                raise Blocked('mask changed during calibration: ' + s)
            with Image.open(mask_path) as im:
                gt_official = np.asarray(im).copy()
            if gt_official.ndim != 2 or gt_official.shape != lg.shape[:2] or not np.isin(gt_official, [0,1,2,3,4,5,6,7,8,255]).all():
                raise Blocked('invalid official mask: ' + s)
            gt = np.full(gt_official.shape, 255, dtype=np.int64)
            valid = (gt_official >= 1) & (gt_official <= 8)
            gt[valid] = gt_official[valid].astype(np.int64) - 1
            for j, vector in enumerate(vectors):
                cm = np.zeros((8, 8), dtype=np.int64)
                accumulate(cm, lg, vector, gt)
                cms[j, 0] += cm
                cms[j, domains[s] + 1] += cm
            result['n_images_used'] = i + 1
            if (i + 1) % 100 == 0:
                print(f'CALIBRATION_PROGRESS {args.model_tag} {i+1}/1399', flush=True)
        if sha256_file(args.precision_audit) != result['precision_audit_sha256'] or sha256_file(args.split_file) != result['split_sha256'] or sha256_file(args.domain_csv) != result['domain_csv_sha256']:
            raise Blocked('provenance inputs changed during calibration')
        names = ('full_val', *DOMAINS)
        baseline = {name: metrics_from_confusion(cms[0, k]) for k, name in enumerate(names)}
        if any(baseline[name]['mIoU'] is None for name in names):
            raise Blocked('empty full-val/domain confusion matrix')
        records = [{ 'name': c['name'], 'group': c['group'], 'aliases': c['aliases'],
                     'calibration_vector': c['vec'],
                     'metrics': {name: metrics_from_confusion(cms[j + 1, k]) for k, name in enumerate(names)}}
                   for j, c in enumerate(cs)]
        ranked = rank(records, baseline)
        result.update(status='PASS', baseline_zero_bias=baseline, all_candidates=ranked,
                      top3=[r for r in ranked if r['stable']][:3],
                      ranking_rule='same-domain zero-bias deltas >= -1e-4 AND full-val gain > 0; min-domain-delta then full-val gain')
    except Exception as exc:
        result['status'] = 'BLOCKED_PRECISION' if 'BLOCKED_PRECISION' in str(exc) else 'BLOCKED'
        result['error'] = str(exc)
        result['elapsed_seconds'] = time.monotonic() - start
        report_pair(args.out_dir, stem, result)
        raise
    result['elapsed_seconds'] = time.monotonic() - start
    report_pair(args.out_dir, stem, result)
    return result


def compare_reports(paths, out_dir):
    """Emit at most one extra vector, eligible in BOTH distinct model top3s."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / name).exists() for name in ('COMPARISON_REPORT.json', 'COMPARISON_REPORT.md', 'phase_d_winner.json', 'COMPARISON_REPORT.started.json')):
        raise Blocked('CONFLICTING_COMPARISON: existing/partial outputs preserved')
    write_exclusive(out_dir / 'COMPARISON_REPORT.started.json', '{"status":"STARTED"}\n')
    summary = {'schema': 'phase_d_comparison_v2', 'status': 'RUNNING', 'winner': None}
    try:
        if len(paths) != 2:
            raise Blocked('requires exactly two source reports')
        reports = [strict_json(p) for p in paths]
        if len({r.get('model_tag') for r in reports}) != 2 or len({r.get('weight_content_sha256') or r.get('checkpoint_sha256') for r in reports}) != 2:
            raise Blocked('comparison needs two distinct models, not aliases of one source')
        coverage = {(r.get('split_sha256'), r.get('domain_csv_sha256'), r.get('mask_manifest_sha256')) for r in reports}
        if len(coverage) != 1 or any(v is None for v in next(iter(coverage))):
            raise Blocked('comparison validation source coverage mismatch')
        source_refs = []
        pools = []
        allowed = {tuple(c['vec']) for c in candidates()}
        for p, r in zip(paths, reports):
            if r.get('schema') != 'constrained_calibration_v2' or r.get('status') != 'PASS' or r.get('n_images_used') != EXPECTED_IMAGES or r.get('precision_status') != 'PASS':
                raise Blocked('comparison requires two full1399 precision-PASS calibrations')
            ranked = rank(r['all_candidates'], r['baseline_zero_bias'])
            if {tuple(x['calibration_vector']) for x in ranked} != allowed:
                raise Blocked('source candidate set differs from preregistration')
            expected_top = [x for x in ranked if x['stable']][:3]
            if [x['name'] for x in expected_top] != [x['name'] for x in r['top3']]:
                raise Blocked('source top3 is not genuine stable ranking')
            pools.append({tuple(x['calibration_vector']): x for x in expected_top})
            source_refs.append({'model_tag': r['model_tag'], 'checkpoint_sha256': r['checkpoint_sha256'],
                                'report_path': str(Path(p).resolve()), 'report_sha256': sha256_file(p)})
        common = set(pools[0]) & set(pools[1])
        common -= {tuple(v) for v in VARIANTS.values()}
        summary['comparisons'] = source_refs
        summary['eligible_common_extra_vectors'] = len(common)
        summary['status'] = 'PASS'
        if common:
            vector = max(sorted(common), key=lambda v: (min(pool[v]['domain_delta_min'] for pool in pools),
                                                         min(pool[v]['full_val_delta'] for pool in pools)))
            winner = {'schema': 'phase_d_winner_v2', 'status': 'PASS', 'name': 'PHASE_D_STABLE',
                      'vector': list(vector), 'comparisons': source_refs,
                      'eligibility': 'stable shortlisted vector in two distinct models; no historical duplicate'}
            winner_path = out_dir / 'phase_d_winner.json'
            validate_extra_bias(winner, winner_path)
            write_exclusive(winner_path, json.dumps(winner, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
            write_exclusive(out_dir / 'phase_d_winner.json.sha256', sha256_file(winner_path) + '  phase_d_winner.json\n')
            summary['winner'] = {'path': str(winner_path.resolve()), 'sha256': sha256_file(winner_path)}
        else:
            summary['reason'] = 'NO_ELIGIBLE_COMMON_STABLE_EXTRA: fixed PhaseF variants only; no forced winner'
    except Exception as exc:
        summary.update(status='BLOCKED', error=str(exc))
        report_pair(out_dir, 'COMPARISON_REPORT', summary)
        raise
    report_pair(out_dir, 'COMPARISON_REPORT', summary)
    return summary


def self_test():
    import tempfile
    root = Path(tempfile.mkdtemp(prefix='calibration_v2_cpu_fixture_'))
    cs = candidates()
    assert sum(len(c['aliases']) for c in cs) == 45
    assert len(cs) == len({tuple(c['vec']) for c in cs}) == 31
    cm = np.zeros((8, 8), dtype=np.int64)
    cm[0, 0], cm[0, 1], cm[1, 1] = 3, 1, 2
    m = metrics_from_confusion(cm)
    assert m['per_class_precision'][:2] == [1., 2/3]
    assert m['per_class_recall'][:2] == [.75, 1.]
    assert m['per_class_iou'][:2] == [.75, 2/3] and m['per_class_iou'][2] is None
    base = {d: {'mIoU': v} for d, v in zip(('full_val', *DOMAINS), (.7, .9, .5, .7))}
    def record(name, values, vector):
        return {'name': name, 'calibration_vector': vector,
                'metrics': {d: {'mIoU': v} for d, v in zip(('full_val', *DOMAINS), values)}}
    stable = record('stable', (.71, .9001, .5001, .7001), cs[0]['vec'])
    wrong = record('wrong_crossdomain_baseline', (.8, .8, .8, .8), cs[1]['vec'])
    boundary = record('tolerated', (.7001, .89995, .5, .7), cs[2]['vec'])
    ranked = rank([wrong, stable, boundary], base)
    assert [r['stable'] for r in ranked] == [True, True, False]
    assert rank([wrong], base)[0]['domain_deltas']['D0'] < -.09
    neutral = record('zero_gain', (.7, .9, .5, .7), cs[0]['vec'])
    assert not rank([neutral], base)[0]['stable']
    write_exclusive(root / 'domains.csv', 'image_name,domain_id\na.png,0\nb.png,1\nc.png,2\n')
    assert read_domains(root / 'domains.csv', ['a', 'b', 'c']) == {'a': 0, 'b': 1, 'c': 2}
    try:
        read_domains(root / 'domains.csv', ['a', 'b', 'c', 'd'])
    except Blocked:
        pass
    else:
        raise AssertionError('missing domain accepted')
    write_exclusive(root / 'failed_precision.json', '{"precision_audit":{"status":"FAIL"}}')
    try:
        precision_gate(root / 'failed_precision.json', root, 'a'*64, 'b'*64, 'c'*64)
    except Blocked as e:
        assert 'BLOCKED_PRECISION' in str(e)
    else:
        raise AssertionError('failed precision accepted')
    cache = root / 'cache'; cache.mkdir()
    np.save(cache / 'a.npy', np.zeros((2, 2, 8), dtype=np.float16))
    try:
        cache_manifest(cache, ['a', 'b'])
    except Blocked:
        pass
    else:
        raise AssertionError('missing cache accepted')
    # Two genuine distinct source reports: winner publication and no-winner path.
    source_paths = []
    flat_base = {d: {'mIoU': .7} for d in ('full_val', *DOMAINS)}
    for index in range(2):
        records = [record(c['name'], (.71, .71, .71, .71) if j == 2 else (.7, .7, .7, .7), c['vec'])
                   for j, c in enumerate(cs)]
        ranked = rank(records, flat_base)
        source = {'schema': 'constrained_calibration_v2', 'status': 'PASS',
                  'model_tag': 'fixture' + str(index), 'checkpoint_sha256': str(index + 1) * 64,
                  'split_sha256': 'a' * 64, 'domain_csv_sha256': 'b' * 64,
                  'mask_manifest_sha256': 'c' * 64, 'n_images_used': 1399, 'precision_status': 'PASS',
                  'baseline_zero_bias': flat_base, 'all_candidates': ranked,
                  'top3': [r for r in ranked if r['stable']][:3]}
        source_path = root / ('source' + str(index) + '.json')
        write_exclusive(source_path, json.dumps(source, allow_nan=False))
        source_paths.append(source_path)
    comparison = compare_reports(source_paths, root / 'positive_comparison')
    assert comparison['winner'] and load_extra_bias(Path(comparison['winner']['path']))[1] == cs[2]['vec']
    try:
        compare_reports(source_paths, root / 'positive_comparison')
    except Blocked:
        pass
    else:
        raise AssertionError('comparison overwrite accepted')
    negative_paths = []
    for index, path in enumerate(source_paths):
        source = strict_json(path)
        source['all_candidates'] = rank([record(c['name'], (.7, .7, .7, .7), c['vec']) for c in cs], flat_base)
        source['top3'] = []
        path = root / ('negative' + str(index) + '.json')
        write_exclusive(path, json.dumps(source, allow_nan=False))
        negative_paths.append(path)
    no_winner = compare_reports(negative_paths, root / 'negative_comparison')
    assert no_winner['winner'] is None and not (root / 'negative_comparison' / 'phase_d_winner.json').exists()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and all(os.environ[k] == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))
    assert 'torch' not in sys.modules
    print('SELF_TEST_OK constrained_calibration_v2: 45 rows / 31 unique; CPU fixtures preserved at ' + str(root))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-tag')
    ap.add_argument('--cache-dir', type=Path)
    ap.add_argument('--precision-audit', type=Path)
    ap.add_argument('--source-checkpoint-sha256')
    ap.add_argument('--source-config-sha256')
    ap.add_argument('--split-file', type=Path)
    ap.add_argument('--domain-csv', type=Path)
    ap.add_argument('--mask-dir', type=Path)
    ap.add_argument('--out-dir', type=Path)
    ap.add_argument('--compare-reports', type=Path, nargs=2)
    ap.add_argument('--nice', action='store_true', help='compatibility flag: nice19 is always requested on POSIX')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if hasattr(os, 'nice'):
        try:
            os.nice(max(0, 19 - os.nice(0)))
        except OSError as exc:
            print('NICE19_UNAVAILABLE: ' + str(exc), file=sys.stderr)
    if not args.out_dir:
        ap.error('--out-dir required')
    try:
        if args.compare_reports:
            result = compare_reports(args.compare_reports, args.out_dir)
        else:
            required = ('model_tag', 'cache_dir', 'precision_audit', 'source_checkpoint_sha256',
                        'source_config_sha256', 'split_file', 'domain_csv', 'mask_dir')
            if any(getattr(args, key) is None for key in required):
                ap.error('source evaluation requires ' + ', '.join('--' + x.replace('_', '-') for x in required))
            result = run(args)
        print(json.dumps({'status': result['status'], 'top3': [r['name'] for r in result.get('top3', [])],
                          'winner': result.get('winner')}, allow_nan=False))
    except Exception as exc:
        print(json.dumps({'status': 'BLOCKED', 'error': str(exc)}, allow_nan=False), file=sys.stderr)
        return 2 if 'BLOCKED_PRECISION' in str(exc) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
