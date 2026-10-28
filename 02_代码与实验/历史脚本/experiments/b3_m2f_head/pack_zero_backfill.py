"""Historical checkpoint ZERO packages: D4-TTA8 FP32, no calibration, no bias.

One job per checkpoint. Each job is an independent process invocation so a
finished package survives a later failure:

    python pack_zero_backfill.py forward <jobkey>
    python pack_zero_backfill.py package <jobkey>

`forward` loads the snapshot model exactly as the frozen engine does
(src.models.build_model with cwfa disabled) and stacks strict=True on the
checkpoint's "model" state dict. `package` refuses to run unless the forward
completion marker exists, so no job can silently re-forward or reuse the
contaminated packages_s59_v1 predictions. Every checkpoint SHA is asserted
against the inventory before and after the forward.

Physical GPU2 only (PCI_BUS_ID + CUDA_VISIBLE_DEVICES=2). No training, no
validation, no platform submission.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
TEST = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images')
ENGINE = Path('/home/jhy/gpu2_day_queue/tta_anatomy_v2.py')
OUTROOT = Path('/home/jhy/rare_diag_20260915/zero_backfill_20260916')
S63 = SNAP / 'stage63_runs/stage63_w1_exp63_b3_cwfa_wd005_seed42'
S66 = SNAP / 'stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42'
S59 = SNAP / 'stage59_runs/stage59_w1_exp59_b3_cwfa_ce_regionC_barrenAgri_seed42'
F69 = Path('/home/jhy/gpu2_day_queue/full6996_refit')

JOBS = {
    'stage63_56000': (S63 / 'config.yaml', S63 / 'best_miou.pt', 56000,
                      'bc052839711d474e847ee9246da8e81b30d7681ccadb747705cedfeda0a7da44'),
    'stage66_52000': (S66 / 'config.yaml', S66 / 'checkpoint_step52000.pt', 52000,
                      '311919e410d43cfaef5138570c754010a62a3c5b84c8a3a736af016157ce8afb'),
    'stage66_56000': (S66 / 'config.yaml', S66 / 'checkpoint_step56000.pt', 56000,
                      'f70c03eceb1ca560bed8c59dbb2b9d790391a01c0a6bc0823e17b2538a708aa2'),
    'stage59_44000': (S59 / 'config.yaml', S59 / 'checkpoint_step44000.pt', 44000,
                      '729a51e3091a40c4be9e6926a27bd95b4aeb30a829a4dbbed9be7b1840d52490'),
    'stage59_48000': (S59 / 'config.yaml', S59 / 'checkpoint_step48000.pt', 48000,
                      '0e7699e45e28184462b78d08ad3dec9e76105c91e6a498deef9cf1ba38d65057'),
    'stage59_52000': (S59 / 'config.yaml', S59 / 'checkpoint_step52000.pt', 52000,
                      '718808b10bbe9f92c04b3e650319f18da1219d4fbf4b7e227aa1504f9b295b05'),
    'full6996_55000': (F69 / 'config.yaml', F69 / 'checkpoint_step55000.pt', 55000,
                       'e2233d8fb3191b73e839a46abdc3155de670b224f5ecea65388858e1a98a082e'),
    'full6996_60000': (F69 / 'config.yaml', F69 / 'checkpoint_step60000.pt', 60000,
                       'cdc088ebc1da7b0261b9d3ee0d042bf4262e9326df8b7dd739b31953421ca2c7'),
}
CHECKER = SNAP / 'scripts/check_submission.py'
NAMES = [f'test1_{i}.png' for i in range(1, 501)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def guard(reserve=0):
    free = shutil.disk_usage(OUTROOT.parent).free
    if free - reserve < int(1.5 * 1024**3):
        raise RuntimeError(f'Disk floor: free={free} reserve={reserve}')


def forward(key):
    config, ckpt, step, expected = JOBS[key]
    if (os.environ.get('CUDA_VISIBLE_DEVICES'), os.environ.get('CUDA_DEVICE_ORDER')) != ('2', 'PCI_BUS_ID'):
        raise RuntimeError('Physical GPU2 only')
    guard(120 * 1024**2)
    if {p.name for p in TEST.glob('*.png')} != set(NAMES):
        raise RuntimeError('Unexpected test image names')
    if sha(ckpt) != expected:
        raise RuntimeError(f'{key}: checkpoint SHA mismatch')
    out = OUTROOT / key
    # the engine module lives beside its own directory, not inside the snapshot
    for p in (str(ENGINE.parent), str(SNAP)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    import torch
    from PIL import Image
    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import forward_views, VIEWS, build_model

    out.mkdir(parents=True, exist_ok=False)
    model = build_model(config, ckpt, 'cuda:0')
    payload = torch.load(ckpt, map_location='cpu', weights_only=False)
    meta = {k: v for k, v in payload.items() if isinstance(v, (str, int, float, bool))}
    if int(meta.get('step', -1)) != step:
        raise RuntimeError(f'{key}: checkpoint step {meta.get("step")} != {step}')
    del payload
    sources = [Path(__file__), ENGINE, SNAP / 'src/transforms.py', SNAP / 'src/models.py']
    save(out / 'INFERENCE_CONFIG.json', {
        'schema': 'HISTORICAL_ZERO_BACKFILL_V1', 'job': key,
        'config': str(config), 'config_sha256': sha(config),
        'checkpoint': str(ckpt), 'checkpoint_sha256': expected, 'step': step,
        'checkpoint_meta': meta, 'tta': 'D4-TTA8', 'views': VIEWS, 'view_batch_size': 1,
        'precision': 'FP32; autocast disabled; FP32 view mean',
        'bias': [0.0] * 8, 'bias_application': 'none (ZERO)',
        'prediction': 'argmax(mean of 8 view logits) + 1 -> uint8 PNG',
        'cwfa': 'disabled at inference (training-only augmentation)',
        'code_sha256': {str(p): sha(p) for p in sources},
        'test_count': 500, 'physical_gpu': 2, 'single_forward_pass': True})
    (out / 'predictions').mkdir()
    start = time.time()
    with torch.inference_mode():
        for i, name in enumerate(NAMES, 1):
            guard(8 * 1024**2)
            with Image.open(TEST / name) as im:
                image = np.asarray(im.convert('RGB')).copy()
            if image.shape != (1024, 1024, 3):
                raise RuntimeError(f'Unexpected shape: {name}')
            px = normalize_to_tensor(image).unsqueeze(0).to('cuda:0')
            with torch.amp.autocast('cuda', enabled=False):
                views = forward_views(model, px, 1)
                scores = views.mean(dim=0, dtype=torch.float32)
            pred = scores.argmax(0).add(1).to(torch.uint8)
            Image.fromarray(pred.cpu().numpy()).save(out / 'predictions' / name)
            del px, views, scores, pred
            if i % 100 == 0:
                print(f'{key} PROGRESS {i}/500 elapsed={time.time()-start:.1f}s', flush=True)
    if sha(ckpt) != expected:
        raise RuntimeError(f'{key}: checkpoint changed while inferring')
    save(out / 'FORWARD_DONE.json', {'job': key, 'images': 500,
                                     'elapsed_sec': time.time() - start})
    print(f'{key} FORWARD_DONE', flush=True)


def package(key):
    _, ckpt, step, expected = JOBS[key]
    out = OUTROOT / key
    if not (out / 'FORWARD_DONE.json').is_file():
        raise RuntimeError(f'{key}: forward marker missing; refusing to re-forward')
    if sha(ckpt) != expected:
        raise RuntimeError(f'{key}: checkpoint SHA mismatch')
    pred = out / 'predictions'
    if {p.name for p in pred.glob('*.png')} != set(NAMES):
        raise RuntimeError(f'{key}: incomplete predictions')
    check_dir = out / 'check'
    subprocess.run([sys.executable, str(CHECKER), '--test-images', str(TEST),
                    '--predictions', str(pred), '--output-dir', str(check_dir),
                    '--expected-count', '500'],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    check = json.loads((check_dir / 'submission_check.json').read_text())
    if check['status'] != 'PASS':
        raise RuntimeError(f'{key}: official checker failed')
    zpath = out / f'ZERO_{key}_TTA8.zip'
    guard(sum((pred / n).stat().st_size for n in NAMES) + 8 * 1024**2)
    with zipfile.ZipFile(zpath, 'x', compression=zipfile.ZIP_STORED) as z:
        for n in NAMES:
            z.write(pred / n, arcname=n)
    with zipfile.ZipFile(zpath) as z:
        if z.namelist() != NAMES or z.testzip() is not None:
            raise RuntimeError(f'{key}: ZIP failed')
    row = {'job': key, 'step': step, 'bias': [0.0] * 8,
           'checkpoint_sha256': expected, 'checkpoint': str(ckpt),
           'inference_config_sha256': sha(out / 'INFERENCE_CONFIG.json'),
           'package_filename': zpath.name, 'package_sha256': sha(zpath),
           'package_bytes': zpath.stat().st_size, 'official_check': 'PASS',
           'entries': 500, 'platform_miou': None}
    save(out / 'PACKAGE.json', row)
    print(f'{key} PACKAGE_READY {zpath.name} {row["package_sha256"]}', flush=True)


if __name__ == '__main__':
    mode, key = sys.argv[1], sys.argv[2]
    if key == 'all':
        keys = list(JOBS)
    else:
        keys = key.split(',')
    for k in keys:
        if k not in JOBS:
            raise SystemExit(f'Unknown job {k}')
        if mode == 'forward':
            forward(k)
        elif mode == 'package':
            package(k)
        else:
            raise SystemExit('Usage: pack_zero_backfill.py forward|package <job|all|k1,k2>')
