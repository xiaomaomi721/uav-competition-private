"""Salvage packaging for stage59_44000, whose 500 predictions completed but whose
post-loop checkpoint re-verification aborted because the checkpoint file was
deleted by the server's own reclamation pipeline mid-run.

Evidence that the predictions are a complete, unmodified run of this checkpoint:
  - the forward log shows PROGRESS 500/500 (all 500 saves issued)
  - the traceback is in `sha(ckpt)` AFTER the loop, not inside it
  - INFERENCE_CONFIG.json records checkpoint_sha256 729a51e3...2490, which is
    byte-identical to the SHA in the inventory report, plus step 44000 and the
    checkpoint's own best_miou metadata
  - the checkpoint SHA was verified BEFORE model load (first statement of forward)

The checkpoint no longer exists on disk, so it cannot be re-hashed. Identity
therefore rests on the pre-run check recorded in INFERENCE_CONFIG.json.

No re-forward. Physical GPU not used. Checker + ZIP only.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
TEST = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images')
CHECKER = SNAP / 'scripts/check_submission.py'
OUT = Path('/home/jhy/rare_diag_20260915/zero_backfill_20260916/stage59_44000')
EXPECTED_CKPT_SHA = '729a51e3091a40c4be9e6926a27bd95b4aeb30a829a4dbbed9be7b1840d52490'
NAMES = [f'test1_{i}.png' for i in range(1, 501)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    cfg = json.loads((OUT / 'INFERENCE_CONFIG.json').read_text())
    if cfg['job'] != 'stage59_44000' or cfg['step'] != 44000:
        raise RuntimeError(f'Unexpected config identity: {cfg["job"]} {cfg["step"]}')
    if cfg['checkpoint_sha256'] != EXPECTED_CKPT_SHA:
        raise RuntimeError('Recorded checkpoint SHA does not match the inventory')
    if cfg['bias'] != [0.0] * 8 or cfg['bias_application'] != 'none (ZERO)':
        raise RuntimeError(f'Not a ZERO package: {cfg["bias"]}')
    if cfg.get('checkpoint_meta', {}).get('step') != 44000:
        raise RuntimeError('Checkpoint metadata step mismatch')

    pred = OUT / 'predictions'
    got = {p.name for p in pred.glob('*.png')}
    if got != set(NAMES):
        raise RuntimeError(f'Expected the full 500, got {len(got)}')

    check_dir = OUT / 'check'
    subprocess.run([sys.executable, str(CHECKER), '--test-images', str(TEST),
                    '--predictions', str(pred), '--output-dir', str(check_dir),
                    '--expected-count', '500'],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    check = json.loads((check_dir / 'submission_check.json').read_text())
    if check['status'] != 'PASS':
        raise RuntimeError('Official checker did not pass')
    print(f"official checker: {check['status']}", flush=True)

    zpath = OUT / 'ZERO_stage59_44000_TTA8.zip'
    with zipfile.ZipFile(zpath, 'x', compression=zipfile.ZIP_STORED) as z:
        for name in NAMES:
            z.write(pred / name, arcname=name)
    with zipfile.ZipFile(zpath) as z:
        if z.namelist() != NAMES or z.testzip() is not None:
            raise RuntimeError('ZIP entry or CRC mismatch')

    row = {
        'job': 'stage59_44000', 'step': 44000, 'bias': [0.0] * 8,
        'checkpoint': cfg['checkpoint'],
        'checkpoint_sha256': cfg['checkpoint_sha256'],
        'checkpoint_sha_evidence': 'verified before model load by '
                                   'pack_zero_backfill.py; recorded in '
                                   'INFERENCE_CONFIG.json; byte-identical to the '
                                   'inventory report SHA',
        'checkpoint_file_now_present': False,
        'salvage_reason': 'forward completed all 500 saves (PROGRESS 500/500); the '
                          'script then aborted in the post-loop `sha(ckpt)` re-check '
                          'because the server reclamation pipeline deleted the '
                          'checkpoint mid-run; no re-forward performed',
        'inference_config_sha256': sha(OUT / 'INFERENCE_CONFIG.json'),
        'package_filename': zpath.name, 'package_sha256': sha(zpath),
        'package_bytes': zpath.stat().st_size, 'official_check': 'PASS',
        'entries': 500, 'platform_miou': None,
    }
    with (OUT / 'PACKAGE.json').open('x', encoding='utf-8') as f:
        json.dump(row, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f'PACKAGE_READY {zpath.name} {row["package_sha256"]} '
          f'{row["package_bytes"]}', flush=True)


if __name__ == '__main__':
    main()
