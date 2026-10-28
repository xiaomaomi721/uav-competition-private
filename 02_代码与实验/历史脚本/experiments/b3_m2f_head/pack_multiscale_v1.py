"""Stage 1: true input-scale multi-scale D4-TTA8 on M2F56, ZERO bias.

DEFINITION (frozen)
  This is INPUT-SCALE TTA, not output-map resizing. For each scale s the RGB
  test image (1024x1024) is genuinely resized to s*1024 and pushed through a full
  D4-TTA8 pass; the resulting native semantic-score tensor is inverse-D4'd and
  bilinearly restored to the 1024x1024 frame. Scores are fused only in that
  common 1024 frame -- never PNGs, never majority vote.

      s = 0.75 -> 768x768
      s = 1.00 -> 1024x1024
      s = 1.25 -> 1280x1280

  per image, per scale:
      RGB_1024 -> resize to s*1024 -> [8 D4 views: model -> semantic_scores ->
      inverse hf/k -> bilinear to 1024] -> FP32 mean of 8  ->  S_s

  six ZERO-bias packages:
      A_S075            S075
      B_S125            S125
      C_S075_S100       (S075 + S100) / 2
      D_S100_S125       (S100 + S125) / 2
      E_S075_S100_S125  (S075 + S100 + S125) / 3
      F_S075_2S100_S125 (S075 + 2*S100 + S125) / 4

  S100 alone is NOT packaged: its platform ZERO score already exists (0.684133).
  It is still computed, and used as a hard drift guard -- its argmax must
  reproduce the known test1 ZERO predictions from test1_zero_run1b_v1.

Physical GPU2 only. No training, no validation, no calibration, no platform
submission.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path('/home/jhy/rare_diag_20260915/m2f_gpu_20260915')
BUNDLE = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
TEST = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images')
CKPT = ROOT / 'formal_run1b/scores/score_iter56000.pt'
REF_ZERO = ROOT / 'test1_zero_run1b_v1/predictions_56k'
OUT = ROOT / 'test1_multiscale_v1'
EXPECTED_CKPT_SHA = '707a582df4cf5a22227ed73db9cf4bdad7e1689f72ae9f99f71fd09e22629272'
SCALES = (0.75, 1.00, 1.25)
PACKAGES = [
    ('A_S075', ('S075',)),
    ('B_S125', ('S125',)),
    ('C_S075_S100', ('S075', 'S100')),
    ('D_S100_S125', ('S100', 'S125')),
    ('E_S075_S100_S125', ('S075', 'S100', 'S125')),
    ('F_S075_2S100_S125', ('S075', 'S100', 'S100', 'S125')),
]
NAMES = [f'test1_{i}.png' for i in range(1, 501)]
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def guard(reserve=0):
    free = shutil.disk_usage(ROOT).free
    if free - reserve < int(1.5 * 1024**3):
        raise RuntimeError(f'Disk floor: free={free} reserve={reserve}')


def forward():
    if (os.environ.get('CUDA_VISIBLE_DEVICES'), os.environ.get('CUDA_DEVICE_ORDER')) != ('2', 'PCI_BUS_ID'):
        raise RuntimeError('Physical GPU2 only')
    guard(200 * 1024**2)
    if {p.name for p in TEST.glob('*.png')} != set(NAMES):
        raise RuntimeError('Unexpected test image names')
    if sha(CKPT) != EXPECTED_CKPT_SHA:
        raise RuntimeError('Checkpoint SHA mismatch')
    OUT.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(BUNDLE))
    sys.path.insert(0, str(ROOT / 'py'))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores
    from src.transforms import normalize_to_tensor

    class M2FRaw(torch.nn.Module):
        def __init__(self, ck):
            super().__init__()
            self.encoder = AutoModelForSemanticSegmentation.from_pretrained(
                '/home/jhy/models/segformer-b3-ade512', num_labels=8,
                ignore_mismatched_sizes=True, local_files_only=True).segformer
            self.head = B3M2FStyleHead(in_channels=(64, 128, 320, 512), num_classes=8,
                                       hidden_dim=256, num_queries=100, nhead=8,
                                       num_layers=6, dim_feedforward=1024, ignore_index=255)
            self.encoder.load_state_dict(ck['model'], strict=True)
            self.head.load_state_dict(ck['head'], strict=True)

        def raw(self, pixel_values):
            f = self.encoder(pixel_values=pixel_values,
                             output_hidden_states=True).hidden_states
            o = self.head(f)
            return o['class_logits'], o['mask_logits']

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    if ck['data_iteration'] != 56000:
        raise RuntimeError('Checkpoint step mismatch')
    c = ck['config']
    if (c['num_queries'], c['hidden_dim'], c['layers']) != (100, 256, 6):
        raise RuntimeError('Checkpoint architecture mismatch')
    model = M2FRaw(ck).to('cuda:0').eval()
    del ck, c

    with (OUT / 'INFERENCE_CONFIG.json').open('x', encoding='utf-8') as f:
        json.dump({
            'schema': 'M2F56_MULTISCALE_TTA8_V1',
            'checkpoint': str(CKPT), 'checkpoint_sha256': EXPECTED_CKPT_SHA, 'step': 56000,
            'physical_gpu': 2, 'precision': 'FP32; autocast disabled; FP32 view mean',
            'bias': [0.0] * 8, 'bias_application': 'none (ZERO)',
            'multiscale_kind': 'TRUE INPUT-SCALE TTA (RGB resized, not output resampling)',
            'scales': {str(s): int(round(1024 * s)) for s in SCALES},
            'resize_filter': 'PIL BILINEAR on RGB before normalize_to_tensor',
            'per_scale_pipeline': 'native semantic_scores -> inverse hf/k -> bilinear to '
                                  '1024x1024 -> FP32 mean of the 8 D4 views',
            'fusion': 'FP32 score tensors in the common 1024 frame; never PNGs, '
                      'never majority vote',
            'packages': {t: list(v) for t, v in PACKAGES},
            's100_reference_platform_miou': 0.684133,
            'drift_guard': f'S100 argmax must match {REF_ZERO}',
            'test_count': 500, 'single_model_pass': True,
            'code_sha256': {str(Path(__file__)): sha(Path(__file__))},
        }, f, indent=2, ensure_ascii=False, allow_nan=False)

    for tag, _ in PACKAGES:
        (OUT / f'predictions_{tag}').mkdir()

    ref = {p.name: p for p in REF_ZERO.glob('*.png')} if REF_ZERO.is_dir() else {}
    if not ref:
        print('WARNING: reference ZERO predictions not found; drift guard skipped', flush=True)
    drift_diff = 0
    t0 = time.time()
    with torch.inference_mode():
        for i, name in enumerate(NAMES, 1):
            guard(32 * 1024**2)
            with Image.open(TEST / name) as im:
                image = np.asarray(im.convert('RGB')).copy()
            if image.shape != (1024, 1024, 3):
                raise RuntimeError(f'Unexpected shape: {name}')
            S = {}
            for s in SCALES:
                n = int(round(1024 * s))
                arr = image if n == 1024 else np.asarray(
                    Image.fromarray(image).resize((n, n), Image.BILINEAR))
                px = normalize_to_tensor(arr).unsqueeze(0).to('cuda:0')
                acc = None
                for k, hf in VIEWS:
                    v = px
                    if k:
                        v = torch.rot90(v, k=k, dims=[-2, -1])
                    if hf:
                        v = v.flip(dims=[-1])
                    with torch.amp.autocast('cuda', enabled=False):
                        cl, ml = model.raw(v)
                        sc = semantic_scores(cl, ml)
                    if hf:
                        sc = sc.flip(dims=[-1])
                    if k:
                        sc = torch.rot90(sc, k=-k, dims=[-2, -1])
                    if tuple(sc.shape[-2:]) != (1024, 1024):
                        sc = F.interpolate(sc, size=(1024, 1024), mode='bilinear',
                                           align_corners=False)
                    acc = sc if acc is None else acc + sc
                    del v, cl, ml, sc
                S[s] = acc / 8.0
                del px, acc
                torch.cuda.empty_cache()
            key = {0.75: 'S075', 1.00: 'S100', 1.25: 'S125'}
            Sm = {key[s]: S[s] for s in SCALES}
            # drift guard: S100 alone must reproduce the known platform ZERO
            if ref:
                p100 = Sm['S100'].squeeze(0).argmax(0).add(1).to(torch.uint8).cpu().numpy()
                with Image.open(ref[name]) as rim:
                    drift_diff += int((np.asarray(rim) != p100).sum())
            for tag, parts in PACKAGES:
                sc = Sm[parts[0]]
                for extra in parts[1:]:
                    sc = sc + Sm[extra]
                sc = sc / float(len(parts))
                if not torch.isfinite(sc).all():
                    raise RuntimeError(f'Nonfinite scores: {name} {tag}')
                pred = sc.squeeze(0).argmax(0).add(1).to(torch.uint8)
                Image.fromarray(pred.cpu().numpy()).save(
                    OUT / f'predictions_{tag}' / name, format='PNG')
                del sc, pred
            del S, Sm
            if i % 50 == 0:
                print(f'PROGRESS {i}/500 elapsed={time.time()-t0:.0f}s '
                      f'drift_diff_px={drift_diff}', flush=True)
    del model
    torch.cuda.empty_cache()
    if sha(CKPT) != EXPECTED_CKPT_SHA:
        raise RuntimeError('Checkpoint changed while inferring')
    total_px = len(NAMES) * 1024 * 1024
    print(f'DRIFT_GUARD s100 vs known ZERO: differing pixels = {drift_diff} '
          f'of {total_px} ({drift_diff/total_px*100:.6f}%)', flush=True)
    with (OUT / 'FORWARD_DONE.json').open('x', encoding='utf-8') as f:
        json.dump({'images': 500, 'scales': list(SCALES), 'packages': len(PACKAGES),
                   's100_vs_reference_differing_pixels': drift_diff,
                   's100_vs_reference_total_pixels': total_px,
                   'elapsed_sec': time.time() - t0}, f, indent=2)
    print('FORWARD_DONE', flush=True)


def package():
    if not (OUT / 'FORWARD_DONE.json').is_file():
        raise RuntimeError('Forward marker missing; refusing to re-forward')
    cfg_sha = sha(OUT / 'INFERENCE_CONFIG.json')
    rows = []
    for order, (tag, parts) in enumerate(PACKAGES, 1):
        pred = OUT / f'predictions_{tag}'
        if {p.name for p in pred.glob('*.png')} != set(NAMES):
            raise RuntimeError(f'Incomplete predictions: {tag}')
        check_dir = OUT / f'check_{tag}'
        subprocess.run([sys.executable, str(BUNDLE / 'scripts/check_submission.py'),
                        '--test-images', str(TEST), '--predictions', str(pred),
                        '--output-dir', str(check_dir), '--expected-count', '500'],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
        st = json.loads((check_dir / 'submission_check.json').read_text())['status']
        if st != 'PASS':
            raise RuntimeError(f'Official checker failed: {tag}')
        zp = OUT / f'{order:02d}_M2F56_{tag}_TTA8_ZERO.zip'
        guard(sum((pred / n).stat().st_size for n in NAMES) + 8 * 1024**2)
        with zipfile.ZipFile(zp, 'x', compression=zipfile.ZIP_STORED) as z:
            for n in NAMES:
                z.write(pred / n, arcname=n)
        with zipfile.ZipFile(zp) as z:
            if z.namelist() != NAMES or z.testzip() is not None:
                raise RuntimeError(f'ZIP failed: {tag}')
        row = {'order': order, 'tag': tag, 'components': list(parts),
               'bias': [0.0] * 8, 'package_filename': zp.name,
               'package_sha256': sha(zp), 'package_bytes': zp.stat().st_size,
               'checkpoint_sha256': EXPECTED_CKPT_SHA,
               'inference_config_sha256': cfg_sha,
               'official_check': 'PASS', 'entries': 500, 'platform_miou': None}
        with (OUT / f'PACKAGE_{tag}.json').open('x', encoding='utf-8') as f:
            json.dump(row, f, indent=2, ensure_ascii=False, allow_nan=False)
        rows.append(row)
        print(f'PACKAGE_READY {zp.name} {row["package_sha256"]}', flush=True)
    with (OUT / 'PACK_RESULT.json').open('x', encoding='utf-8') as f:
        json.dump({'status': 'COMPLETE', 'packages': rows}, f,
                  indent=2, ensure_ascii=False, allow_nan=False)
    print('ALL_PACKAGES_DONE', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['forward']:
        forward()
    elif sys.argv[1:] == ['package']:
        package()
    else:
        raise SystemExit('Usage: pack_multiscale_v1.py forward|package')
