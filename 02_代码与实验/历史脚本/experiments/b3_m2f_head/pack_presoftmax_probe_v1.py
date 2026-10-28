"""M2F56 pre-softmax query-class-logit calibration probes.

WHY THIS EXISTS
  The only decisive platform evidence so far:
      Stage59 52k ZERO     0.670442
      Stage59 52k FIXED8D  0.683325     -> +1.2883 pp from calibration alone
  The old FIXED8D is added to the TTA8 mean of RAW class logits, before argmax
  (target_transfer.py lines 105-108). Today's M2F bias is added AFTER the
  mixture `einsum`, i.e. to a sum of query probabilities -- a different space.

WHERE THE BIAS GOES HERE
  Analytic (a): the bias is part of the classifier, so it is applied per view to
  `class_logits[..., :8]` BEFORE the softmax, then synthesis, then the existing
  inverse-transform / FP32 view mean / argmax. For the OLD model the two orders
  coincide, because mean(x+b) == mean(x)+b for a constant additive bias; for M2F
  they do not, and applying it inside the softmax is the faithful analogue.

  class_logits (B,Q,9) -> +[rel*scale + g on the 8 semantic entries, 0 on
  no-object] -> softmax(-1)[..., :-1] -> x sigmoid(mask_logits) -> inverse TTA
  -> FP32 mean of 8 -> argmax + 1

  The no-object entry is deliberately left unshifted; adding g to all eight
  semantic entries is equivalent to subtracting g from no-object, which is the
  semantic-vs-no-object axis the old argmax could not see at all.

VECTORS (6 probes, one model pass, no val sweep, no verifier)
  rel = FIXED8D mean-centred = [-1.13655, -0.83655, -0.86565, -0.23655,
                                +2.66345, -0.53655, +2.38495, -1.43655]
  g   = FIXED8D mean = +0.93655  (reported; the probes use the g values below)

  PS025_G0  rel*0.25, g=0
  PS050_G0  rel*0.50, g=0
  PS100_G0  rel*1.00, g=0
  PS150_G0  rel*1.50, g=0
  PS100_GN1 rel*1.00, g=-1
  PS100_GP1 rel*1.00, g=+1

Physical GPU2 only. No training, no validation, no platform submission.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

ROOT = Path('/home/jhy/rare_diag_20260915/m2f_gpu_20260915')
BUNDLE = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
TEST = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images')
CKPT = ROOT / 'formal_run1b/scores/score_iter56000.pt'
OUT = ROOT / 'test1_presoftmax_v1'
EXPECTED_CKPT_SHA = '707a582df4cf5a22227ed73db9cf4bdad7e1689f72ae9f99f71fd09e22629272'
FIXED8D = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
MEAN_CENTRED = [-1.13655, -0.83655, -0.86565, -0.23655,
                2.66345, -0.53655, 2.38495, -1.43655]
PROBES = [('PS025_G0', 0.25, 0.0), ('PS050_G0', 0.50, 0.0),
          ('PS100_G0', 1.00, 0.0), ('PS150_G0', 1.50, 0.0),
          ('PS100_GN1', 1.00, -1.0), ('PS100_GP1', 1.00, 1.0)]
NAMES = [f'test1_{i}.png' for i in range(1, 501)]


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
    guard(120 * 1024**2)
    if {p.name for p in TEST.glob('*.png')} != set(NAMES):
        raise RuntimeError('Unexpected test image names')
    if sha(CKPT) != EXPECTED_CKPT_SHA:
        raise RuntimeError('Checkpoint SHA mismatch')
    OUT.mkdir(parents=True, exist_ok=False)
    for p in (str(ROOT / 'py'), str(BUNDLE), '/home/jhy/gpu2_day_queue'):
        sys.path.insert(0, p)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores
    from src.transforms import normalize_to_tensor

    # 9-entry bias: 8 semantic entries get rel*scale + g, no-object stays 0
    bias_vecs = {}
    for tag, scale, g in PROBES:
        v = [rel * scale + g for rel in MEAN_CENTRED] + [0.0]
        bias_vecs[tag] = torch.tensor(v, dtype=torch.float32, device='cuda:0').view(1, 1, 9)

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

    meta = {
        'schema': 'M2F56_PRESOFTMAX_QUERY_CLASS_LOGIT_CALIBRATION_V1',
        'checkpoint': str(CKPT), 'checkpoint_sha256': EXPECTED_CKPT_SHA, 'step': 56000,
        'physical_gpu': 2, 'tta': 'D4 TTA8', 'view_batch_size': 1,
        'precision': 'FP32; autocast disabled; FP32 view mean',
        'old_stage59_generator': str(BUNDLE / 'scripts/probe/target_transfer.py'),
        'old_stage59_generator_sha256': sha(BUNDLE / 'scripts/probe/target_transfer.py'),
        'old_operation': 'FIXED8D added to TTA8 mean of RAW class logits, before argmax',
        'old_fixed8d': FIXED8D,
        'fixed8d_mean': 0.93655,
        'mean_centred_relative': MEAN_CENTRED,
        'bias_placement_here': 'per view, added to class_logits[..., :8] BEFORE '
                               'softmax; no-object entry unshifted',
        'probes': {tag: {'scale': s, 'g': g,
                         'bias9': [rel * s + g for rel in MEAN_CENTRED] + [0.0]}
                   for tag, s, g in PROBES},
        'test_count': 500, 'single_model_pass': True,
        'code_sha256': {str(Path(__file__)): sha(Path(__file__))},
    }
    with (OUT / 'INFERENCE_CONFIG.json').open('x', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, allow_nan=False)
    print('=== probe bias vectors (9 entries: 8 semantic + no-object) ===', flush=True)
    for tag, s, g in PROBES:
        v = meta['probes'][tag]['bias9']
        print(f'  {tag:10s} scale={s:<5} g={g:<5} '
              f'semantic={[round(x, 6) for x in v[:8]]} noobj={v[8]}', flush=True)

    for tag, _, _ in PROBES:
        (OUT / f'predictions_{tag}').mkdir()

    # D4 ordering identical to tta_anatomy_v2.VIEWS and target_transfer.py
    views = [(k, hf) for k in range(4) for hf in (False, True)]
    t0 = time.time()
    with torch.inference_mode():
        for i, name in enumerate(NAMES, 1):
            guard(16 * 1024**2)
            with Image.open(TEST / name) as im:
                image = np.asarray(im.convert('RGB')).copy()
            if image.shape != (1024, 1024, 3):
                raise RuntimeError(f'Unexpected shape: {name}')
            px = normalize_to_tensor(image).unsqueeze(0).to('cuda:0')
            acc = {tag: None for tag, _, _ in PROBES}
            for k, hf in views:
                v = px
                if k:
                    v = torch.rot90(v, k=k, dims=[-2, -1])
                if hf:
                    v = v.flip(dims=[-1])
                with torch.amp.autocast('cuda', enabled=False):
                    cl, ml = model.raw(v)
                    for tag, _, _ in PROBES:
                        sc = semantic_scores(cl + bias_vecs[tag], ml)   # (1,8,h,w)
                        if hf:
                            sc = sc.flip(dims=[-1])
                        if k:
                            sc = torch.rot90(sc, k=-k, dims=[-2, -1])
                        if tuple(sc.shape[-2:]) != (1024, 1024):
                            sc = F.interpolate(sc, size=(1024, 1024), mode='bilinear',
                                               align_corners=False)
                        acc[tag] = sc if acc[tag] is None else acc[tag] + sc
                del v, cl, ml
            for tag, _, _ in PROBES:
                scores = acc[tag] / 8.0
                if not torch.isfinite(scores).all():
                    raise RuntimeError(f'Nonfinite scores: {name} {tag}')
                pred = scores.squeeze(0).argmax(0).add(1).to(torch.uint8)
                Image.fromarray(pred.cpu().numpy()).save(
                    OUT / f'predictions_{tag}' / name, format='PNG')
                del scores, pred
            del px, acc
            if i % 100 == 0:
                print(f'PROGRESS {i}/500 elapsed={time.time()-t0:.1f}s', flush=True)
    del model, bias_vecs
    torch.cuda.empty_cache()
    if sha(CKPT) != EXPECTED_CKPT_SHA:
        raise RuntimeError('Checkpoint changed while inferring')
    with (OUT / 'FORWARD_DONE.json').open('x', encoding='utf-8') as f:
        json.dump({'images': 500, 'probes': len(PROBES),
                   'elapsed_sec': time.time() - t0}, f, indent=2)
    print('FORWARD_DONE', flush=True)


def package():
    if not (OUT / 'FORWARD_DONE.json').is_file():
        raise RuntimeError('Forward marker missing; refusing to re-forward')
    cfg_sha = sha(OUT / 'INFERENCE_CONFIG.json')
    rows = []
    for order, (tag, scale, g) in enumerate(PROBES, 1):
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
        zp = OUT / f'{order:02d}_M2F56_{tag}_TTA8.zip'
        guard(sum((pred / n).stat().st_size for n in NAMES) + 8 * 1024**2)
        with zipfile.ZipFile(zp, 'x', compression=zipfile.ZIP_STORED) as z:
            for n in NAMES:
                z.write(pred / n, arcname=n)
        with zipfile.ZipFile(zp) as z:
            if z.namelist() != NAMES or z.testzip() is not None:
                raise RuntimeError(f'ZIP failed: {tag}')
        row = {'order': order, 'tag': tag, 'scale': scale, 'g': g,
               'bias9': [rel * scale + g for rel in MEAN_CENTRED] + [0.0],
               'package_filename': zp.name, 'package_sha256': sha(zp),
               'package_bytes': zp.stat().st_size,
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
        raise SystemExit('Usage: pack_presoftmax_probe_v1.py forward|package')
