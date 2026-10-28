#!/usr/bin/env python3
"""Generate A05 teacher raw-logit cache over 1300 test2 images.

Produces, per image, native [8,256,256] D4-TTA8 FP32-mean raw logits
(NO Bstar, NO bilinear upsample) — byte-compatible with the existing
cache_261_test2_raw / cache_278_test2_raw protocol. This cache is the
frozen teacher for the blind-recall self-training pair.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/jhy/rare_diag_20260915')
SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
CONFIG = SNAP / 'stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml'
A05 = Path('/home/jhy/rare_diag_20260915/breakthrough_v2_20260926/03_weight_projection/A05_WEIGHT_alpha05.pt')
TEST2 = Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')
EXPECTED_A05_SHA = '023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134'
EXPECTED_GPU_UUID = 'GPU-cacb7500-f2e9-c482-d85c-53c7f8d1def9'
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_state(path):
    import torch
    obj = torch.load(str(path), map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        for k in ('model', 'student_model', 'model_state_dict', 'state_dict'):
            v = obj.get(k)
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
                return v
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError('no state')


def verify_gpu2():
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise SystemExit('GPU2 required: CUDA_VISIBLE_DEVICES=2')
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    m = [x for x in rows if x.strip().startswith('2,')]
    if not m or EXPECTED_GPU_UUID not in m[0]:
        raise SystemExit(f'GPU2 UUID mismatch: {rows}')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('expected exactly one visible CUDA device')
    dev_uuid = torch.cuda.get_device_properties(0).uuid
    if not str(dev_uuid).startswith('GPU-'):
        dev_uuid = 'GPU-' + str(dev_uuid)
    if dev_uuid != EXPECTED_GPU_UUID:
        raise SystemExit(f'PyTorch device UUID mismatch: {dev_uuid}')


def main():
    verify_gpu2()
    for p in (str(SNAP), str(ROOT / 'round2_multiprobe_20260922')):
        if p not in sys.path:
            sys.path.insert(0, p)
    from src.models import build_model as mm
    from src.utils import load_config
    from src.transforms import normalize_to_tensor
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=1300)
    a = ap.parse_args()
    out = Path(a.output_dir)
    scores = out / 'scores'
    scores.mkdir(parents=True, exist_ok=False)

    if sha256_file(A05) != EXPECTED_A05_SHA:
        raise SystemExit('A05 SHA mismatch')

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    model = mm(cfg)
    model.load_state_dict(load_state(A05), strict=True)
    model = model.to(a.device).eval()

    names = [f'test2_{i}.png' for i in range(a.start, a.end + 1)]
    start = time.time()
    manifest = {}
    with torch.inference_mode():
        for idx, name in enumerate(names, 1):
            with Image.open(TEST2 / name) as im:
                rgb = np.asarray(im.convert('RGB')).copy()
            px = normalize_to_tensor(rgb).unsqueeze(0).to(a.device)
            acc = None
            for r, hf in VIEWS:
                t = px
                if r:
                    t = torch.rot90(t, r, (-2, -1))
                if hf:
                    t = t.flip((-1,))
                with torch.amp.autocast('cuda', enabled=False):
                    logits = model(pixel_values=t).logits.float()
                if hf:
                    logits = logits.flip((-1,))
                if r:
                    logits = torch.rot90(logits, -r, (-2, -1))
                canon = logits[0].contiguous()
                acc = canon if acc is None else acc + canon
            raw = (acc / 8.0).cpu().numpy()  # [8,256,256] FP32, no Bstar
            assert raw.shape == (8, 256, 256), f'{name}: {raw.shape}'
            assert np.isfinite(raw).all(), f'{name}: non-finite'
            stem = name[:-4]
            np.save(scores / f'{stem}.npy', raw)
            manifest[stem] = hashlib.sha256(raw.tobytes()).hexdigest()
            if idx % 200 == 0:
                print(f'{idx}/{len(names)} {time.time()-start:.0f}s', flush=True)

    meta = {
        'teacher_checkpoint': str(A05),
        'teacher_checkpoint_sha256': EXPECTED_A05_SHA,
        'count': len(names),
        'protocol': 'D4 TTA8, FP32 view mean, no Bstar, no bilinear upsample',
        'native_grid': 256,
        'num_classes': 8,
        'views': VIEWS,
        'bstar_absent': True,
        'dtype': 'float32',
        'runtime_sec': round(time.time() - start, 1),
    }
    (out / 'META.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'MANIFEST.json').write_text(json.dumps({'count': len(manifest), 'per_file_sha256': manifest}, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
