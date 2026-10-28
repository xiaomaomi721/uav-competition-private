#!/usr/bin/env python3
"""Generate TREAT teacher raw-logit cache over 1300 test2 images (new server).

Produces, per image, native [8,256,256] D4-TTA8 FP32-mean raw logits
(NO Bstar, NO bilinear upsample) — protocol identical to gen_a05_teacher_cache.py,
but the teacher is TREAT (70.8671, checkpoint 1aa73962...) instead of A05.

This cache is the frozen teacher for the round-2 iterative self-training (招2):
    TREAT -> blind-recall soft-KL -> stronger student.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/uav')
SRC = ROOT / 'code/src'
CONFIG = ROOT / 'code/config.yaml'
MODEL_SOURCE = ROOT / 'models/segformer-b3-ade512'
TEST2 = ROOT / 'datasets/test_2/images'
TEST1 = ROOT / 'datasets/训练集/images'
A05 = ROOT / 'checkpoints/A05_WEIGHT_alpha05.pt'
TREAT = ROOT / 'checkpoints/TREAT_blindrecall_step2000.pt'
ARMB = ROOT / 'runs/formal_armB/checkpoint_step2000.pt'
ARMA = ROOT / 'runs/formal_armA/checkpoint_step2000.pt'
EXPECTED_A05_SHA = '023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134'
EXPECTED_TREAT_SHA = '1aa7396249d963d17fbf3664dad7e787360ef3105b315abb8101e21c15807c48'
EXPECTED_ARMB_SHA = 'f186e06ddd531724f647757587e5afec1415fb672410534de9658be60e3c58b9'
EXPECTED_ARMA_SHA = '61f586de60ec7e9dc0e66e808336147af6a21f3efbd5b1f4cd4d8f9097bf5632'
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]

# teacher selection: single checkpoint, or offline weight-averaged (SWA, compliant)
#   armB  -> Arm B (71.045315, current champion)
#   treat -> TREAT (70.8671)
#   swa_treat_armb -> 0.5*TREAT + 0.5*ArmB
#   swa_armb_arma  -> 0.5*ArmB + 0.5*ArmA
TEACHER_SPECS = {
    'armb': {'sha': EXPECTED_ARMB_SHA, 'path': ARMB, 'score': 71.045315},
    'treat': {'sha': EXPECTED_TREAT_SHA, 'path': TREAT, 'score': 70.8671},
    'arma': {'sha': EXPECTED_ARMA_SHA, 'path': ARMA, 'score': None},
    'swa_treat_armb': {'sha': None, 'components': [('treat', 0.5), ('armb', 0.5)]},
    'swa_armb_arma': {'sha': None, 'components': [('armb', 0.5), ('arma', 0.5)]},
}
CKPT_PATHS = {
    'a05': A05, 'treat': TREAT, 'armb': ARMB, 'arma': ARMA,
}
CKPT_SHAS = {
    'a05': EXPECTED_A05_SHA, 'treat': EXPECTED_TREAT_SHA,
    'armb': EXPECTED_ARMB_SHA, 'arma': EXPECTED_ARMA_SHA,
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_state(path):
    obj = torch.load(str(path), map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        for k in ('model', 'student_model', 'model_state_dict', 'state_dict'):
            v = obj.get(k)
            if isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
                return v
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError('no state')


def verify_gpu(gpu_id):
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu_id):
        raise SystemExit(f'GPU{gpu_id} required: CUDA_VISIBLE_DEVICES={gpu_id}')
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    m = [x for x in rows if x.strip().startswith(f'{gpu_id},')]
    if not m:
        raise SystemExit(f'GPU{gpu_id} not found: {rows}')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('expected exactly one visible CUDA device')


def build_teacher_state(teacher_key):
    """Return (state_dict, meta_desc) for the chosen teacher.

    Single checkpoints are SHA-verified. SWA entries average component state
    dicts offline (compliant weight averaging) and have no SHA (derived).
    """
    spec = TEACHER_SPECS[teacher_key]
    if 'components' not in spec:
        path = CKPT_PATHS[teacher_key]
        expect = CKPT_SHAS[teacher_key]
        if sha256_file(path) != expect:
            raise SystemExit(f'{teacher_key} SHA mismatch')
        sd = load_state(path)
        return sd, f'{teacher_key} ({spec.get("score")})'
    # SWA: average component state dicts in float
    comps = []
    for ck, w in spec['components']:
        path = CKPT_PATHS[ck]
        if sha256_file(path) != CKPT_SHAS[ck]:
            raise SystemExit(f'{ck} SHA mismatch')
        comps.append((load_state(path), w))
    keys = list(comps[0][0].keys())
    out_sd = {}
    for k in keys:
        acc = None
        for sd, w in comps:
            t = sd[k].float()
            acc = w * t if acc is None else acc + w * t
        out_sd[k] = acc
    return out_sd, f'swa({teacher_key})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--teacher', default='armb', choices=list(TEACHER_SPECS.keys()))
    ap.add_argument('--dataset', default='test2', choices=['test1', 'test2'])
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int, default=0)
    a = ap.parse_args()
    verify_gpu(a.gpu)
    for p in (str(ROOT / 'code'), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ['UAVSEG_MODEL_SOURCE'] = str(MODEL_SOURCE)
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    from src.models import build_model as mm
    from src.utils import load_config
    from src.transforms import normalize_to_tensor
    from PIL import Image

    out = Path(a.output_dir)
    scores = out / 'scores'
    scores.mkdir(parents=True, exist_ok=False)

    teacher_state, teacher_desc = build_teacher_state(a.teacher)

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    cfg['pretrained_model_path'] = str(MODEL_SOURCE)
    model = mm(cfg)
    model.load_state_dict(teacher_state, strict=True)
    model = model.to(a.device).eval()

    img_dir = TEST1 if a.dataset == 'test1' else TEST2
    total = 500 if a.dataset == 'test1' else 1300
    end = a.end if a.end > 0 else total
    names = [f'{a.dataset}_{i}.png' for i in range(a.start, end + 1)]
    start = time.time()
    manifest = {}
    with torch.inference_mode():
        for idx, name in enumerate(names, 1):
            with Image.open(img_dir / name) as im:
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
        'teacher': a.teacher,
        'teacher_desc': teacher_desc,
        'dataset': a.dataset,
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
