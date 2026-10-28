#!/usr/bin/env python3
"""Package a round-2 (TREAT-teacher) checkpoint into a compliant test2 ZIP (new server).

Protocol identical to scored A05: single checkpoint, D4 TTA8, FP32 native-grid
mean, Bstar once, bilinear 1024 align_corners=False, argmax+1, PIL L, labels 1-8.
"""
import os, sys, hashlib, time, zipfile
from pathlib import Path
import numpy as np

ROOT = Path('/data/uav')
SRC = ROOT / 'code/src'
CONFIG = ROOT / 'code/config.yaml'
MODEL_SOURCE = ROOT / 'models/segformer-b5-ade640'
TEST2 = ROOT / 'datasets/test_2/images'
BSTAR = np.asarray([-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5], np.float32)
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
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


def verify_gpu(gpu_id):
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu_id):
        raise SystemExit(f'GPU{gpu_id} required')
    import subprocess
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    m = [x for x in rows if x.strip().startswith(f'{gpu_id},')]
    if not m:
        raise SystemExit(f'GPU{gpu_id} not found: {rows}')
    import torch as _t
    if not _t.cuda.is_available() or _t.cuda.device_count() != 1:
        raise SystemExit('expected one visible CUDA device')


def main():
    for p in (str(ROOT / 'code'), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ['UAVSEG_MODEL_SOURCE'] = str(MODEL_SOURCE)
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    import argparse
    import torch, torch.nn.functional as F
    from PIL import Image
    from src.models import build_model as mm
    from src.utils import load_config
    from src.transforms import normalize_to_tensor

    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    verify_gpu(a.gpu)

    ckpt = Path(a.checkpoint)
    out = Path(a.output_dir)
    pred_dir = out / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=False)

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    cfg['pretrained_model_path'] = str(MODEL_SOURCE)
    model = mm(cfg)
    model.load_state_dict(load_state(ckpt), strict=True)
    model = model.to(a.device).eval()
    bias = torch.from_numpy(BSTAR).to(a.device)[:, None, None]

    names = [f'test2_{i}.png' for i in range(1, 1301)]
    start = time.time()
    with torch.inference_mode():
        for idx, name in enumerate(names, 1):
            with Image.open(TEST2 / name) as im:
                rgb = np.asarray(im.convert('RGB')).copy()
            px = normalize_to_tensor(rgb).unsqueeze(0).to(a.device)
            acc = None
            for r, hf in VIEWS:
                t = px
                if r: t = torch.rot90(t, r, (-2, -1))
                if hf: t = t.flip((-1,))
                logits = model(pixel_values=t).logits.float()
                if hf: logits = logits.flip((-1,))
                if r: logits = torch.rot90(logits, -r, (-2, -1))
                canon = logits[0].contiguous()
                acc = canon if acc is None else acc + canon
            raw = acc / 8.0
            dep = F.interpolate((raw + bias)[None], size=(1024, 1024), mode='bilinear', align_corners=False)[0]
            pred = dep.argmax(0).byte().cpu().numpy()
            Image.fromarray(pred + 1, mode='L').save(pred_dir / name)
            if idx % 200 == 0:
                print(f'{idx}/1300 {time.time()-start:.0f}s', flush=True)

    zip_path = out / f'{a.tag}_test2.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.write(pred_dir / name, name)

    with zipfile.ZipFile(zip_path) as zf:
        assert zf.testzip() is None
        assert len(zf.namelist()) == 1300
        for name in names:
            with zf.open(name) as f:
                im = Image.open(f)
                im.load()
                arr = np.asarray(im)
                assert im.mode == 'L' and im.size == (1024, 1024)
                assert arr.dtype == np.uint8 and arr.min() >= 1 and arr.max() <= 8

    result = {
        'checkpoint': str(ckpt),
        'checkpoint_sha256': sha256_file(ckpt),
        'zip': str(zip_path),
        'zip_sha256': sha256_file(zip_path),
        'count': 1300,
        'single_model': True, 'checkpoint_count': 1, 'ensemble': False,
        'teacher_used_for_candidate_inference': False,
        'test2_ground_truth_used': False,
        'platform_submission_performed': False,
        'runtime_sec': round(time.time() - start, 1),
    }
    import json
    (out / 'PACKAGE_MANIFEST.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
