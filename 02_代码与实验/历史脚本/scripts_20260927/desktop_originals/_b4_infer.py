#!/usr/bin/env python3
"""B4 Phase B: D4 TTA8 ZERO-bias inference on test2.
Single checkpoint: best step53200.
NO Bstar, NO calibration. FP32 view mean.
"""
import sys, os, json, hashlib, time, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
B4_CONFIG = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/b4_probe_config.yaml')
B4_CKPT = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/base_probe/best_miou.pt')
TEST2 = Path('/home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images')
OUT = Path('/home/jhy/rare_diag_20260915/b4_migration_6296/candidate_best53200_zero_20260927')
EXPECTED_CKPT_SHA = 'd7a63b0c1124f5ab213b027125322ace8ce7cf834f48fcfb45e34b20fdef6000'

for p in (str(SNAP), '/home/jhy/gpu2_day_queue'):
    if p not in sys.path: sys.path.insert(0, p)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

# D4 transforms: 8 views
# (rot_k, flip_h)
D4_VIEWS = [
    (0, False),  # 0: identity
    (0, True),   # 1: hflip
    (2, False),  # 2: rot180 (=vflip+hflip)
    (2, True),   # 3: rot180+hflip (=vflip)
    (1, False),  # 4: rot90
    (1, True),   # 5: rot90+hflip
    (3, False),  # 6: rot270
    (3, True),   # 7: rot270+hflip
]

def d4_apply(x, rot_k, flip_h):
    if flip_h: x = x.flip(-1)
    if rot_k: x = torch.rot90(x, rot_k, (-2, -1))
    return x

def d4_inverse(x, rot_k, flip_h):
    if rot_k: x = torch.rot90(x, -rot_k, (-2, -1))
    if flip_h: x = x.flip(-1)
    return x

def main():
    from src.models import build_model as make_model
    from src.utils import load_config
    from src.transforms import normalize_to_tensor

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'predictions').mkdir()

    # Verify checkpoint SHA
    actual_sha = sha256_file(B4_CKPT)
    if actual_sha != EXPECTED_CKPT_SHA:
        raise SystemExit(f'Checkpoint SHA mismatch: expected {EXPECTED_CKPT_SHA}, got {actual_sha}')
    print(f'Checkpoint SHA verified: {actual_sha}', flush=True)

    device = 'cuda:0'
    cfg = load_config(str(B4_CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)

    model = make_model(cfg)
    ckpt = torch.load(str(B4_CKPT), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt), strict=True)
    model = model.to(device).eval().half()
    print('Model loaded', flush=True)

    # Runtime identity
    runtime_id = {
        'gpu_uuid': str(torch.cuda.get_device_properties(0).uuid),
        'gpu_name': torch.cuda.get_device_name(0),
        'checkpoint_sha256': actual_sha,
        'protocol': 'D4_TTA8_ZERO_bias_FP32_mean',
        'num_views': 8,
        'bstar_applied': False,
    }
    (OUT / 'RUNTIME_IDENTITY.json').write_text(json.dumps(runtime_id, indent=2))

    names = sorted([f for f in os.listdir(TEST2) if f.endswith('.png')])
    print(f'Processing {len(names)} images', flush=True)

    t0 = time.time()
    for idx, name in enumerate(names, 1):
        with Image.open(TEST2 / name) as f:
            img = np.asarray(f.convert('RGB'))
        x = normalize_to_tensor(img).unsqueeze(0).to(device)

        # D4 TTA8
        logits_sum = None
        with torch.no_grad():
            for rot_k, flip_h in D4_VIEWS:
                x_v = d4_apply(x, rot_k, flip_h)
                with torch.amp.autocast('cuda', enabled=True):
                    z = model(pixel_values=x_v).logits
                z = z.float()  # FP32
                z = F.interpolate(z, size=(1024, 1024), mode='bilinear', align_corners=False)
                z = d4_inverse(z, rot_k, flip_h)
                if logits_sum is None:
                    logits_sum = z
                else:
                    logits_sum = logits_sum + z

        logits_mean = logits_sum / len(D4_VIEWS)
        pred = logits_mean.argmax(1)[0].cpu().numpy() + 1  # labels 1-8

        out_img = Image.fromarray(pred.astype(np.uint8), mode='L')
        out_img.save(OUT / 'predictions' / name)

        if idx % 100 == 0:
            elapsed = time.time() - t0
            print(f'{idx}/{len(names)} ({elapsed:.1f}s, {elapsed/idx:.2f}s/img)', flush=True)

    elapsed = time.time() - t0
    print(f'Done in {elapsed:.1f}s', flush=True)

    # Zip
    zip_path = OUT / 'B4_STAGE56_6097_BEST53200_D4_ZERO_test2.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted((OUT / 'predictions').iterdir()):
            zf.write(f, f.name)
    zip_sha = sha256_file(zip_path)
    print(f'ZIP: {zip_path} SHA={zip_sha}', flush=True)

    # Package manifest
    manifest = {
        'checkpoint_sha256': actual_sha,
        'zip_path': str(zip_path),
        'zip_sha256': zip_sha,
        'num_pngs': len(names),
        'protocol': 'D4_TTA8_ZERO_bias_FP32_mean',
        'bstar_applied': False,
        'checkpoint_count': 1,
        'platform_submission_performed': False,
        'elapsed_sec': round(elapsed, 1),
    }
    (OUT / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
