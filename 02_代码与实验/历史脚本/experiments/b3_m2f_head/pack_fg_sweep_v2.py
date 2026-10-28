"""Five test1 foreground offsets, one M2F56 FP32 D4-TTA8 pass; no validation."""
import csv
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
OUT = ROOT / 'test1_fg_sweep_v2'
EXPECTED_SHA = '707a582df4cf5a22227ed73db9cf4bdad7e1689f72ae9f99f71fd09e22629272'
VECTORS = [(f'FG_POS_{int(round(a * 100)):03d}', [0.0] + [a] * 7)
           for a in (0.15, 0.20, 0.30, 0.40, 0.50)]
NAMES = [f'test1_{i}.png' for i in range(1, 501)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def guard(reserve=0):
    free = shutil.disk_usage(ROOT).free
    if free - reserve < int(1.5 * 1024**3):
        raise RuntimeError(f'Disk floor: free={free}, reserve={reserve}')


def forward():
    if (os.environ.get('CUDA_VISIBLE_DEVICES'), os.environ.get('CUDA_DEVICE_ORDER')) != ('2', 'PCI_BUS_ID'):
        raise RuntimeError('Physical GPU2 only')
    guard(100 * 1024**2)
    if {p.name for p in TEST.glob('*.png')} != set(NAMES):
        raise RuntimeError('Unexpected test image names')
    if sha(CKPT) != EXPECTED_SHA:
        raise RuntimeError('Checkpoint SHA mismatch')
    OUT.mkdir(exist_ok=False)
    for p in (str(ROOT / 'py'), str(BUNDLE), '/home/jhy/gpu2_day_queue'):
        sys.path.insert(0, p)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores
    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import forward_views, VIEWS

    class M2FScores(torch.nn.Module):
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

        def forward(self, pixel_values):
            f = self.encoder(pixel_values=pixel_values, output_hidden_states=True).hidden_states
            o = self.head(f)
            return SimpleNamespace(logits=semantic_scores(o['class_logits'], o['mask_logits']))

    torch.manual_seed(42)
    torch.set_num_threads(4)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    if ck['data_iteration'] != 56000:
        raise RuntimeError('Checkpoint step mismatch')
    c = ck['config']
    if (c['num_queries'], c['hidden_dim'], c['layers']) != (100, 256, 6):
        raise RuntimeError('Checkpoint architecture mismatch')
    model = M2FScores(ck).to('cuda:0').eval()
    del ck, c
    sources = [Path(__file__), ROOT / 'py/b3_m2f_head/head.py',
               ROOT / 'py/b3_m2f_head/semantic.py', BUNDLE / 'src/transforms.py',
               Path('/home/jhy/gpu2_day_queue/tta_anatomy_v2.py'),
               BUNDLE / 'scripts/check_submission.py']
    config = {'checkpoint': str(CKPT), 'checkpoint_sha256': EXPECTED_SHA,
              'step': 56000, 'tta': 'D4-TTA8', 'views': VIEWS, 'view_batch_size': 1,
              'precision': 'FP32; autocast disabled; FP32 view mean',
              'bias_application': 'FP32 vector added after view mean, before argmax + 1',
              'bias_vectors_exact': dict(VECTORS),
              'bias_vectors_hex': {t: [x.hex() for x in v] for t, v in VECTORS},
              'code_sha256': {str(p): sha(p) for p in sources},
              'physical_gpu': 2, 'test_count': 500, 'single_forward_pass': True}
    save(OUT / 'INFERENCE_CONFIG.json', config)
    biases = {}
    for tag, v in VECTORS:
        (OUT / f'predictions_{tag}').mkdir()
        biases[tag] = torch.tensor(v, dtype=torch.float32, device='cuda:0').view(8, 1, 1)
        print(tag, repr(v), [x.hex() for x in v], flush=True)
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
            if not torch.isfinite(scores).all():
                raise RuntimeError(f'Nonfinite scores: {name}')
            for tag, _ in VECTORS:
                pred = (scores + biases[tag]).argmax(0).add(1).to(torch.uint8)
                Image.fromarray(pred.cpu().numpy()).save(OUT / f'predictions_{tag}' / name)
            del px, views, scores, pred
            if i % 100 == 0:
                print(f'PROGRESS {i}/500 elapsed={time.time()-start:.1f}s', flush=True)
    save(OUT / 'FORWARD_DONE.json', {'images': 500, 'vectors': 5, 'elapsed_sec': time.time()-start})
    print('FORWARD_DONE', flush=True)


def package():
    if not (OUT / 'FORWARD_DONE.json').is_file():
        raise RuntimeError('Forward completion marker missing; will not re-forward')
    config_sha = sha(OUT / 'INFERENCE_CONFIG.json')
    rows = []
    for order, (tag, v) in enumerate(VECTORS, 1):
        pred = OUT / f'predictions_{tag}'
        if {p.name for p in pred.glob('*.png')} != set(NAMES):
            raise RuntimeError(f'Incomplete predictions: {tag}')
        check_dir = OUT / f'check_{tag}'
        subprocess.run([sys.executable, str(BUNDLE / 'scripts/check_submission.py'),
                        '--test-images', str(TEST), '--predictions', str(pred),
                        '--output-dir', str(check_dir), '--expected-count', '500'],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
        check = json.loads((check_dir / 'submission_check.json').read_text())
        if check['status'] != 'PASS':
            raise RuntimeError(f'Official checker failed: {tag}')
        zpath = OUT / f'{order:02d}_M2F56_{tag}_TTA8.zip'
        guard(sum((pred / n).stat().st_size for n in NAMES) + 8 * 1024**2)
        with zipfile.ZipFile(zpath, 'x', compression=zipfile.ZIP_STORED) as z:
            for n in NAMES:
                z.write(pred / n, arcname=n)
        with zipfile.ZipFile(zpath) as z:
            if z.namelist() != NAMES or z.testzip() is not None:
                raise RuntimeError(f'ZIP failed: {tag}')
        row = {'order': order, 'tag': tag, 'bias_vector_exact': v,
               'package_filename': zpath.name, 'package_sha256': sha(zpath),
               'checkpoint_sha256': EXPECTED_SHA, 'inference_config_sha256': config_sha,
               'official_check': 'PASS', 'entries': 500, 'platform_miou': None}
        save(OUT / f'PACKAGE_{tag}.json', row)
        rows.append(row)
        print('PACKAGE_READY', zpath.name, row['package_sha256'], flush=True)
    save(OUT / 'PACK_RESULT.json', {'status': 'COMPLETE', 'packages': rows})
    with (OUT / 'PLATFORM_FILL.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('ALL_PACKAGES_DONE', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['forward']:
        forward()
    elif sys.argv[1:] == ['package']:
        package()
    else:
        raise SystemExit('Usage: pack_fg_sweep_v2.py forward|package')
