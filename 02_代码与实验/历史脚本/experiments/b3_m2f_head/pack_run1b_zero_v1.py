"""Run1b: three single-checkpoint test1 ZERO packages, 56k -> 40k -> 20k.
No training, calibration, multi-checkpoint fusion or platform submission.
"""
from __future__ import annotations

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
OUT = ROOT / 'test1_zero_run1b_v1'
FLOOR = int(1.5 * 1024**3)
STEPS = (56000, 40000, 20000)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def disk_guard(reserve=0):
    free = shutil.disk_usage(ROOT).free
    if free - reserve < FLOOR:
        raise RuntimeError(f'disk floor: free={free} reserve={reserve} floor={FLOOR}')


def save_json(path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('Only PCI_BUS_ID physical GPU2 / logical cuda:0 allowed')
    if OUT.exists():
        raise RuntimeError(f'Refuse existing output: {OUT}')
    disk_guard(128 * 1024**2)
    for step in STEPS:
        p = ROOT / 'eval_run1b' / f'val_eval_{step}.json'
        d = json.loads(p.read_text())
        if d['checkpoint_data_iteration'] != step or d['protocol']['n_val'] != 1399 or d['morphology']['images_used'] != 1399:
            raise RuntimeError(f'Incomplete validation: {p}')
    names = sorted([p.name for p in TEST.glob('*.png')], key=lambda n: int(Path(n).stem.split('_')[-1]))
    if set(names) != {f'test1_{i}.png' for i in range(1, 501)} or len(names) != 500:
        raise RuntimeError('Test1 names must be exactly test1_1.png .. test1_500.png')
    head_path = ROOT / 'py/b3_m2f_head/head.py'
    prereg = json.loads((ROOT / 'PREREG_M2F_Run1.json').read_text())
    if sha(head_path) != prereg['head_sha256']:
        raise RuntimeError('Head SHA differs from preregistration')
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
                hidden_dim=256, num_queries=100, nhead=8, num_layers=6,
                dim_feedforward=1024, ignore_index=255)
            self.encoder.load_state_dict(ck['model'], strict=True)
            self.head.load_state_dict(ck['head'], strict=True)

        def forward(self, pixel_values):
            features = self.encoder(pixel_values=pixel_values, output_hidden_states=True).hidden_states
            o = self.head(features)
            # The shared D4 engine names this field logits; for M2F it is the
            # native semantic mixture, not log(score) or a class-softmax of it.
            return SimpleNamespace(logits=semantic_scores(o['class_logits'], o['mask_logits']))

    torch.manual_seed(42)
    torch.set_num_threads(4)
    OUT.mkdir()
    files = [Path(__file__), head_path, ROOT / 'py/b3_m2f_head/semantic.py',
             BUNDLE / 'src/transforms.py', Path('/home/jhy/gpu2_day_queue/tta_anatomy_v2.py'),
             BUNDLE / 'scripts/check_submission.py', ROOT / 'formal_run1b/run/RUN_META.json',
             ROOT / 'PREREG_M2F_Run1.json', ROOT / 'PREREG_M2F_Run1b.json',
             Path('/home/jhy/models/segformer-b3-ade512/config.json')]
    sources = {str(p): sha(p) for p in files}
    protocol = {
        'schema': 'M2F_RUN1B_TEST1_ZERO_V1', 'physical_gpu': 2, 'logical_device': 'cuda:0',
        'test_dir': str(TEST), 'test_count': 500,
        'test_names_sha256': hashlib.sha256(('\n'.join(names) + '\n').encode()).hexdigest(),
        'tta': 'D4 TTA8', 'views': VIEWS, 'view_batch_size': 1,
        'precision': 'FP32, autocast disabled; FP32 view mean',
        'normalization': 'frozen src.transforms.normalize_to_tensor',
        'semantic_synthesis': 'softmax(class)[...,:8] x sigmoid(mask), einsum bqc,bqhw->bchw',
        'order': 'per-view mixture -> inverse mirror -> inverse rotation -> bilinear to original size -> FP32 mean of 8 -> argmax + 1',
        'resize': {'mode': 'bilinear', 'align_corners': False, 'size': 'original image (1024,1024)'},
        'calibration_offset': [0.0] * 8, 'postprocessing': None,
        'single_checkpoint_per_package': True, 'checkpoint_order': list(STEPS),
        'training': False, 'cwfa_inference': False,
        'historical_reference': '/home/jhy/rare_diag_20260915/packages_stage23_v1/RUN_META.json',
        'reference_note': 'Same D4/normalization/FP32 zero protocol; M2F native mixture replaces SegFormer logits. Validation is single-view AMP, NOT this test1 protocol.',
        'code_and_provenance_sha256': sources,
        'environment': {'python': sys.version, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)},
    }
    save_json(OUT / 'INFERENCE_CONFIG.json', protocol)
    config_sha = sha(OUT / 'INFERENCE_CONFIG.json')
    rows = []
    t0 = time.time()
    for ordinal, step in enumerate(STEPS, 1):
        checkpoint = ROOT / 'formal_run1b/scores' / f'score_iter{step}.pt'
        ck_sha = sha(checkpoint)
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck['data_iteration'] != step:
            raise RuntimeError('Checkpoint step mismatch')
        c = ck['config']
        if (c['num_queries'], c['hidden_dim'], c['layers']) != (100, 256, 6):
            raise RuntimeError('Unexpected checkpoint architecture')
        model = M2FScores(ck).to('cuda:0').eval()
        del ck
        tag = f'{step // 1000}k'
        pred_dir = OUT / f'predictions_{tag}'
        pred_dir.mkdir()
        print(f'START {tag} sha={ck_sha}', flush=True)
        with torch.inference_mode():
            for i, name in enumerate(names, 1):
                disk_guard(8 * 1024**2)
                with Image.open(TEST / name) as image:
                    img = np.asarray(image.convert('RGB')).copy()
                if img.shape != (1024, 1024, 3):
                    raise RuntimeError(f'Unexpected image shape {name}: {img.shape}')
                px = normalize_to_tensor(img).unsqueeze(0).to('cuda:0')
                with torch.amp.autocast('cuda', enabled=False):
                    views = forward_views(model, px, 1)
                    scores = views.mean(dim=0, dtype=torch.float32)
                if not torch.isfinite(scores).all():
                    raise RuntimeError(f'Nonfinite scores: {name}')
                pred = scores.argmax(0).add(1).to(torch.uint8).cpu().numpy()
                Image.fromarray(pred).save(pred_dir / name, format='PNG')
                del px, views, scores, pred
                if i % 50 == 0:
                    print(f'PROGRESS {tag} {i}/500 elapsed={time.time()-t0:.0f}s', flush=True)
        del model
        torch.cuda.empty_cache()
        if sha(checkpoint) != ck_sha:
            raise RuntimeError('Checkpoint changed while inferring')
        for path, digest in sources.items():
            if sha(path) != digest:
                raise RuntimeError(f'Source changed while inferring: {path}')
        check_dir = OUT / f'check_{tag}'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        subprocess.run([sys.executable, str(BUNDLE / 'scripts/check_submission.py'),
            '--test-images', str(TEST), '--predictions', str(pred_dir),
            '--output-dir', str(check_dir), '--expected-count', '500'], env=env, check=True)
        check = json.loads((check_dir / 'submission_check.json').read_text())
        if check['status'] != 'PASS':
            raise RuntimeError('Official check did not pass')
        package = OUT / f'{ordinal:02d}_M2F_Run1b_{tag}_TTA8_ZERO.zip'
        disk_guard(sum(p.stat().st_size for p in pred_dir.iterdir()) + 8 * 1024**2)
        with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_STORED) as z:
            for name in names:
                z.write(pred_dir / name, arcname=name)
        with zipfile.ZipFile(package) as z:
            if z.namelist() != names or z.testzip() is not None:
                raise RuntimeError('ZIP entry or CRC mismatch')
            for name in names:
                if z.read(name) != (pred_dir / name).read_bytes():
                    raise RuntimeError(f'ZIP content mismatch {name}')
        row = {'suggested_submission_order': ordinal, 'package_filename': package.name,
            'package_sha256': sha(package), 'package_bytes': package.stat().st_size,
            'checkpoint_path': str(checkpoint), 'checkpoint_sha256': ck_sha,
            'checkpoint_step': step, 'inference_script_sha256': sources[str(Path(__file__))],
            'inference_config_sha256': config_sha, 'calibration': 'ZERO [0,0,0,0,0,0,0,0]',
            'tta': 'D4 TTA8 FP32', 'official_check': 'PASS', 'entries': 500,
            'platform_submission_id': None, 'actual_submission_order': None,
            'platform_filename': None, 'platform_mIoU': None, 'platform_evidence': None,
            'submission_status': 'NOT_SUBMITTED_BY_ASSISTANT', 'binding_grade': 'UNBOUND'}
        save_json(OUT / f'PACKAGE_{tag}.json', row)
        rows.append(row)
        print(f'PACKAGE_READY {package.name} sha={row["package_sha256"]} bytes={row["package_bytes"]}', flush=True)
    save_json(OUT / 'PACK_RESULT.json', {'status': 'COMPLETE', 'packages': rows, 'elapsed_sec': time.time()-t0})
    with (OUT / '人工提交回填模板.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'README_提交与回填.md').write_text(
        '# M2F Run1b 三档 ZERO\n\n按 01(56k) → 02(40k) → 03(20k) 手工提交。每个 ZIP 为一个独立单模型，不合并。\n'
        '每条回填必须含完整包名、平台 submission ID（如有）、平台显示文件名、分数，并保留截图或导出。\n'
        '本地 SHA 证明交付文件身份，不自动证明平台使用了哪个文件。只贴三个分数不能升级为文件名级绑定。\n'
        '平台分数目前全部留空；本助手未操作平台。\n', encoding='utf-8')
    print('PACK_ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
