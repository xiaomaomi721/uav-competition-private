"""M2F56 test1 D4-TTA8 FP32 class-offset probe packages (A..H).

ONE forward pass, EIGHT bias vectors. The 500 test1 images are pushed through the
verified D4-TTA8 FP32 pipeline exactly once; for each image the resulting FP32
semantic score map is used to write all eight prediction PNGs immediately. The
per-image score map is never spilled to disk, so the intermediate cost is a
single 8x1024x1024 float32 tensor.

Bias vectors (Background is ALWAYS 0 -- adding one constant to all eight classes
would leave argmax unchanged and the package would be identical to ZERO):
  A  joint_local_best            read verbatim from JOINT_RESULT.json
  B  joint_1d_best               read verbatim from C2_RESULT.json per_class_best
  C  2x_joint_1d_best            B's seven non-background biases x2
  D  opposite_joint_1d_best      B's seven non-background biases negated
  E  uniform_neg_005             [0, -.05] x7
  F  uniform_pos_005             [0, +.05] x7
  G  uniform_neg_010             [0, -.10] x7
  H  uniform_pos_010             [0, +.10] x7

A and B are never transcribed by hand: they are loaded from the frozen result
files and printed with full repr() and .hex() as provenance. C and D are derived
programmatically from the loaded B.

Sanity only: 500 entries, correct shape, legal label range, official checker PASS,
readable ZIP, recorded SHA. No validation, no verifier, no acceptance, no local
mIoU analysis.

Physical GPU2 only (PCI_BUS_ID + CUDA_VISIBLE_DEVICES=2). No training, no platform
submission.
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
CKPT = ROOT / 'formal_run1b/scores/score_iter56000.pt'
C2_JSON = ROOT / 'c2_56k' / 'C2_RESULT.json'
JOINT_JSON = ROOT / 'c2_56k' / 'JOINT_RESULT.json'
OUT = ROOT / 'test1_calib_probe_v1'
FLOOR = int(1.5 * 1024**3)
NAMES8 = ["Background", "Building", "Road", "Water", "Barren", "Vegetation",
          "Agricultural", "Vehicle"]


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
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2' or \
            os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('Only PCI_BUS_ID physical GPU2 / logical cuda:0 allowed')
    resume = '--resume' in sys.argv[1:]
    OUT.mkdir(exist_ok=True)

    # ---------- exact bias vectors, loaded not typed ----------
    c2 = json.loads(C2_JSON.read_text())
    jr = json.loads(JOINT_JSON.read_text())
    B = [0.0] * 8
    for name, d in c2["per_class_best"].items():
        B[NAMES8.index(name)] = float(d["bias"])
    if any(B[i] == 0.0 for i in range(1, 8)):
        raise RuntimeError(f'C2 1D best has a zero non-background entry: {B}')
    A = [float(x) for x in jr["joint_local_best_vector"]]
    if len(A) != 8 or A[0] != 0.0:
        raise RuntimeError(f'joint_local_best_vector malformed: {A}')
    C = [0.0] + [B[i] * 2.0 for i in range(1, 8)]
    D = [0.0] + [-B[i] for i in range(1, 8)]
    E = [0.0] + [-0.05] * 7
    F = [0.0] + [0.05] * 7
    G = [0.0] + [-0.10] * 7
    H = [0.0] + [0.10] * 7
    VECTORS = [("A_joint_local_best", A), ("B_joint_1d_best", B),
               ("C_2x_joint_1d_best", C), ("D_opposite_joint_1d_best", D),
               ("E_uniform_neg_005", E), ("F_uniform_pos_005", F),
               ("G_uniform_neg_010", G), ("H_uniform_pos_010", H)]
    for tag, v in VECTORS:
        if len(v) != 8 or v[0] != 0.0:
            raise RuntimeError(f'{tag}: Background must be exactly 0, got {v[0]!r}')
    # the forward pass is the expensive part; with --resume, if every bias
    # directory already holds all 500 predictions we skip model loading entirely
    # and go straight to sanity + official checker + zip
    resume_ok = False
    if resume:
        resume_ok = all(
            len({p.name for p in (OUT / f'predictions_{tag}').glob('*.png')}) == 500
            for tag, _ in VECTORS)
        print(f'RESUME requested, predictions complete: {resume_ok}', flush=True)
        if not resume_ok:
            raise RuntimeError('--resume given but prediction dirs are incomplete')

    print('=== bias provenance (exact) ===', flush=True)
    for tag, v in VECTORS:
        print(f'  {tag}', flush=True)
        print(f'    repr = {[repr(x) for x in v]}', flush=True)
        print(f'    hex  = {[float(x).hex() for x in v]}', flush=True)

    names = sorted([p.name for p in TEST.glob('*.png')],
                   key=lambda n: int(Path(n).stem.split('_')[-1]))
    if set(names) != {f'test1_{i}.png' for i in range(1, 501)} or len(names) != 500:
        raise RuntimeError('Test1 names must be exactly test1_1.png .. test1_500.png')

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
                                       num_layers=6, dim_feedforward=1024,
                                       ignore_index=255)
            self.encoder.load_state_dict(ck['model'], strict=True)
            self.head.load_state_dict(ck['head'], strict=True)

        def forward(self, pixel_values):
            features = self.encoder(pixel_values=pixel_values,
                                    output_hidden_states=True).hidden_states
            o = self.head(features)
            return SimpleNamespace(
                logits=semantic_scores(o['class_logits'], o['mask_logits']))

    torch.manual_seed(42)
    torch.set_num_threads(4)
    if not resume_ok:
        OUT.mkdir(exist_ok=True)

    files = [Path(__file__), ROOT / 'py/b3_m2f_head/head.py',
             ROOT / 'py/b3_m2f_head/semantic.py', BUNDLE / 'src/transforms.py',
             Path('/home/jhy/gpu2_day_queue/tta_anatomy_v2.py'),
             BUNDLE / 'scripts/check_submission.py', C2_JSON, JOINT_JSON]
    sources = {str(p): sha(p) for p in files}
    ck_sha = sha(CKPT)

    model = None
    if not resume_ok:
        ck = torch.load(CKPT, map_location='cpu', weights_only=False)
        if ck['data_iteration'] != 56000:
            raise RuntimeError('Checkpoint step mismatch')
        c = ck['config']
        if (c['num_queries'], c['hidden_dim'], c['layers']) != (100, 256, 6):
            raise RuntimeError('Unexpected checkpoint architecture')
        model = M2FScores(ck).to('cuda:0').eval()
        del ck, c

    protocol = {
        'schema': 'M2F56_TEST1_CALIB_PROBE_V1',
        'physical_gpu': 2, 'logical_device': 'cuda:0',
        'checkpoint': str(CKPT), 'checkpoint_sha256': ck_sha,
        'checkpoint_step': 56000,
        'test_dir': str(TEST), 'test_count': 500,
        'test_names_sha256': hashlib.sha256(('\n'.join(names) + '\n').encode()).hexdigest(),
        'tta': 'D4 TTA8', 'views': VIEWS, 'view_batch_size': 1,
        'precision': 'FP32, autocast disabled; FP32 view mean',
        'normalization': 'frozen src.transforms.normalize_to_tensor',
        'semantic_synthesis': 'softmax(class)[...,:8] x sigmoid(mask), einsum bqc,bqhw->bchw',
        'order': 'per-view mixture -> inverse mirror -> inverse rotation -> bilinear '
                 'to original size -> FP32 mean of 8 -> + bias -> argmax + 1',
        'bias_application': 'AFTER TTA8 FP32 averaging, BEFORE argmax; Background fixed 0',
        'single_forward_multiple_biases': True,
        'bias_vectors_exact': {tag: [float(x) for x in v] for tag, v in VECTORS},
        'bias_vectors_hex': {tag: [float(x).hex() for x in v] for tag, v in VECTORS},
        'bias_sources': {'A': 'JOINT_RESULT.json -> joint_local_best_vector',
                         'B': 'C2_RESULT.json -> per_class_best',
                         'C': 'B x2 on the 7 non-background entries',
                         'D': 'B negated on the 7 non-background entries',
                         'E..H': 'uniform on the 7 non-background entries'},
        'code_and_provenance_sha256': sources,
        'training': False, 'calibration_search': False,
        'environment': {'python': sys.version, 'torch': torch.__version__,
                        'gpu': torch.cuda.get_device_name(0)},
    }
    # on --resume this file already exists, so write it rather than create-exclusive
    (OUT / 'INFERENCE_CONFIG.json').write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='utf-8')
    config_sha = sha(OUT / 'INFERENCE_CONFIG.json')

    pred_dirs = {}
    for tag, _ in VECTORS:
        d = OUT / f'predictions_{tag}'
        d.mkdir(exist_ok=resume_ok)
        pred_dirs[tag] = d

    # ---------- ONE forward pass, eight predictions per image ----------
    t0 = time.time()
    # bias tensors must live on the same device as the score map
    bias_t = {tag: torch.tensor(v, dtype=torch.float32).view(8, 1, 1).to('cuda:0')
              for tag, v in VECTORS}
    # on --resume every prediction is already on disk, so iterate an empty list;
    # this keeps the loop body untouched (a re-indent here previously broke it)
    loop_names = [] if resume_ok else names
    with torch.inference_mode():
        for i, name in enumerate(loop_names, 1):
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
            for tag, _ in VECTORS:
                pred = (scores + bias_t[tag]).argmax(0).add(1).to(torch.uint8)
                arr = pred.cpu().numpy()
                if arr.min() < 1 or arr.max() > 8:
                    raise RuntimeError(f'Illegal label range for {name} {tag}')
                Image.fromarray(arr).save(pred_dirs[tag] / name, format='PNG')
                del pred, arr
            del px, views, scores
            if i % 50 == 0:
                print(f'PROGRESS {i}/500 elapsed={time.time()-t0:.0f}s '
                      f'free={shutil.disk_usage(ROOT).free/2**30:.2f}GiB', flush=True)
    del model, bias_t
    torch.cuda.empty_cache()

    if sha(CKPT) != ck_sha:
        raise RuntimeError('Checkpoint changed while inferring')
    for path, digest in sources.items():
        if sha(path) != digest:
            raise RuntimeError(f'Source changed while inferring: {path}')

    # ---------- minimal sanity + official checker + zip ----------
    rows = []
    for ordinal, (tag, v) in enumerate(VECTORS, 1):
        pred_dir = pred_dirs[tag]
        # compare as a SET: `names` is ordinal-sorted (test1_2 before test1_10)
        # while a plain sorted() of the directory is lexicographic, so the same
        # 500 files compare unequal as lists.
        got = {p.name for p in pred_dir.glob('*.png')}
        if got != set(names) or len(got) != 500:
            raise RuntimeError(f'{tag}: expected 500 entries, got {len(got)}')
        check_dir = OUT / f'check_{tag}'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        subprocess.run([sys.executable, str(BUNDLE / 'scripts/check_submission.py'),
                        '--test-images', str(TEST), '--predictions', str(pred_dir),
                        '--output-dir', str(check_dir), '--expected-count', '500'],
                       env=env, check=True)
        check = json.loads((check_dir / 'submission_check.json').read_text())
        if check['status'] != 'PASS':
            raise RuntimeError(f'Official check did not pass for {tag}')
        package = OUT / f'{ordinal:02d}_M2F56_{tag}_TTA8.zip'
        disk_guard(sum(p.stat().st_size for p in pred_dir.iterdir()) + 8 * 1024**2)
        with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_STORED) as z:
            for name in names:
                z.write(pred_dir / name, arcname=name)
        with zipfile.ZipFile(package) as z:
            if z.namelist() != names or z.testzip() is not None:
                raise RuntimeError(f'ZIP entry or CRC mismatch for {tag}')
            for name in names:
                if z.read(name) != (pred_dir / name).read_bytes():
                    raise RuntimeError(f'ZIP content mismatch {tag} {name}')
        row = {'suggested_order': ordinal, 'tag': tag,
               'bias_vector_exact': [float(x) for x in v],
               'bias_vector_hex': [float(x).hex() for x in v],
               'package_filename': package.name, 'package_sha256': sha(package),
               'package_bytes': package.stat().st_size,
               'checkpoint_sha256': ck_sha, 'inference_config_sha256': config_sha,
               'official_check': 'PASS', 'entries': 500,
               'platform_M2F56_ZERO_reference': 0.684133,
               'platform_submission_id': None, 'platform_filename': None,
               'platform_mIoU': None, 'submission_status': 'NOT_SUBMITTED_BY_ASSISTANT'}
        save_json(OUT / f'PACKAGE_{tag}.json', row)
        rows.append(row)
        print(f'PACKAGE_READY {package.name} sha={row["package_sha256"]} '
              f'bytes={row["package_bytes"]}', flush=True)

    save_json(OUT / 'PACK_RESULT.json',
              {'status': 'COMPLETE', 'packages': rows,
               'elapsed_sec': time.time() - t0,
               'single_forward_pass': True, 'n_bias_vectors': len(VECTORS)})
    with (OUT / '人工提交回填模板.csv').open('x', encoding='utf-8-sig',
                                             newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('ALL_PACKAGES_DONE', flush=True)


if __name__ == '__main__':
    main()
