"""IDENTITY CHECK: is measure_final_margin.py's tensor the SAME tensor the
submitted packages' argmax(S + bias) was taken on?

GPT raised a concrete contradiction:
    measure_final_margin.py reports the M2F final semantic score range
    0 .. 0.999997  (so nothing ever exceeds 1)
    earlier notes describe this same space reaching ~1.29
Both cannot describe the same tensor.  One of them is measuring the wrong object.

This script does NOT theorise.  It pins the object by bytes:

  1. Rebuild the exact tensor  S = forward_views(model, px, 1).mean(dim=0, float32)
     -- the identical expression used by the submitted packers.
  2. Report shape / dtype / min / max / mean / margin quantiles.
  3. argmax(S)  -> must match the known ZERO reference PNGs.
  4. argmax(S + [0,.2x7]) -> must match the bytes inside the SUBMITTED zip
     02_M2F56_FG_POS_020_TTA8.zip, entry by entry.
  5. CONTROL (bias BEFORE the mean, (views + b).mean()): mathematically
     identical, different FP32 rounding.  If this also matches byte-for-byte the
     comparison has low sensitivity and must be reported as such.
  6. DISCRIMINATION: argmax(log(S + 1e-6)) vs argmax(S) -- how many pixels move.
     Shows the test can in fact tell two nearby tensors apart.

Read-only: nothing is written except this script's own small result JSON.
Physical GPU2 only.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path('/home/jhy/rare_diag_20260915/m2f_gpu_20260915')
BUNDLE = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
TEST = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images')
CKPT = ROOT / 'formal_run1b/scores/score_iter56000.pt'
REF_ZERO = ROOT / 'test1_zero_run1b_v1/predictions_56k'
FG_ZIP = ROOT / 'test1_fg_sweep_v2/02_M2F56_FG_POS_020_TTA8.zip'
OUT = Path('/home/jhy/rare_diag_20260915/identity_check_20260916')
EXPECTED_SHA = '707a582df4cf5a22227ed73db9cf4bdad7e1689f72ae9f99f71fd09e22629272'
N_IMAGES = 12
BG = [0.0] + [0.2] * 7          # FG_POS_020, Background pinned to 0


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


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES'),
            os.environ.get('CUDA_DEVICE_ORDER')) != ('2', 'PCI_BUS_ID'):
        raise RuntimeError('Physical GPU2 only')
    guard(32 * 1024**2)
    if sha(CKPT) != EXPECTED_SHA:
        raise RuntimeError('Checkpoint SHA mismatch')
    for p in (str(ROOT / 'py'), str(BUNDLE), '/home/jhy/gpu2_day_queue'):
        sys.path.insert(0, p)

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation
    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import forward_views, VIEWS
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores

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
            feats = self.encoder(pixel_values=pixel_values,
                                 output_hidden_states=True).hidden_states
            o = self.head(feats)
            return SimpleNamespace(
                logits=semantic_scores(o['class_logits'], o['mask_logits']))

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    model = M2FScores(ck).to('cuda:0').eval()
    del ck

    names = [f'test1_{i}.png' for i in range(1, N_IMAGES + 1)]
    bias = torch.tensor(BG, dtype=torch.float32, device='cuda:0').view(8, 1, 1)

    print('=== TENSOR IDENTITY: S = forward_views(...).mean(dim=0, float32) ===',
          flush=True)
    stats = []
    zero_diff = 0
    fg_after_diff = 0
    fg_before_diff = 0
    logeps_diff = 0
    total_px = 0
    ref_ok = REF_ZERO.is_dir()
    print(f'reference ZERO predictions present: {ref_ok}  ({REF_ZERO})', flush=True)

    with zipfile.ZipFile(FG_ZIP) as z:
        zip_entries = set(z.namelist())
        zip_png = {n: z.read(n) for n in zip_entries}   # PNG bytes, decode below
        with torch.inference_mode():
            for name in names:
                guard(8 * 1024**2)
                with Image.open(TEST / name) as im:
                    img = np.asarray(im.convert('RGB')).copy()
                px = normalize_to_tensor(img).unsqueeze(0).to('cuda:0')
                with torch.amp.autocast('cuda', enabled=False):
                    views = forward_views(model, px, 1)
                    S = views.mean(dim=0, dtype=torch.float32)      # <-- THE tensor
                    S_before = (views + bias).mean(dim=0, dtype=torch.float32)
                S = S.float()
                stats.append(dict(name=name, shape=list(S.shape),
                                  dtype=str(S.dtype), min=float(S.min()),
                                  max=float(S.max()), mean=float(S.mean()),
                                  n_above_1=int((S > 1.0).sum()),
                                  n_above_0999=int((S > 0.999).sum())))
                if not torch.isfinite(S).all():
                    raise RuntimeError(f'Nonfinite: {name}')
                total_px += S.shape[-2] * S.shape[-1]

                p_zero = S.argmax(0).add(1).to(torch.uint8).cpu().numpy()
                p_fg_after = (S + bias).argmax(0).add(1).to(torch.uint8).cpu().numpy()
                p_fg_before = (S_before + bias).argmax(0).add(1).to(
                    torch.uint8).cpu().numpy()
                p_logeps = torch.log(S + 1e-6).argmax(0).add(1).to(
                    torch.uint8).cpu().numpy()

                if name in zip_entries:
                    import io
                    with Image.open(io.BytesIO(zip_png[name])) as h:
                        ref_fg = np.asarray(h)
                    fg_after_diff += int((p_fg_after != ref_fg).sum())
                    fg_before_diff += int((p_fg_before != ref_fg).sum())
                else:
                    raise RuntimeError(f'{name} absent from FG zip')

                if ref_ok:
                    with Image.open(REF_ZERO / name) as r:
                        ref_z = np.asarray(r)
                    zero_diff += int((p_zero != ref_z).sum())
                logeps_diff += int((p_logeps != p_zero).sum())
                del px, views, S, S_before
                torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    if sha(CKPT) != EXPECTED_SHA:
        raise RuntimeError('Checkpoint changed while inferring')

    mx = max(s['max'] for s in stats)
    mn = min(s['min'] for s in stats)
    above1 = sum(s['n_above_1'] for s in stats)
    above999 = sum(s['n_above_0999'] for s in stats)

    print(f'  images            {N_IMAGES}   total pixels {total_px}', flush=True)
    print(f'  shape/dtype       {stats[0]["shape"]} / {stats[0]["dtype"]}', flush=True)
    print(f'  global min        {mn:.9f}', flush=True)
    print(f'  global max        {mx:.9f}', flush=True)
    print(f'  pixels > 1.0      {above1}', flush=True)
    print(f'  pixels > 0.999    {above999}  ({above999/total_px*100:.4f}%)', flush=True)
    print(f'  mean over images  {sum(s["mean"] for s in stats)/len(stats):.9f}', flush=True)
    print(flush=True)

    print('=== BYTE-LEVEL PREDICTION COMPARISON ===', flush=True)
    print(f'  ZERO  argmax(S)          vs reference PNGs : differing px '
          f'{zero_diff} / {total_px}', flush=True)
    print(f'  FG_POS_020 argmax(S+b)   vs SUBMITTED zip  : differing px '
          f'{fg_after_diff} / {total_px}   <-- THE identity test', flush=True)
    print(f'  CONTROL bias-before-mean vs SUBMITTED zip  : differing px '
          f'{fg_before_diff} / {total_px}', flush=True)
    print(f'  DISCRIMINATION log(S+1e-6) vs S            : differing px '
          f'{logeps_diff} / {total_px}', flush=True)
    print(flush=True)

    if fg_after_diff == 0:
        verdict = ('CONFIRMED_SAME_TENSOR: measure_final_margin.py measures the exact '
                   'tensor the submitted +0.20 package took its argmax on; '
                   'margins and the [0,1] range are therefore statements about the '
                   'real calibration object.')
    else:
        verdict = ('MISMATCH: the tensor in measure_final_margin.py is NOT the tensor '
                   'the submitted packages were built on. All geometric conclusions '
                   'from this round must be withdrawn.')
    print('VERDICT:', verdict, flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / 'IDENTITY_RESULT.json').write_text(json.dumps({
        'script_sha256': sha(Path(__file__)), 'checkpoint_sha256': EXPECTED_SHA,
        'n_images': N_IMAGES, 'total_pixels': total_px,
        'tensor_expression': 'forward_views(model, px, 1).mean(dim=0, dtype=torch.float32)',
        'dtype': stats[0]['dtype'], 'shape': stats[0]['shape'],
        'global_min': mn, 'global_max': mx,
        'pixels_above_1': above1, 'pixels_above_0.999': above999,
        'zero_vs_reference_diff_px': zero_diff,
        'fg020_vs_submitted_zip_diff_px': fg_after_diff,
        'control_bias_before_mean_diff_px': fg_before_diff,
        'discrimination_logeps_diff_px': logeps_diff,
        'fg_zip': str(FG_ZIP), 'fg_zip_sha256': sha(FG_ZIP),
        'verdict': verdict.split(':')[0],
        'per_image': stats,
    }, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('IDENTITY_DONE', flush=True)


if __name__ == '__main__':
    main()
