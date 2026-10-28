"""Measure the pre-bias tensor statistics for BOTH score spaces.

OLD (Stage59 / target_transfer.py):
    acc = mean over the 8 D4 views of the model's RAW class logits (C=8)
    FIXED8D is added to `acc`             -> raw logits additive bias

M2F56 (pack_calib_probe_v1.py):
    class_logits (B,Q,9) -> softmax(-1)[..., :-1] -> class_prob (B,Q,8)
    mask_logits  (B,Q,h,w) -> sigmoid -> mask_prob
    scores = einsum("bqc,bqhw->bchw")      -> bias added AFTER this, before argmax

This prints the actual numeric ranges so the g (semantic-vs-no-object shift)
scale can be chosen from measured data rather than guessed.

Read-only diagnostic: no predictions written, no packages, no training.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

SNAP = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
ROOT = Path('/home/jhy/rare_diag_20260915/m2f_gpu_20260915')
BUNDLE = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
DS = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images')

import numpy as np
import torch

sys.path.insert(0, str(SNAP))
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(ROOT / 'py'))
sys.path.insert(0, '/home/jhy/gpu2_day_queue')

from PIL import Image
from src.transforms import normalize_to_tensor

FIXED8D = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
NAMES = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation',
         'Agricultural', 'Vehicle']


def natural(p):
    return int(p.stem[len('test1_'):])


def images(n):
    ps = sorted(DS.glob('test1_*.png'), key=natural)[:n]
    for p in ps:
        with Image.open(p) as h:
            yield np.asarray(h.convert('RGB')).copy()


def stats(name, t):
    t = t.float()
    q = torch.quantile(t.reshape(-1)[: 4_000_000].float(),
                       torch.tensor([0.01, 0.5, 0.99], device=t.device))
    return (f'{name:34s} shape={tuple(t.shape)} dtype={t.dtype} '
            f'min={t.min():.4f} p01={q[0]:.4f} med={q[1]:.4f} p99={q[2]:.4f} '
            f'max={t.max():.4f} mean={t.mean():.4f} std={t.std():.4f}')


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dev = 'cuda:0'

    # ---------------- OLD: raw TTA8 mean logits ----------------
    from scripts.probe.target_transfer import load_model, FIXED8D as F8
    old = load_model(Path(sys.argv[2]), dev)
    print('=== OLD (Stage59-style) : raw class logits ===', flush=True)
    print('FIXED8D =', F8, flush=True)
    accs = []
    with torch.inference_mode():
        for im in images(n):
            px = normalize_to_tensor(im).unsqueeze(0).to(dev)
            acc = None
            for k in range(4):
                for hf in (False, True):
                    v = px
                    if k:
                        v = torch.rot90(v, k=k, dims=[-2, -1])
                    if hf:
                        v = v.flip(dims=[-1])
                    lg = old(pixel_values=v).logits
                    if hf:
                        lg = lg.flip(dims=[-1])
                    if k:
                        lg = torch.rot90(lg, k=-k, dims=[-2, -1])
                    if tuple(lg.shape[-2:]) != (1024, 1024):
                        lg = torch.nn.functional.interpolate(
                            lg, size=(1024, 1024), mode='bilinear', align_corners=False)
                    acc = lg if acc is None else acc + lg
            accs.append((acc / 8).squeeze(0))
    A = torch.stack(accs)
    print(stats('TTA8 mean logits (pre-bias)', A), flush=True)
    print(f'{"per-class mean":34s} ' +
          ' '.join(f'{NAMES[i]}={A[:,i].mean():.4f}' for i in range(8)), flush=True)
    m = A.topk(2, dim=1).values
    margin = (m[:, 0] - m[:, 1]).reshape(-1)
    print(f'{"per-pixel top1-top2 margin":34s} med={margin.median():.4f} '
          f'p90={torch.quantile(margin[:4_000_000], 0.9):.4f}', flush=True)
    B = A + torch.tensor(F8, device=A.device).view(1, 8, 1, 1)
    chg = (B.argmax(1) != A.argmax(1)).float().mean()
    print(f'{"argmax changed by FIXED8D":34s} {chg*100:.3f}%', flush=True)
    del old
    torch.cuda.empty_cache()

    # ---------------- M2F56: query class logits ----------------
    print(flush=True)
    print('=== M2F56 : query class logits (8 semantic + 1 no-object) ===', flush=True)
    from transformers import AutoModelForSemanticSegmentation
    from b3_m2f_head.head import B3M2FStyleHead

    ck = torch.load(ROOT / 'formal_run1b/scores/score_iter56000.pt',
                    map_location='cpu', weights_only=False)
    enc = AutoModelForSemanticSegmentation.from_pretrained(
        '/home/jhy/models/segformer-b3-ade512', num_labels=8,
        ignore_mismatched_sizes=True, local_files_only=True).segformer
    head = B3M2FStyleHead(in_channels=(64, 128, 320, 512), num_classes=8,
                          hidden_dim=256, num_queries=100, nhead=8, num_layers=6,
                          dim_feedforward=1024, ignore_index=255)
    enc.load_state_dict(ck['model'], strict=True)
    head.load_state_dict(ck['head'], strict=True)
    enc, head = enc.to(dev).eval(), head.to(dev).eval()
    del ck

    cl, ml = [], []
    with torch.inference_mode():
        for im in images(n):
            px = normalize_to_tensor(im).unsqueeze(0).to(dev)
            f = enc(pixel_values=px, output_hidden_states=True).hidden_states
            o = head(f)
            cl.append(o['class_logits'].squeeze(0).float().cpu())
            ml.append(o['mask_logits'].squeeze(0).float().cpu())
    CL = torch.stack(cl)
    ML = torch.stack(ml)
    print(stats('class_logits (Q,9)', CL), flush=True)
    print(f'{"per-class-query mean":34s} ' +
          ' '.join(f'{NAMES[i]}={CL[..., i].mean():.4f}' for i in range(8)) +
          f' noobj={CL[..., 8].mean():.4f}', flush=True)
    sem = CL[..., :8]
    spread = (sem.max(-1).values - sem.min(-1).values).reshape(-1)
    print(f'{"semantic logit spread (max-min)":34s} med={spread.median():.4f} '
          f'p90={torch.quantile(spread[:4_000_000], 0.9):.4f}', flush=True)
    vs = (sem - CL[..., 8:9]).reshape(-1)
    print(f'{"semantic-minus-noobject margin":34s} med={vs.median():.4f} '
          f'p10={torch.quantile(vs[:4_000_000], 0.1):.4f} '
          f'p90={torch.quantile(vs[:4_000_000], 0.9):.4f}', flush=True)
    print(flush=True)
    print(stats('mask_logits', ML), flush=True)
    p = ML.sigmoid()
    print(f'{"sigmoid(mask_logits)":34s} min={p.min():.6f} med={p.median():.6f} '
          f'max={p.max():.6f}', flush=True)
    cp = CL.softmax(-1)[..., :8]
    print(f'{"softmax(class_logits)[:8]":34s} min={cp.min():.6f} '
          f'med={cp.median():.6f} max={cp.max():.6f}', flush=True)
    print('MEASURE_DONE', flush=True)


if __name__ == '__main__':
    main()
