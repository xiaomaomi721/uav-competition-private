"""Measure the per-pixel decision margin of M2F56's FINAL semantic scores.

This is the number that decides how much leverage an additive bias has.
The old model (Stage59/Stage23 family) had:
    TTA8 mean raw logits, per-pixel top1-top2 margin median = 6.47
    FIXED8D span = 4.10  ->  ratio 0.63  ->  changed 6.443% of pixels
    platform gain = +1.2883 pp

If M2F's final-score margin is much larger relative to the biases we have been
testing (span 0.05-0.20), that structurally explains the weak response, and it
is a property of the OPERATOR/space, independent of any validation set.

Read-only. 20 images. No predictions written.
"""
import sys
from pathlib import Path

ROOT = Path('/home/jhy/rare_diag_20260915/m2f_gpu_20260915')
BUNDLE = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
DS = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images')
CKPT = ROOT / 'formal_run1b/scores/score_iter56000.pt'

import numpy as np
import torch

sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(ROOT / 'py'))
sys.path.insert(0, '/home/jhy/gpu2_day_queue')

from PIL import Image
from src.transforms import normalize_to_tensor


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dev = 'cuda:0'
    import torch.nn.functional as F
    from transformers import AutoModelForSemanticSegmentation
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
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

    views = [(k, hf) for k in range(4) for hf in (False, True)]
    ps = sorted(DS.glob('test1_*.png'),
                key=lambda p: int(p.stem[len('test1_'):]))[:n]
    scores_all, mask_sum = [], []
    with torch.inference_mode():
        for p in ps:
            with Image.open(p) as h:
                im = np.asarray(h.convert('RGB')).copy()
            px = normalize_to_tensor(im).unsqueeze(0).to(dev)
            acc = None
            with torch.amp.autocast('cuda', enabled=False):
                for k, hf in views:
                    v = px
                    if k:
                        v = torch.rot90(v, k=k, dims=[-2, -1])
                    if hf:
                        v = v.flip(dims=[-1])
                    f = enc(pixel_values=v, output_hidden_states=True).hidden_states
                    o = head(f)
                    sc = semantic_scores(o['class_logits'], o['mask_logits'])
                    if hf:
                        sc = sc.flip(dims=[-1])
                    if k:
                        sc = torch.rot90(sc, k=-k, dims=[-2, -1])
                    if tuple(sc.shape[-2:]) != (1024, 1024):
                        sc = F.interpolate(sc, size=(1024, 1024), mode='bilinear',
                                           align_corners=False)
                    acc = sc if acc is None else acc + sc
                    del v, f, o, sc
            m = (acc / 8).squeeze(0).float().cpu()
            scores_all.append(m)
            del px, acc
            torch.cuda.empty_cache()
    A = torch.stack(scores_all)                       # (n,8,1024,1024)

    print('=== M2F56 FINAL semantic score space (the space today\'s bias acts on) ===')
    print(f'  shape {tuple(A.shape)}   range {A.min():.6f} .. {A.max():.6f}   '
          f'mean {A.mean():.6f}')
    top2 = A.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).reshape(-1)
    q = torch.quantile(margin[: 4_000_000],
                       torch.tensor([0.10, 0.50, 0.90, 0.99]))
    print(f'  per-pixel top1-top2 margin: p10={q[0]:.6f} median={q[1]:.6f} '
          f'p90={q[2]:.6f} p99={q[3]:.6f}')
    print(f'  OLD model raw-logit margin median was 6.4687 for comparison')
    print()
    for span, label in ((0.05, 'FG uniform +0.05'), (0.10, 'FG uniform +0.10'),
                        (0.20, 'FG uniform +0.20 (platform best)'),
                        (4.10, 'old FIXED8D span')):
        b = torch.zeros(8)
        b[1:] = span          # Background stays 0: [0, +span x7]
        p0 = A.argmax(1)
        p1 = (A + b.view(1, 8, 1, 1)).argmax(1)
        chg = (p0 != p1).float().mean().item()
        print(f'  {label:34s} span={span:5.2f}  '
              f'changed {chg*100:8.4f}% of pixels   '
              f'span/median-margin = {span/q[1]:7.4f}')
    print()
    print('=== same quantity for the pre-softmax probes is NOT computed here;')
    print('    those were measured end-to-end as 0.002-0.005% pixel change ===')
    print('MARGIN_DONE', flush=True)


if __name__ == '__main__':
    main()
