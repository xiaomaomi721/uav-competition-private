"""Stage A (parallel): M2F 56k D4-TTA8 FP32 validation score cache.

Protocol is identical to the test1 zero packages:
  per-view M2F native semantic scores -> inverse mirror/rotation -> bilinear to
  original size -> FP32 mean over 8 views.
Writes one .npy per image into a /dev/shm directory (RAM-backed, no disk use).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path('/home/jhy/rare_diag_20260915/m2f_gpu_20260915')
BUNDLE = Path('/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1')
CKPT = ROOT / 'formal_run1b/scores/score_iter56000.pt'
SHM = Path('/dev/shm/tta8_56k_v2')
DATASET = Path('/home/jhy/datasets/2026-低空图像语义分割赛道-训练集')
VAL = BUNDLE / 'splits/val.txt'


def worker(stems, gpu_slot):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path[:0] = [str(ROOT / 'py'), str(BUNDLE), '/home/jhy/gpu2_day_queue']
    from transformers import AutoModelForSemanticSegmentation
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores
    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import forward_views

    dev = 'cuda:0'
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    enc = AutoModelForSemanticSegmentation.from_pretrained(
        '/home/jhy/models/segformer-b3-ade512', num_labels=8,
        ignore_mismatched_sizes=True, local_files_only=True).segformer.to(dev)
    head = B3M2FStyleHead(in_channels=(64, 128, 320, 512), num_classes=8,
                          hidden_dim=256, num_queries=100, nhead=8, num_layers=6,
                          dim_feedforward=1024, ignore_index=255).to(dev)
    enc.load_state_dict(ck['model'], strict=True)
    head.load_state_dict(ck['head'], strict=True)
    enc.eval(); head.eval(); del ck

    class M(torch.nn.Module):
        def forward(self, pixel_values):
            f = enc(pixel_values=pixel_values, output_hidden_states=True).hidden_states
            o = head(f)
            from types import SimpleNamespace
            return SimpleNamespace(
                logits=semantic_scores(o['class_logits'].float(), o['mask_logits'].float()))

    model = M()
    img_dir = DATASET / 'train/images'
    mask_dir = DATASET / 'train/masks'
    t0 = time.time()
    with torch.inference_mode():
        for n, stem in enumerate(stems, 1):
            img = np.asarray(Image.open(img_dir / f'{stem}.png').convert('RGB')).copy()
            gt = np.asarray(Image.open(mask_dir / f'{stem}.png')).copy()
            px = normalize_to_tensor(img).unsqueeze(0).to(dev)
            with torch.amp.autocast('cuda', enabled=False):
                views = forward_views(model, px, 1)
                sc = views.mean(dim=0, dtype=torch.float32)
            if not torch.isfinite(sc).all():
                raise RuntimeError(f'nonfinite {stem}')
            out = sc.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
            np.save(SHM / f'{stem}.npy', out, allow_pickle=False)
            np.save(SHM / f'gt_{stem}.npy', gt.astype(np.uint8), allow_pickle=False)
            del px, views, sc, out
            if n % 25 == 0:
                print(f'[slot{gpu_slot}] {n}/{len(stems)} {time.time()-t0:.0f}s', flush=True)


def main():
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    stems = [Path(l.strip()).stem for l in VAL.read_text().split() if l.strip()]
    assert len(stems) == 1399 and len(set(stems)) == 1399
    SHM.mkdir(parents=True, exist_ok=False)
    chunks = [stems[i::nproc] for i in range(nproc)]
    ctx = mp.get_context('spawn')
    ps = [ctx.Process(target=worker, args=(c, i)) for i, c in enumerate(chunks)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
        if p.exitcode != 0:
            raise SystemExit(f'worker failed: {p.exitcode}')
    (SHM / '_DONE').write_text('ok')
    print('CACHE_DONE', flush=True)


if __name__ == '__main__':
    main()
