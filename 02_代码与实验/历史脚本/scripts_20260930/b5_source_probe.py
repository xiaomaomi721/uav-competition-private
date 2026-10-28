#!/usr/bin/env python3
"""B5 source-only CE probe: does B5 cold-start fit the UAV domain fast enough?

B5 (segformer-b5-ade640, ADE20K pretrained) trained from scratch (NOT from A05)
on 5597 source images with plain CE. Probe runs N steps and logs per-500 loss
+ a local-val mIoU so we can see how fast B5 fits vs the known B3 curve
(B3 cold-start CE needed ~56k steps to reach 66.739).

If B5 loss/mIoU improves fast -> strong pretrain helps, worth full training.
If slow -> B5 cold-start is too expensive, don't sink time into it.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/uav')
SRC = ROOT / 'code/src'
CONFIG = ROOT / 'code/config.yaml'
MODEL_SOURCE = ROOT / 'models/segformer-b5-ade640'
DATA = ROOT / 'datasets/训练集'
SPLIT = ROOT / 'code/splits/train.txt'
TRAIN_PREFIX = ('segformer.stages.2.', 'segformer.stages.3.')
HEAD_PREFIX = 'decode_head.'
SEED = 20260922


def verify_gpu(gpu_id):
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu_id):
        raise SystemExit(f'GPU{gpu_id} required: CUDA_VISIBLE_DEVICES={gpu_id}')
    import subprocess
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines()
    m = [x for x in rows if x.strip().startswith(f'{gpu_id},')]
    if not m:
        raise SystemExit(f'GPU{gpu_id} not found: {rows}')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit('expected exactly one visible CUDA device')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=5000)
    ap.add_argument('--lr', type=float, default=6e-5)
    a = ap.parse_args()
    verify_gpu(a.gpu)
    for p in (str(ROOT / 'code'), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ['UAVSEG_MODEL_SOURCE'] = str(MODEL_SOURCE)
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    from torch.utils.data import DataLoader
    from src.models import build_model as mm
    from src.utils import load_config
    from src.dataset import UAVSegmentationDataset, segmentation_collate
    from src.rare_crop import RareClassCropper
    from src.transforms import TrainTransform

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    device = 'cuda:0'

    random_mod = __import__('random')
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    cfg['pretrained_model_path'] = str(MODEL_SOURCE)
    cfg['model_name'] = 'nvidia/segformer-b5-finetuned-ade-640-640'  # structure is overridden by B5 config.json

    model = mm(cfg)
    model = model.to(device)
    # full training (cold start): all params trainable
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    # print parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'B5 trainable params: {n_params/1e6:.1f}M', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    ids = [x.strip() for x in SPLIT.read_text().split() if x.strip()]
    assert len(ids) == 5597, f'expected 5597 source ids, got {len(ids)}'
    crop = RareClassCropper(crop_size=1024, rare_probability=.5, priority_ids=(7, 4, 6), max_offset_fraction=.25)
    src = UAVSegmentationDataset(DATA, ids, TrainTransform(crop_size=1024, config=cfg['transforms'], rare_cropper=crop, ignore_index=255), maximum=None, seed=SEED, random_subset=False)
    sl = DataLoader(src, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=segmentation_collate, generator=torch.Generator().manual_seed(SEED + 10001), persistent_workers=True)

    i_src = iter(sl)
    t0 = time.time()
    amp_skips = 0
    for step in range(1, a.max_steps + 1):
        try:
            s = next(i_src)
        except StopIteration:
            i_src = iter(sl)
            s = next(i_src)
        lr = a.lr * (1 - (step - 1) / a.max_steps)
        for g in opt.param_groups:
            g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        sx = s['pixel_values'].to(device)
        sy = s['labels'].to(device)
        with torch.amp.autocast('cuda', enabled=True):
            sz = model(pixel_values=sx).logits
        loss = F.cross_entropy(F.interpolate(sz.float(), size=sy.shape[-2:], mode='bilinear', align_corners=False), sy, ignore_index=255)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        sb = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < sb:
            amp_skips += 1
        if step % 500 == 0 or step == 1:
            print(json.dumps({'step': step, 'loss': round(float(loss), 4), 'lr': round(lr, 7), 'elapsed': round(time.time() - t0, 1), 'amp_skips': amp_skips}), flush=True)

        if step % 5000 == 0 or step == a.max_steps:
            torch.save({'model': model.state_dict(), 'step': step}, out / f'checkpoint_step{step}.pt')
            print(f'Saved checkpoint_step{step}.pt', flush=True)

    summary = {'steps': a.max_steps, 'elapsed_sec': round(time.time() - t0, 1), 'lr': a.lr, 'amp_skips': amp_skips, 'params_M': round(n_params/1e6, 1)}
    (out / 'RUN_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
