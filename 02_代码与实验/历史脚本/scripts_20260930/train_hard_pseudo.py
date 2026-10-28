#!/usr/bin/env python3
"""Hard pseudo-label self-training (pseudo500-style): source GT + test-domain
hard pseudo-labels, joint CE training. Student init = A05 (B3), train stages 2/3 + head.

This is the "hard label" variant of test-domain self-training (pseudo500 +1.309),
NOT the soft-KL blind-recall (+0.27). Hard pseudo-labels are argmax of the Arm B
teacher, filtered by the blind-recall mask, stored as official 1-8 PNG.
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
MODEL_SOURCE = ROOT / 'models/segformer-b3-ade512'
A05 = ROOT / 'checkpoints/A05_WEIGHT_alpha05.pt'
DATA = ROOT / 'datasets/训练集'
SPLIT = ROOT / 'code/splits/train.txt'
TEST1_IMG = ROOT / 'datasets/训练集/images'
TEST2_IMG = ROOT / 'datasets/test_2/images'
PSEUDO_T1 = ROOT / 'teacher_cache/armb_hard_pseudo/test1'
PSEUDO_T2 = ROOT / 'teacher_cache/armb_hard_pseudo/test2'
EXPECTED_A05_SHA = '023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134'
TRAIN_PREFIX = ('segformer.stages.2.', 'segformer.stages.3.')
HEAD_PREFIX = 'decode_head.'
SEED = 20260922


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def is_train(name):
    return name.startswith(TRAIN_PREFIX) or name.startswith(HEAD_PREFIX)


def official_to_train(mask):
    out = np.full(mask.shape, 255, dtype=np.int64)
    for c in range(1, 9):
        out[mask == c] = c - 1
    return out


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


class HardPseudoDataset:
    """Test-domain images + Arm-B hard pseudo-labels, official 1-8 -> train 0-7+255."""
    def __init__(self, items, transform):
        # items: list of (img_path, pseudo_path)
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, pseudo_path = self.items[i]
        from PIL import Image
        image = np.asarray(Image.open(img_path).convert('RGB')).copy()
        mask = official_to_train(np.asarray(Image.open(pseudo_path)))
        # apply the same TrainTransform (crop + geometric aug, image & mask together)
        pixel_values, labels = self.transform(image, mask)
        return {"pixel_values": pixel_values, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=2000)
    ap.add_argument('--source-coeff', type=float, default=0.1)
    ap.add_argument('--target-coeff', type=float, default=1.0)
    a = ap.parse_args()
    verify_gpu(a.gpu)
    for p in (str(ROOT / 'code'), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ['UAVSEG_MODEL_SOURCE'] = str(MODEL_SOURCE)
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    from torch.utils.data import DataLoader, ConcatDataset
    from src.models import build_model as make_model
    from src.utils import load_config
    from src.dataset import UAVSegmentationDataset, segmentation_collate
    from src.rare_crop import RareClassCropper
    from src.transforms import TrainTransform

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    device = 'cuda:0'

    if sha256_file(A05) != EXPECTED_A05_SHA:
        raise SystemExit('A05 SHA mismatch')

    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    cfg = load_config(str(CONFIG))
    cfg['cwfa'] = {'enabled': False, 'probability': .3, 'epsilon': 4., 'norm_eps': 1e-12, 'start_step': 1}
    cfg.pop('mixstyle', None)
    cfg['pretrained_model_path'] = str(MODEL_SOURCE)

    model = make_model(cfg)
    ckpt = torch.load(str(A05), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model', ckpt), strict=True)
    model = model.to(device)

    model.eval()
    model.get_submodule('segformer.stages.2').train()
    model.get_submodule('segformer.stages.3').train()
    for n, p in model.named_parameters():
        p.requires_grad_(is_train(n))

    base_params = [p for n, p in model.named_parameters() if n.startswith(TRAIN_PREFIX)]
    head_params = [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([
        {'params': base_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-5 * 0.05},
    ], weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    # source dataset (real GT)
    ids = [x.strip() for x in SPLIT.read_text().split() if x.strip()]
    assert len(ids) == 5597
    crop = RareClassCropper(crop_size=1024, rare_probability=.5, priority_ids=(7, 4, 6), max_offset_fraction=.25)
    src_transform = TrainTransform(crop_size=1024, config=cfg['transforms'], rare_cropper=crop, ignore_index=255)
    src = UAVSegmentationDataset(DATA, ids, src_transform, maximum=None, seed=SEED, random_subset=False)

    # hard pseudo dataset (test1 + test2)
    pseudo_items = []
    for i in range(1, 501):
        pseudo_items.append((TEST1_IMG / f'test1_{i}.png', PSEUDO_T1 / f'test1_{i}.png'))
    for i in range(1, 1301):
        pseudo_items.append((TEST2_IMG / f'test2_{i}.png', PSEUDO_T2 / f'test2_{i}.png'))
    pseudo = HardPseudoDataset(pseudo_items, src_transform)
    print(f'source={len(src)}, pseudo={len(pseudo)}', flush=True)

    # joint dataset: source + pseudo, shuffle
    joint = ConcatDataset([src, pseudo])
    loader = DataLoader(joint, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=segmentation_collate, generator=torch.Generator().manual_seed(SEED + 10001), persistent_workers=True)

    it = iter(loader)
    t0 = time.time()
    amp_skips = 0
    for step in range(1, a.max_steps + 1):
        try:
            s = next(it)
        except StopIteration:
            it = iter(loader)
            s = next(it)
        lr = 1e-5 * (1 - (step - 1) / a.max_steps)
        for gi, g in enumerate(opt.param_groups):
            g['lr'] = (lr if gi == 0 else lr * 0.05)
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
        if step % 100 == 0 or step == 1:
            print(json.dumps({'step': step, 'loss': round(float(loss), 4), 'elapsed': round(time.time() - t0, 1), 'amp_skips': amp_skips}), flush=True)
        if step % 1000 == 0 or step == a.max_steps:
            torch.save({'model': model.state_dict(), 'step': step}, out / f'checkpoint_step{step}.pt')
            print(f'Saved checkpoint_step{step}.pt', flush=True)

    summary = {'steps': a.max_steps, 'elapsed_sec': round(time.time() - t0, 1), 'amp_skips': amp_skips, 'source': len(src), 'pseudo': len(pseudo)}
    (out / 'RUN_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
