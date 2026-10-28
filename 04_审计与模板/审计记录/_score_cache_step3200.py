#!/usr/bin/env python3
"""soupB KD Run1 @3200: ONE D4-TTA8 score forward, cached for CPU re-packing.

Frozen-plan contract
--------------------
The 3200 checkpoint gets exactly ONE official D4-TTA8 score forward. The
resulting per-image mean logits are cached here; the three submission packages
(ZERO / C / #13) are then generated on CPU from that SAME cache, so the three
bias vectors differ only by the constant added to the class axis -- no second
TTA8 is run, and no bias can influence another's score.

Inference contract is the project's already-validated one, reused verbatim from
_kd_pack_eight.py: tta_anatomy_v2.build_model + forward_views (TTA8, D4 group),
FP32 mean over the 8 views. Bias is applied to the class axis of the mean logits.

Read-only with respect to training. Writes only under --outdir (must not exist).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
PHYSICAL_GPU_INDEX = 2
SNAP = "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1"
DEFAULT_TEST_DIR = "/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_gpu(device: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise RuntimeError("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("require CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if device != "cuda:0":
        raise RuntimeError("logical device must be cuda:0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="gate checkpoint; its 'model' entry is loaded strict")
    ap.add_argument("--expected-sha256", default=None,
                    help="fail-closed guard on the checkpoint file digest")
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    ap.add_argument("--names", default=None, help="default: probs/*.npy stems of the run")
    ap.add_argument("--view-batch-size", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt}")

    ckpt_sha = sha256_file(ckpt)
    if args.expected_sha256 and ckpt_sha != args.expected_sha256:
        raise SystemExit(f"checkpoint sha256 {ckpt_sha} != expected "
                         f"{args.expected_sha256}")

    assert_gpu(args.device)
    import torch
    for p in ("/home/jhy/gpu2_day_queue", SNAP):
        if p not in sys.path:
            sys.path.insert(0, p)
    from PIL import Image
    from tta_anatomy_v2 import build_model, forward_views
    from src.transforms import normalize_to_tensor

    test_dir = Path(args.test_dir)
    if args.names:
        names = [l.strip() for l in Path(args.names).read_text().split() if l.strip()]
    else:
        names = sorted(p.name for p in test_dir.glob("*.png"))
    if len(names) != 500:
        raise SystemExit(f"expected 500 test images, got {len(names)}")

    outdir.mkdir(parents=True)
    model = build_model(args.config, str(ckpt), args.device)
    print(f"model loaded from {ckpt.name} (sha256 {ckpt_sha[:16]}), "
          f"{len(names)} images, D4-TTA8", flush=True)

    argmax = np.empty((len(names), 1024, 1024), dtype=np.uint8)
    t0 = time.time()
    for i, name in enumerate(names):
        img = np.asarray(Image.open(test_dir / name).convert("RGB")).copy()
        px = normalize_to_tensor(img).unsqueeze(0).to(args.device, non_blocking=True)
        with torch.inference_mode():
            views = forward_views(model, px, args.view_batch_size)
            if not isinstance(views, torch.Tensor) or views.dtype != torch.float32:
                raise ValueError("engine contract violated")
            if views.shape[0] != 8:
                raise ValueError(f"expected 8 TTA views, got {views.shape[0]}")
            mean = views.mean(dim=0, dtype=torch.float32).cpu().numpy()
        if mean.shape != (NUM_CLASSES, 1024, 1024):
            raise ValueError(f"unexpected mean logits shape {mean.shape}")
        np.save(outdir / f"{Path(name).stem}.npy", mean)
        argmax[i] = mean.argmax(axis=0).astype(np.uint8)
        del views, mean, px
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            rate = el / (i + 1)
            print(f"  {i+1}/{len(names)}  {el:.0f}s  "
                  f"({rate:.2f}s/img, eta {rate*(len(names)-i-1):.0f}s)", flush=True)

    np.save(outdir / "ARGMAX_ZERO.npy", argmax)
    (outdir / "SCORE_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "SOUPB_KD_RUN1_STEP3200_SCORE_CACHE",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "config": args.config,
        "test_dir": str(test_dir),
        "images": len(names),
        "tta": "D4-TTA8, raw logits FP32 mean over 8 views",
        "class_axis": "index i = train class i = official class i+1",
        "stored": "per-image FP32 mean logits [8,1024,1024] in <stem>.npy; "
                  "ARGMAX_ZERO.npy = argmax(mean,0) uint8 with NO bias",
        "elapsed_s": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"score cache done in {time.time()-t0:.0f}s -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
