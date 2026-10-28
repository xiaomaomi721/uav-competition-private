#!/usr/bin/env python3
"""Generate single-scale validation predictions for error-morphology auditing.

The training pipeline's own `evaluate_model` scores on a single scale (no TTA),
so single-scale is the right reference for "what does this model actually get
wrong on data it has labels for".

Memory-frugal on purpose: the KD training job is holding GPU2, so this uses
AMP, batch size 1, and drops the model between bodies.

Writes <outdir>/<tag>/*.png (uint8, official class ids 1-8).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PHYSICAL_GPU_INDEX = 2
SNAP = "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bodies", required=True, help="JSON {tag:{config,checkpoint}}")
    ap.add_argument("--val-split", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise RuntimeError("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("require CUDA_DEVICE_ORDER=PCI_BUS_ID")

    import json
    import torch
    import torch.nn.functional as F
    for p in (SNAP, "/home/jhy/gpu2_day_queue"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from PIL import Image
    from src.models import build_model, load_model_checkpoint
    from src.transforms import normalize_to_tensor

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bodies = json.loads(Path(args.bodies).read_text())
    stems = [Path(l.strip()).stem for l in Path(args.val_split).read_text().split()
             if l.strip()]
    if args.limit:
        stems = stems[: args.limit]
    img_dir = Path(args.dataset_root) / "train" / "images"
    print(f"val images: {len(stems)}", flush=True)

    for tag, spec in bodies.items():
        dst = outdir / tag
        dst.mkdir(exist_ok=True)
        model = build_model(spec["config"]).to(args.device)
        load_model_checkpoint(model, Path(spec["checkpoint"]), args.device)
        model.eval()
        t0 = time.time()
        with torch.inference_mode():
            for i, stem in enumerate(stems, 1):
                with Image.open(img_dir / f"{stem}.png") as h:
                    img = np.asarray(h.convert("RGB")).copy()
                px = normalize_to_tensor(img).unsqueeze(0).to(args.device)
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(pixel_values=px).logits
                logits = F.interpolate(logits, size=img.shape[:2], mode="bilinear",
                                       align_corners=False)
                pred = logits.argmax(1)[0].add(1).to(torch.uint8).cpu().numpy()
                Image.fromarray(pred, mode="L").save(dst / f"{stem}.png")
                del px, logits, pred
                if i % 200 == 0:
                    print(f"  [{tag} {i}/{len(stems)}] {time.time()-t0:.0f}s", flush=True)
        print(f"[{tag}] done in {time.time()-t0:.0f}s -> {dst}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("VAL_PRED_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
