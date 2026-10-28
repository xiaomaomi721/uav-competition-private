#!/usr/bin/env python3
"""B2: champion per-class audit + affine-calibration cache. GPU once, keep cache.

Inputs
------
  champion   : weighted_soups_20260917/wH_58_64_55b_40_40_20.pt  (0.698641 @b*)
  comparison : stage59_..._barrenAgri/best_miou.pt               (a model actually
               trained toward Barren+Agricultural, for error-overlap analysis)

Why
---
The 0.71 gap is 0.090872 class-IoU units. Whether that is a concentrated class
deficit or a global representation deficit decides the next GPU investment, and
the project's existing per-class numbers are single-view ZERO-bias while the
submission operator is champion-bias on test1, so they cannot answer it.

What this writes
----------------
For every val image, the FULL-RESOLUTION interpolated logits after a single-view
forward, exactly as src.inference.evaluate_model does it (bilinear to the GT
grid, align_corners=False). Saved as fp16 [8,1024,1024] so every downstream
question -- per-class IoU, FN/FP overlap between the two models, classwise
affine scale -- is answerable on CPU later without another forward.

The val convention is the project's own: splits/val.txt, 1399 stems, masks
mapped through official_to_train, ignore 255. NOT the D4-TTA8 test1 protocol.

This cache is to be KEPT (the previous round's caches were deleted to free disk
and that cost a re-forward for every follow-up question).

Physical GPU2 only. Writes only under --outdir.
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
IGNORE = 255
PHYSICAL_GPU_INDEX = 2
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
DATASET = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
R = Path("/home/jhy/rare_diag_20260915")
MODELS = {
    "champ": R / "weighted_soups_20260917/wH_58_64_55b_40_40_20.pt",
    "barrenAgri": (SNAP / "stage59_runs/stage59_w1_exp59_b3_cwfa_ce_regionC_"
                          "barrenAgri_seed42/best_miou.pt"),
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise SystemExit("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("require CUDA_DEVICE_ORDER=PCI_BUS_ID")

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    for p in ("/home/jhy/gpu2_day_queue", str(SNAP)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from src.transforms import normalize_to_tensor
    from src.label_mapping import official_to_train
    from tta_anatomy_v2 import build_model

    stems = [l.strip() for l in (SNAP / "splits/val.txt").read_text().split() if l.strip()]
    if len(stems) != 1399:
        raise SystemExit(f"expected 1399 val stems, got {len(stems)}")
    img_dir = DATASET / "train" / "images"
    mask_dir = DATASET / "train" / "masks"

    meta = {"created": datetime.now(timezone.utc).isoformat(),
            "scope": "B2_CHAMPION_PERCLASS_AND_AFFINE_CACHE",
            "val_split": str(SNAP / "splits/val.txt"), "val_count": len(stems),
            "protocol": "single-view, no TTA, bilinear to GT grid, align_corners=False",
            "label_mapping": "official_to_train, ignore 255",
            "saved": "fp16 [8,H,W] interpolated logits + uint8 GT, unwarped",
            "config": str(CONFIG), "models": {}}

    # GT once
    gt_dir = outdir / "gt"
    gt_dir.mkdir()
    for s in stems:
        with Image.open(mask_dir / f"{s}.png") as m:
            t = official_to_train(np.asarray(m).copy())
        np.save(gt_dir / f"{s}.npy", t.astype(np.uint8))
    print(f"GT saved: {len(stems)}", flush=True)

    for tag, path in MODELS.items():
        if not path.is_file():
            raise SystemExit(f"missing model: {path}")
        msha = sha(path)
        model = build_model(CONFIG, path, args.device)
        d = outdir / f"logits_{tag}"
        d.mkdir()
        t0 = time.time()
        print(f"START {tag} sha={msha[:16]}", flush=True)
        with torch.inference_mode():
            for i, s in enumerate(stems, 1):
                with Image.open(img_dir / f"{s}.png") as im:
                    img = np.asarray(im.convert("RGB")).copy()
                with Image.open(mask_dir / f"{s}.png") as m:
                    msk = official_to_train(np.asarray(m).copy())
                gt = torch.from_numpy(msk.astype(np.int64)).unsqueeze(0)
                px = normalize_to_tensor(img).unsqueeze(0).to(args.device)
                with torch.amp.autocast("cuda", enabled=False):
                    lg = model(pixel_values=px).logits
                    # exactly as src.inference.evaluate_model does it
                    lg = F.interpolate(lg, size=gt.shape[-2:], mode="bilinear",
                                       align_corners=False)
                if not torch.isfinite(lg).all():
                    raise SystemExit(f"nonfinite logits: {s}")
                np.save(d / f"{s}.npy",
                        lg[0].to(torch.float16).cpu().numpy())
                del px, lg, gt
                if i % 200 == 0:
                    el = time.time() - t0
                    print(f"  {tag} {i}/{len(stems)} {el:.0f}s ({el/i:.2f}s/img)",
                          flush=True)
        del model
        torch.cuda.empty_cache()
        if sha(path) != msha:
            raise SystemExit(f"model changed while inferring: {path}")
        meta["models"][tag] = {"path": str(path), "sha256": msha,
                               "elapsed_sec": round(time.time() - t0, 1)}
        print(f"DONE {tag} in {time.time()-t0:.0f}s", flush=True)

    (outdir / "B2_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
