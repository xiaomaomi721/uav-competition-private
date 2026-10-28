#!/usr/bin/env python3
"""Regenerate the target reference cache for theta261.

Saves the D4 TTA8 canonical NATIVE 256x256 logits (stride-4, FP32 view mean,
inverse-transformed to canonical, no upsampling, no bilinear), one fp16 array
per test1 image. This is the fixed anchor for the ATDA target branch.

theta261 = phase0_candidates_20260918/A_cleanSEM_M_k100_th209.pt
           platform score 0.702836 (package 261)

Physical GPU2 only. Writes only under --outdir/scores and its meta files.
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
LOGIT_GRID = 256
PHYSICAL_GPU_INDEX = 2
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
THETA261 = Path("/home/jhy/rare_diag_20260915/phase0_candidates_20260918/"
                "A_cleanSEM_M_k100_th209.pt")
THETA261_SHA256 = "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786"
THETA261_SCORE = 0.702836
CHAMPION_CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
TEST_IMAGES = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise SystemExit("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("require CUDA_DEVICE_ORDER=PCI_BUS_ID")

    for extra in ("/home/jhy/gpu2_day_queue", str(SNAP)):
        if extra not in sys.path:
            sys.path.insert(0, extra)

    import torch
    from PIL import Image

    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import build_model
    from _atda_train import native_tta8_mean_logits  # exact same native TTA8 as training

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)

    if sha256_file(THETA261) != THETA261_SHA256:
        raise SystemExit("theta261 sha mismatch")

    names = sorted((p.name for p in TEST_IMAGES.glob("*.png")),
                   key=lambda n: int(Path(n).stem.split("_")[-1]))
    if names != [f"test1_{i}.png" for i in range(1, 501)]:
        raise SystemExit("target dir is not exactly test1_1..test1_500")

    model = build_model(CHAMPION_CONFIG, THETA261, args.device)
    print(f"theta261 loaded sha={THETA261_SHA256[:16]}, {len(names)} images", flush=True)

    (outdir / "FORWARD_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "REF261_NATIVE256_D4_TTA8_CACHE",
        "theta261": str(THETA261),
        "theta261_sha256": THETA261_SHA256,
        "theta261_platform_score": THETA261_SCORE,
        "config": str(CHAMPION_CONFIG),
        "test_dir": str(TEST_IMAGES),
        "test_count": len(names),
        "protocol": "D4 TTA8, native stride-4 logits (256x256), FP32 view mean, "
                    "inverse-transformed to canonical, no upsampling, autocast off",
        "stored": "fp16 [8,256,256] canonical mean logits per image",
        "target_dir_verified": names == [f"test1_{i}.png" for i in range(1, 501)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    d = outdir / "scores"
    d.mkdir()
    t0 = time.time()
    with torch.inference_mode():
        for i, name in enumerate(names, 1):
            with Image.open(TEST_IMAGES / name) as im:
                img = np.asarray(im.convert("RGB")).copy()
            if img.shape != (1024, 1024, 3):
                raise SystemExit(f"bad shape {name}: {img.shape}")
            px = normalize_to_tensor(img).unsqueeze(0).to(args.device)
            sc = native_tta8_mean_logits(model, px, args.device)
            if not torch.isfinite(sc).all():
                raise SystemExit(f"nonfinite: {name}")
            if sc.shape != (NUM_CLASSES, LOGIT_GRID, LOGIT_GRID):
                raise SystemExit(f"bad score shape {sc.shape}")
            np.save(d / f"{Path(name).stem}.npy",
                    sc.to(torch.float16).cpu().numpy())
            del px, sc
            if i % 100 == 0:
                el = time.time() - t0
                print(f"  {i}/500 {el:.0f}s ({el/i:.2f}s/img, eta {el/i*(500-i):.0f}s)",
                      flush=True)

    if sha256_file(THETA261) != THETA261_SHA256:
        raise SystemExit("theta261 changed while inferring")
    (outdir / "FORWARD_DONE.json").write_text(json.dumps({
        "images": len(names),
        "theta261_sha256": THETA261_SHA256,
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {time.time()-t0:.0f}s -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
