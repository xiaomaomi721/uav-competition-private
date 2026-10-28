#!/usr/bin/env python3
"""Champion test1 TTA8 forward, score cache KEPT permanently.

Purpose
-------
B2's val audit found three live affine-scale signals on the champion:

    Barren        a=0.80   +0.3741pp val   (P 0.555 / R 0.878 -> over-predicts)
    Agricultural  a=0.80   +0.1961pp val   (P 0.779 / R 0.965 -> over-predicts)
    Background    a=1.20   +0.1714pp val   (under-predicts)

Every one of them extends the operator the project already knows works, along
the sign the audit predicts. Testing any of them on test1 needs the champion's
raw per-pixel score, which the earlier TTA8 caches no longer contain (they were
deleted to free disk, and that decision cost a re-forward for every follow-up).

So this writes the full-resolution mean score for each test1 image and KEEPS it.
Everything downstream -- ZERO, b*, any (a_c) or (a_c, b_c) -- is then pure CPU.

Protocol: the validated one for this body. src.models.build_model with CWFA
disabled, tta_anatomy_v2.forward_views, autocast off, FP32 mean over 8 D4 views.
Score shape [8,1024,1024], float32 saved as float16.

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
PHYSICAL_GPU_INDEX = 2
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
TEST = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")
CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
CHAMP = Path("/home/jhy/rare_diag_20260915/weighted_soups_20260917/"
             "wH_58_64_55b_40_40_20.pt")


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
    if not CHAMP.is_file():
        raise SystemExit(f"missing champion: {CHAMP}")
    ck_sha = sha(CHAMP)

    for p in ("/home/jhy/gpu2_day_queue", str(SNAP)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from PIL import Image
    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import build_model, forward_views

    names = sorted((p.name for p in TEST.glob("*.png")),
                   key=lambda n: int(Path(n).stem.split("_")[-1]))
    if names != [f"test1_{i}.png" for i in range(1, 501)]:
        raise SystemExit("test1 names must be exactly test1_1..test1_500")

    (outdir / "FORWARD_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "CHAMPION_TEST1_TTA8_SCORE_CACHE",
        "champion": str(CHAMP), "champion_sha256": ck_sha,
        "config": str(CONFIG), "test_dir": str(TEST), "test_count": len(names),
        "champion_score": 0.698641, "operator_at_score": "b*",
        "protocol": "D4 TTA8, FP32, autocast off, FP32 view mean",
        "stored": "fp16 [8,1024,1024] mean score per image",
        "note": "KEEP THIS CACHE -- every affine/scale/bias candidate is CPU-only "
                "from here",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    model = build_model(CONFIG, CHAMP, args.device)
    print(f"champion loaded sha={ck_sha[:16]}, {len(names)} images", flush=True)

    d = outdir / "scores"
    d.mkdir()
    t0 = time.time()
    with torch.inference_mode():
        for i, name in enumerate(names, 1):
            with Image.open(TEST / name) as im:
                img = np.asarray(im.convert("RGB")).copy()
            if img.shape != (1024, 1024, 3):
                raise SystemExit(f"bad shape {name}: {img.shape}")
            px = normalize_to_tensor(img).unsqueeze(0).to(args.device)
            with torch.amp.autocast("cuda", enabled=False):
                sc = forward_views(model, px, 1).mean(dim=0, dtype=torch.float32)
            if not torch.isfinite(sc).all():
                raise SystemExit(f"nonfinite: {name}")
            if sc.shape != (NUM_CLASSES, 1024, 1024):
                raise SystemExit(f"bad score shape {sc.shape}")
            np.save(d / f"{Path(name).stem}.npy",
                    sc.to(torch.float16).cpu().numpy())
            del px, sc
            if i % 100 == 0:
                el = time.time() - t0
                print(f"  {i}/500 {el:.0f}s ({el/i:.2f}s/img, "
                      f"eta {el/i*(500-i):.0f}s)", flush=True)

    if sha(CHAMP) != ck_sha:
        raise SystemExit("champion changed while inferring")
    (outdir / "FORWARD_DONE.json").write_text(json.dumps({
        "images": len(names), "champion_sha256": ck_sha,
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {time.time()-t0:.0f}s -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
