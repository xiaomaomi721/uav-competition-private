#!/usr/bin/env python3
"""#103 (Stage66 SEMEDA+CWFA best@51800): ONE D4-TTA8, cached for CPU re-bias.

Why #103 and not the others
---------------------------
After 102-105 came back (0.676086 / 0.680750 / 0.666864 / 0.676150), an upper
bound check with the strongest R_C ever measured on a single model (Paramwise01,
1.5398pp) settles three of them:

    102: 0.676086 + 0.015398 = 0.691484  < record 0.693096
    104: 0.666864 + 0.015398 = 0.682262  < record
    105: 0.676150 + 0.015398 = 0.691548  < record

No bias vector can save those three; their calibration response is irrelevant.
Only #103 is still arithmetically alive:

    0.680750 + 0.015398 = 0.696148  > record
    needed R_C to tie the record = 0.693096 - 0.680750 = 1.2346pp

and the SEMEDA family has been measured at R_C = 1.2676pp (Stage47 C8), which
clears that bar. So this is the single measurement that can still matter.

This is NOT a replay of a previous experiment: the historical SEMEDA calibration
number (Exp47, 0.690776) comes from stage47_w1_exp47_b3_semeda_formal_seed42,
which has NO CWFA. #103 is stage66 = Exp47's config + CWFA delayed from 28001.
Different checkpoint, different recipe.

Inference path is the project's validated one for this checkpoint, reused
verbatim from pack_zero_stage66_best51800.py: src.models.build_model (cwfa
disabled) + tta_anatomy_v2.forward_views, FP32 mean over 8 views. Scores are
cached so ZERO and every calibration variant come from the SAME forward.

ZERO is compared pixel-for-pixel against the already-scored #103 package as a
pipeline check.

Physical GPU2 only. Read-only w.r.t. training. Writes only under --outdir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
PHYSICAL_GPU_INDEX = 2
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
TEST = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")
S66 = SNAP / "stage66_runs/stage66_formal_b3_1024_ce_semeda_l05_rarecrop_cwfa_p03_e4_s28001_seed42"
CKPT = S66 / "best_miou.pt"
CKPT_SHA = "c50d4133d35d690b9735d8710c6eeeb50844db5924394789a00f9ee7eaa03331"
FLOOR = int(1.5 * 1024**3)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
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
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    assert_gpu(args.device)

    if not CKPT.is_file():
        raise SystemExit(f"missing checkpoint: {CKPT}")
    got = sha(CKPT)
    if got != CKPT_SHA:
        raise SystemExit(f"checkpoint sha256 {got} != {CKPT_SHA}")

    for p in ("/home/jhy/gpu2_day_queue", str(SNAP)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from PIL import Image
    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import build_model, forward_views, VIEWS

    names = sorted((p.name for p in TEST.glob("*.png")),
                   key=lambda n: int(Path(n).stem.split("_")[-1]))
    if names != [f"test1_{i}.png" for i in range(1, 501)]:
        raise SystemExit("test1 names must be exactly test1_1..test1_500")

    outdir.mkdir(parents=True)
    (outdir / "INFERENCE_CONFIG.json").write_text(json.dumps({
        "schema": "STAGE66_51800_CALIBRATION_SCORE_CACHE_V1",
        "created": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CKPT), "checkpoint_sha256": got,
        "config": str(S66 / "config.yaml"),
        "config_sha256": sha(S66 / "config.yaml"),
        "physical_gpu": 2, "logical_device": "cuda:0",
        "test_dir": str(TEST), "test_count": len(names),
        "tta": "D4 TTA8", "views": VIEWS, "view_batch_size": 1,
        "precision": "FP32, autocast disabled; FP32 view mean",
        "model": "src.models.build_model, cwfa disabled, strict state load",
        "stored": "per-image FP32 mean score [8,1024,1024] in <stem>.npy",
        "bias_applied": "after the FP32 view mean, on the class axis",
        "class_axis": "index i = train class i = official class i+1",
        "note": "this checkpoint was scored as package #103 ZERO = 0.680750",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    model = build_model(S66 / "config.yaml", CKPT, args.device)
    print(f"model loaded from {CKPT.name} sha={got[:16]}, {len(names)} images, D4-TTA8",
          flush=True)

    free = shutil.disk_usage(outdir).free
    if free < FLOOR:
        raise SystemExit(f"disk floor: free={free}")

    cdir = outdir / "scores"
    cdir.mkdir()
    t0 = time.time()
    with torch.inference_mode():
        for i, name in enumerate(names, 1):
            with Image.open(TEST / name) as im:
                img = np.asarray(im.convert("RGB")).copy()
            if img.shape != (1024, 1024, 3):
                raise SystemExit(f"unexpected image shape {name}: {img.shape}")
            px = normalize_to_tensor(img).unsqueeze(0).to(args.device)
            with torch.amp.autocast("cuda", enabled=False):
                views = forward_views(model, px, 1)
                scores = views.mean(dim=0, dtype=torch.float32)
            if not torch.isfinite(scores).all():
                raise SystemExit(f"nonfinite scores: {name}")
            if scores.shape != (NUM_CLASSES, 1024, 1024):
                raise SystemExit(f"unexpected score shape {scores.shape}")
            np.save(cdir / f"{Path(name).stem}.npy",
                    scores.cpu().numpy().astype(np.float32))
            del px, views, scores
            if i % 50 == 0:
                el = time.time() - t0
                print(f"  {i}/500 {el:.0f}s ({el/i:.2f}s/img, "
                      f"eta {el/i*(500-i):.0f}s)", flush=True)

    if sha(CKPT) != CKPT_SHA:
        raise SystemExit("checkpoint changed while inferring")

    (outdir / "FORWARD_DONE.json").write_text(json.dumps({
        "images": len(names), "checkpoint_sha256": CKPT_SHA,
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"score cache done in {time.time()-t0:.0f}s -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
