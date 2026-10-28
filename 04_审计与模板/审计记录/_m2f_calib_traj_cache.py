#!/usr/bin/env python3
"""M2F Run1b calibration-response trajectory: ONE TTA8 per checkpoint, cached.

Why this exists
---------------
We know the ZERO trajectory of M2F Run1b (20k 0.680570 -> 40k 0.683170 ->
56k 0.684133) but the CALIBRATION-RESPONSE trajectory is completely unmeasured.
That response (R_C, R_FG020) is the whole question: M2F56 has R_C = 0.0967pp,
the weakest of any model in the project, and nobody has ever checked whether
that is a property of the architecture/init or a property of the training state.

This produces (Z, R) points, not just Z.

Contract
--------
Inference is the project's already-validated M2F Run1b protocol, reused verbatim
from pack_run1b_zero_v1.py: D4-TTA8 via tta_anatomy_v2.forward_views, autocast
off, FP32 mean over 8 views, M2F native semantic mixture (softmax(class)[...,:8]
x sigmoid(mask)). Bias is added to the class axis of the mean score afterwards.

One TTA8 per checkpoint; the per-image FP32 mean score is cached so ZERO and all
calibration variants come from the SAME forward. ZERO must reproduce the
already-submitted package byte-for-byte -- that is the free pipeline check.

Read-only w.r.t. training. Writes only under --outdir (must not exist).
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
from types import SimpleNamespace

import numpy as np

NUM_CLASSES = 8
IGNORE = 255
PHYSICAL_GPU_INDEX = 2
ROOT = Path("/home/jhy/rare_diag_20260915/m2f_gpu_20260915")
BUNDLE = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
TEST = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")
MODEL_DIR = "/home/jhy/models/segformer-b3-ade512"
FLOOR = int(1.5 * 1024**3)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def disk_guard(reserve=0):
    free = shutil.disk_usage(ROOT).free
    if free - reserve < FLOOR:
        raise RuntimeError(f"disk floor: free={free} reserve={reserve} floor={FLOOR}")


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
    ap.add_argument("--steps", type=int, nargs="+", default=[20000, 40000, 56000])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    assert_gpu(args.device)
    disk_guard(512 * 1024**2)

    for p in (str(ROOT / "py"), str(BUNDLE), "/home/jhy/gpu2_day_queue"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from PIL import Image
    from transformers import AutoModelForSemanticSegmentation
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores
    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import forward_views, VIEWS

    class M2FScores(torch.nn.Module):
        def __init__(self, ck):
            super().__init__()
            self.encoder = AutoModelForSemanticSegmentation.from_pretrained(
                MODEL_DIR, num_labels=8, ignore_mismatched_sizes=True,
                local_files_only=True).segformer
            self.head = B3M2FStyleHead(in_channels=(64, 128, 320, 512), num_classes=8,
                                       hidden_dim=256, num_queries=100, nhead=8,
                                       num_layers=6, dim_feedforward=1024,
                                       ignore_index=IGNORE)
            self.encoder.load_state_dict(ck["model"], strict=True)
            self.head.load_state_dict(ck["head"], strict=True)

        def forward(self, pixel_values):
            features = self.encoder(pixel_values=pixel_values,
                                    output_hidden_states=True).hidden_states
            o = self.head(features)
            # native semantic mixture, NOT log(score) and not a class-softmax of it
            return SimpleNamespace(logits=semantic_scores(o["class_logits"],
                                                          o["mask_logits"]))

    names = sorted((p.name for p in TEST.glob("*.png")),
                   key=lambda n: int(Path(n).stem.split("_")[-1]))
    if len(names) != 500:
        raise SystemExit(f"expected 500 test images, got {len(names)}")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    outdir.mkdir(parents=True)
    (outdir / "INFERENCE_CONFIG.json").write_text(json.dumps({
        "schema": "M2F_RUN1B_CALIB_TRAJECTORY_V1",
        "created": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": 2, "logical_device": "cuda:0",
        "test_dir": str(TEST), "test_count": len(names),
        "tta": "D4 TTA8", "views": VIEWS, "view_batch_size": 1,
        "precision": "FP32, autocast disabled; FP32 view mean",
        "semantic_synthesis": "softmax(class)[...,:8] x sigmoid(mask), einsum bqc,bqhw->bchw",
        "stored": "per-image FP32 mean score [8,1024,1024] in <stem>.npy",
        "bias_applied": "after the FP32 view mean, on the class axis",
        "class_axis": "index i = train class i = official class i+1",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    t0 = time.time()
    for step in args.steps:
        ckpt = ROOT / "formal_run1b/scores" / f"score_iter{step}.pt"
        if not ckpt.is_file():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        ck_sha = sha(ckpt)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if ck["data_iteration"] != step:
            raise SystemExit(f"step mismatch in {ckpt}")
        model = M2FScores(ck).to(args.device).eval()
        del ck

        cdir = outdir / f"score_{step}"
        cdir.mkdir()
        t1 = time.time()
        print(f"START {step} sha={ck_sha[:16]}", flush=True)
        with torch.inference_mode():
            for i, name in enumerate(names, 1):
                disk_guard(8 * 1024**2)
                with Image.open(TEST / name) as image:
                    img = np.asarray(image.convert("RGB")).copy()
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
                    el = time.time() - t1
                    print(f"  {step} {i}/500 {el:.0f}s "
                          f"({el/i:.2f}s/img, eta {el/i*(500-i):.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
        if sha(ckpt) != ck_sha:
            raise SystemExit(f"checkpoint changed while inferring: {ckpt}")
        print(f"DONE {step} in {time.time()-t1:.0f}s", flush=True)

    (outdir / "FORWARD_DONE.json").write_text(json.dumps({
        "steps": args.steps, "images": len(names),
        "elapsed_sec": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"score cache done in {time.time()-t0:.0f}s -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
