#!/usr/bin/env python3
"""Precompute the 3-teacher TTA8 ensemble soft targets for the soft-KD student.

Design notes
------------
* Teachers: stage23 B + s62 + s58, all ZERO-bias (no leaderboard calibration is
  baked into the soft targets -- see 决策记录_20260915_第二轮评审定案.md).
* TTA8: the same 4-rotation x {plain, mirror} D4 protocol used by the 30.zip
  raw ensemble, so the target reproduces the combination logic that scored
  0.683237 rather than a weaker single-view approximation.
* Stored as softmax(logits / T) with T=2, laid out [H, W, C].

  [H, W, C] (channels-last) is deliberate: it lets the training loop reuse the
  project's own `apply_geometric_ops(image, mask)` helper to apply an IDENTICAL
  flip/rotate/crop to the image and to the stored teacher probabilities, which is
  what makes teacher-guided consistency training well-posed.

  Two storage knobs:

    --store fp16  (default) keeps the full soft distribution. uint8 quantises to
                  1/255, which can zero out low-probability classes -- exactly the
                  "second/third choice" information that makes KD worth doing.
                  At native resolution fp16 costs under 1 GB for 500 images.
    --store uint8  legacy: 1/255 quantisation, HxW = 1024. Costs ~4 GB.

    --resolution native (default) stores the network's own 256x256 logit grid.
    --resolution full            upsamples to 1024x1024 first (4x the bytes).

Writes only under --outdir.
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assert_gpu(device: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise RuntimeError("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("require CUDA_DEVICE_ORDER=PCI_BUS_ID")
    if device != "cuda:0":
        raise RuntimeError("require --device cuda:0")
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device required")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--teachers", required=True, help="JSON: {tag: {config, checkpoint}}")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--store", choices=["uint8", "fp16"], default="fp16",
                    help="uint8 quantises p to 1/255 (loses dark knowledge in "
                         "low-probability classes); fp16 keeps full precision.")
    ap.add_argument("--resolution", choices=["native", "full"], default="native",
                    help="native = the network's own 256x256 logit grid; full = "
                         "upsample to 1024x1024 before storing.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)
    prob_dir = outdir / "probs"
    prob_dir.mkdir()

    names = [l.strip() for l in Path(args.names).read_text().split() if l.strip()]
    teachers = json.loads(Path(args.teachers).read_text())
    T = float(args.temperature)

    assert_gpu(args.device)
    import torch
    if "/home/jhy/gpu2_day_queue" not in sys.path:
        sys.path.insert(0, "/home/jhy/gpu2_day_queue")
    from tta_anatomy_v2 import build_model, forward_views
    if SNAP not in sys.path:
        sys.path.insert(0, SNAP)
    from PIL import Image
    from src.transforms import normalize_to_tensor

    meta = {
        "schema": "KD_TEACHER_SOFT_TARGETS_V1",
        "scope": "SOFT_KD_TEACHER_CACHE",
        "created": datetime.now(timezone.utc).isoformat(),
        "temperature": T,
        "store": args.store,
        "resolution": args.resolution,
        "layout": "[H,W,C] channels-last so geometry can be applied jointly",
        "tta": "D4 TTA8 (4 rotations x plain/mirror), image-space FP32 mean",
        "bias": "ZERO (no leaderboard calibration in the soft targets)",
        "names": len(names),
        "teachers": {},
    }
    for tag, spec in teachers.items():
        meta["teachers"][tag] = {
            "config": spec["config"],
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": sha256_file(Path(spec["checkpoint"])),
        }
    (outdir / "TEACHER_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    models = {}
    for tag, spec in teachers.items():
        print(f"loading teacher {tag} ...", flush=True)
        models[tag] = build_model(spec["config"], spec["checkpoint"], args.device)
        for p in models[tag].parameters():
            p.requires_grad_(False)
    tags = list(teachers.keys())

    t0 = time.time()
    written = []
    for i, name in enumerate(names, 1):
        img = np.asarray(Image.open(Path(args.test_dir) / name).convert("RGB")).copy()
        px = normalize_to_tensor(img).unsqueeze(0).to(args.device, non_blocking=True)
        acc = None
        with torch.inference_mode():
            for tag in tags:
                views = forward_views(models[tag], px, 1)
                if not isinstance(views, torch.Tensor) or views.dtype != torch.float32:
                    raise ValueError("engine contract violated")
                mean = views.mean(dim=0, dtype=torch.float32)  # [C,H,W]
                acc = mean if acc is None else acc + mean
                del views, mean
            ens = acc / float(len(tags))
            if args.resolution == "full" and ens.shape[-1] != img.shape[0]:
                ens = torch.nn.functional.interpolate(
                    ens.unsqueeze(0), size=img.shape[:2], mode="bilinear",
                    align_corners=False).squeeze(0)
            soft = torch.softmax(ens / T, dim=0)             # [C,H,W]
            if args.store == "uint8":
                arr = (soft * 255.0).round().clamp(0, 255).to(torch.uint8)
            else:
                arr = soft.to(torch.float16)
            arr = arr.permute(1, 2, 0).contiguous().cpu().numpy()  # [H,W,C]
            del acc, ens, soft, px
        out = prob_dir / f"{Path(name).stem}.npy"
        np.save(out, arr)
        written.append({"name": name, "file": out.name,
                        "sum": float(np.asarray(arr, dtype=np.float64).sum()
                                     / (255.0 if arr.dtype == np.uint8 else 1.0)),
                        "shape": list(arr.shape)})
        if i % 50 == 0:
            print(f"  [{i}/{len(names)}] {time.time() - t0:.0f}s", flush=True)

    (outdir / "PROBS_MANIFEST.json").write_text(json.dumps({
        "count": len(written),
        "temperature": T,
        "teachers": tags,
        "items": written,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done {len(written)} soft targets in {time.time() - t0:.0f}s")
    total = sum(f.stat().st_size for f in prob_dir.glob("*.npy"))
    print(f"cache size {total / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
