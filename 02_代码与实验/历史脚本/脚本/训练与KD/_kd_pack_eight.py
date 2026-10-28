#!/usr/bin/env python3
"""KD Run 1: 4 checkpoints x {zero, CHAMPION} -> 8 submission packages.

Reuses the inference contract already validated by `_s62_combo_probe.py`
(tta_anatomy_v2.forward_views + src.transforms.normalize_to_tensor, TTA8,
FP32 view mean) and the same CHAMPION vector, so these packages are directly
comparable with submissions 07-28.

Read-only with respect to training. Writes only under --outdir (must not exist).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
PHYSICAL_GPU_INDEX = 2
SNAP = "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1"
CHAMPION = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
CHECKPOINTS = [3200, 4800, 6400, 8000]
VARIANTS = [("zero", [0.0] * NUM_CLASSES), ("champion", CHAMPION)]


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


def predict(mean_logits_t, vector):
    import torch
    bias = torch.tensor(vector, dtype=torch.float32, device=mean_logits_t.device)
    return (mean_logits_t + bias.view(NUM_CLASSES, 1, 1)).argmax(dim=0) \
        .add(1).to(torch.uint8).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-dir", required=True, help="dir with checkpoint_step<N>.pt")
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="override checkpoints as name=path (repeatable); used by the "
                         "s62 reproduction check so it runs through THIS same code path")
    ap.add_argument("--variants", default="both", choices=("both", "zero", "champion"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)
    names = [l.strip() for l in Path(args.names).read_text().split() if l.strip()]

    assert_gpu(args.device)
    import torch
    for p in ("/home/jhy/gpu2_day_queue", SNAP):
        if p not in sys.path:
            sys.path.insert(0, p)
    from PIL import Image
    from tta_anatomy_v2 import build_model, forward_views
    from src.transforms import normalize_to_tensor

    if args.checkpoint:
        checkpoints = {}
        for spec in args.checkpoint:
            name, _, path = spec.partition("=")
            path = Path(path)
            if not path.is_file():
                raise SystemExit(f"missing checkpoint: {path}")
            checkpoints[name] = {"path": str(path), "sha256": sha256_file(path)}
    else:
        checkpoints = {}
        for step in CHECKPOINTS:
            path = Path(args.runs_dir) / f"checkpoint_step{step}.pt"
            if not path.is_file():
                raise SystemExit(f"missing checkpoint: {path}")
            checkpoints[step] = {"path": str(path), "sha256": sha256_file(path)}

    variants = VARIANTS
    if args.variants == "zero":
        variants = [("zero", [0.0] * NUM_CLASSES)]
    elif args.variants == "champion":
        variants = [("champion", CHAMPION)]

    meta = {
        "scope": "KD_RUN1_S62INIT_EIGHT_PACKAGES",
        "body": "s62-init raw three-teacher soft KD, Run 1",
        "started": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "checkpoints": checkpoints,
        "champion": CHAMPION,
        "variants": [v for v, _ in variants],
        "axis_semantics": "index i = train class i = official class i+1",
        "note": "zero = no bias; champion = historical CHAMPION vector, a shared "
                "diagnostic coordinate, NOT a model-specific optimum",
    }
    (outdir / "RUN_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    handles, digests = {}, {}
    for step in checkpoints:
        for vname, _ in variants:
            key = f"s{step}_{vname}"
            p = outdir / f"{key}.zip"
            handles[key] = (p, zipfile.ZipFile(p, "x", compression=zipfile.ZIP_STORED))
            digests[key] = hashlib.sha256()

    t0 = time.time()
    for step in checkpoints:
        ckpt = checkpoints[step]["path"]
        model = build_model(args.config, ckpt, args.device)
        for i, name in enumerate(names, 1):
            img = np.asarray(Image.open(Path(args.test_dir) / name).convert("RGB")).copy()
            px = normalize_to_tensor(img).unsqueeze(0).to(args.device, non_blocking=True)
            with torch.inference_mode():
                views = forward_views(model, px, 1)
                if not isinstance(views, torch.Tensor) or views.dtype != torch.float32:
                    raise ValueError("engine contract violated")
                mean = views.mean(dim=0, dtype=torch.float32)
                for vname, vector in variants:
                    key = f"s{step}_{vname}"
                    payload = io.BytesIO()
                    Image.fromarray(predict(mean, vector), mode="L").save(
                        payload, format="PNG")
                    data = payload.getvalue()
                    handles[key][1].writestr(name, data)
                    digests[key].update(name.encode() + b"\0" + data)
                del views, mean, px
            if i % 100 == 0:
                print(f"  [step {step} {i}/{len(names)}] {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
        print(f"[step {step}] done at {time.time() - t0:.0f}s", flush=True)

    for _, h in handles.values():
        h.close()

    results = {}
    for step in checkpoints:
        for vname, vector in variants:
            key = f"s{step}_{vname}"
            p, _ = handles[key]
            results[key] = {"step": step, "variant": vname, "vector": vector,
                            "zip_path": str(p), "zip_sha256": sha256_file(p),
                            "entries": len(names),
                            "png_content_sha256": digests[key].hexdigest()}
    (outdir / "PACK_RESULT.json").write_text(
        json.dumps({"finished": datetime.now(timezone.utc).isoformat(),
                    "images": len(names), "packages": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done {len(results)} packages in {time.time() - t0:.0f}s")
    for k, r in results.items():
        print(f"  {k:16s} {r['zip_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
