#!/usr/bin/env python3
"""Evaluate the soup family: one D4-TTA8 per soup, ZERO + CHAMPION from cache.

Inference contract is the project's validated one for these bodies
(src.models.build_model with cwfa disabled, strict state load,
tta_anatomy_v2.forward_views, FP32 mean over 8 D4 views), identical to the
pipeline that produced the submitted soupB packages.

CHAMPION is applied AFTER the FP32 view mean, on the class axis, then argmax+1.
index i = train class i = official class i+1.

Reports flip% per soup: on these Stage23-lineage bodies a normal calibration
operator flips a few percent of pixels (Stage59 FIXED8D: 5.41%). A flip near
90% means the bias is destroying the prediction, not calibrating it.

Physical GPU2 only. Writes only under --outdir.
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
SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
TEST = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")
CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
ZERO = [0.0] * NUM_CLASSES
C = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
V13 = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 2.9215, -0.5]
VARIANTS = [("ZERO", ZERO), ("C", C), ("13", V13)]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pack(cdir: Path, names, outdir: Path, tag: str):
    from PIL import Image
    handles = {}
    for vn, vec in VARIANTS:
        zp = outdir / f"{tag}_{vn}.zip"
        handles[vn] = [vec, zp, zipfile.ZipFile(zp, "x", compression=zipfile.ZIP_STORED),
                       hashlib.sha256()]
    total = 0
    flips = {n: 0 for n, _ in VARIANTS}
    bg2fg = {n: 0 for n, _ in VARIANTS}
    for name in names:
        arr = np.load(cdir / f"{Path(name).stem}.npy")
        base = arr.argmax(0)
        total += base.size
        for vn, (vec, _, zf, dig) in handles.items():
            pred = base if vn == "ZERO" else (
                arr + np.asarray(vec, dtype=np.float32)[:, None, None]).argmax(0)
            if vn != "ZERO":
                moved = base != pred
                flips[vn] += int(moved.sum())
                bg2fg[vn] += int(((base == 0) & moved).sum())
            png = np.ascontiguousarray(pred.astype(np.uint8) + 1)
            buf = io.BytesIO()
            Image.fromarray(png, mode="L").save(buf, format="PNG")
            data = buf.getvalue()
            fn = f"{Path(name).stem}.png"
            zf.writestr(fn, data)
            dig.update(fn.encode() + b"\0" + data)
    out = {}
    for vn, (vec, zp, zf, dig) in handles.items():
        zf.close()
        out[vn] = {"vector": vec, "zip_path": str(zp), "zip_sha256": sha(zp),
                   "entries": len(names)}
        if vn != "ZERO":
            f = flips[vn]
            out[vn]["flip_pct"] = 100.0 * f / total
            out[vn]["bg_to_fg_pct_of_flips"] = 100.0 * bg2fg[vn] / f if f else 0.0
            print(f"  [{tag} {vn}] flip={out[vn]['flip_pct']:.4f}% "
                  f"BG->FG {out[vn]['bg_to_fg_pct_of_flips']:.1f}%", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--soup", action="append", required=True)
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
    from PIL import Image
    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import build_model, forward_views

    names = sorted((p.name for p in TEST.glob("*.png")),
                   key=lambda n: int(Path(n).stem.split("_")[-1]))
    if names != [f"test1_{i}.png" for i in range(1, 501)]:
        raise SystemExit("test1 names must be exactly test1_1..test1_500")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    results = {}

    for soup in args.soup:
        ck = Path(soup)
        if not ck.is_file():
            raise SystemExit(f"missing soup: {ck}")
        tag = ck.stem
        ck_sha = sha(ck)
        model = build_model(CONFIG, ck, args.device)
        cdir = outdir / f"scores_{tag}"
        cdir.mkdir()
        t1 = time.time()
        print(f"START {tag} sha={ck_sha[:16]}", flush=True)
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
                np.save(cdir / f"{Path(name).stem}.npy",
                        sc.cpu().numpy().astype(np.float32))
                del px, sc
                if i % 100 == 0:
                    el = time.time() - t1
                    print(f"  {tag} {i}/500 {el:.0f}s ({el/i:.2f}s/img)", flush=True)
        del model
        torch.cuda.empty_cache()
        if sha(ck) != ck_sha:
            raise SystemExit(f"soup changed during inference: {ck}")
        results[tag] = {"soup": str(ck), "soup_sha256": ck_sha,
                        "packages": pack(cdir, names, outdir, tag),
                        "elapsed_sec": round(time.time() - t1, 1)}
        print(f"DONE {tag} in {time.time()-t1:.0f}s", flush=True)

    (outdir / "EVAL_META.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "SOUP_FAMILY_HIGH_RESPONSE",
        "config": str(CONFIG),
        "tta": "D4 TTA8, FP32, autocast off, FP32 view mean",
        "model": "src.models.build_model, cwfa disabled, strict load",
        "class_axis": "index i = train class i = official class i+1",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndone -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
