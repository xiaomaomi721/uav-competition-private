#!/usr/bin/env python3
"""Create one exact native D4 source-holdout cache for theta261/theta278.

No target GT is referenced.  GT is deliberately outside this script.  For the
278 pass, per-pixel D4 disagreement counts are retained from the same forward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _r2_contracts import (CONFIG, DATA, EXPECTED_FILE_SHA, NUM_CLASSES, OUT,
                           SNAP, THETA261, THETA278, canonical_state_sha,
                           d4_forward, d4_inverse, load_state, sha256_file,
                           sha256_lines, write_json)


def build_model(checkpoint: Path, device: str):
    import torch
    from src.models import build_model as make_model
    from src.utils import load_config
    cfg = load_config(str(CONFIG))
    cfg["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0,
                   "norm_eps": 1e-12, "start_step": 1}
    state, _ = load_state(checkpoint)
    model = make_model(cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), canonical_state_sha(state)


def views_and_mean(model, px):
    import torch
    views = []
    with torch.inference_mode():
        for i in range(8):
            with torch.amp.autocast("cuda", enabled=False):
                logits = model(pixel_values=d4_forward(px, i)).logits.float()
            logits = d4_inverse(logits, i)[0]
            if tuple(logits.shape) != (8, 256, 256):
                raise RuntimeError(f"bad native logits shape {tuple(logits.shape)}")
            views.append(logits)
    stacked = torch.stack(views)
    return stacked, stacked.mean(0, dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=("theta261", "theta278"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    physical = os.environ.get("CUDA_VISIBLE_DEVICES")
    if physical not in ("2", "3") or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("require physical GPU2/3 mapping")
    if physical != "2":
        raise SystemExit("source-holdout cache is scheduled on physical GPU2")
    for path in (str(SNAP), "/home/jhy/gpu2_day_queue"):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch
    from PIL import Image
    from src.transforms import normalize_to_tensor

    checkpoint = THETA261 if args.tag == "theta261" else THETA278
    expected_sha = EXPECTED_FILE_SHA[args.tag]
    if sha256_file(checkpoint) != expected_sha:
        raise SystemExit("checkpoint SHA mismatch")
    inventory = OUT / "00_inventory"
    split = json.loads((inventory / "SOURCE_AUDIT_SPLIT.json").read_text())
    names = split["R1099"] + split["H300"]
    if len(names) != 1399 or len(set(names)) != 1399:
        raise SystemExit("bad frozen source-holdout list")
    out = OUT / "01_holdout_cache" / args.tag
    if out.exists():
        raise SystemExit(f"outdir exists: {out}")
    scores = out / "scores"
    scores.mkdir(parents=True)
    d4_dir = out / "d4_disagreement_count" if args.tag == "theta278" else None
    if d4_dir is not None:
        d4_dir.mkdir()

    model, canonical_sha = build_model(checkpoint, args.device)
    image_dir = DATA / "train/images"
    rows, quant_diff, quant_pixels = [], 0, 0
    t0 = time.time()
    for j, stem in enumerate(names, 1):
        with Image.open(image_dir / f"{stem}.png") as im:
            image = np.asarray(im.convert("RGB")).copy()
        px = normalize_to_tensor(image).unsqueeze(0).to(args.device)
        views, mean = views_and_mean(model, px)
        if not bool(torch.isfinite(mean).all()):
            raise RuntimeError(f"nonfinite {stem}")
        half = mean.half()
        quant_diff += int((mean.argmax(0) != half.float().argmax(0)).sum())
        quant_pixels += 256 * 256
        np.save(scores / f"{stem}.npy", half.cpu().numpy())
        pred = mean.argmax(0)
        disagree = (views.argmax(1) != pred[None]).to(torch.uint8).sum(0)
        if d4_dir is not None:
            np.save(d4_dir / f"{stem}.npy", disagree.cpu().numpy())
        rows.append({"image": f"{stem}.png", "d4_disagreement_mean": float(disagree.float().mean() / 8.0),
                     "d4_disagreement_p95": float(torch.quantile(disagree.float() / 8.0, .95)),
                     "d4_any_disagree_frac": float((disagree > 0).float().mean())})
        del px, views, mean, half, disagree
        if j % 100 == 0:
            print(f"{args.tag} {j}/1399 {time.time()-t0:.0f}s", flush=True)

    sanity = []
    fixed = [names[i] for i in (0, 1, 2, 99, 499, 899, 1397, 1398)]
    for stem in fixed:
        with Image.open(image_dir / f"{stem}.png") as im:
            image = np.asarray(im.convert("RGB")).copy()
        px = normalize_to_tensor(image).unsqueeze(0).to(args.device)
        _, live = views_and_mean(model, px)
        cached = torch.from_numpy(np.load(scores / f"{stem}.npy").astype(np.float32)).to(args.device)
        sanity.append({"image": f"{stem}.png", "max_abs": float((live-cached).abs().max()),
                       "argmax_disagree_frac": float((live.argmax(0) != cached.argmax(0)).float().mean())})
    meta = {"created": datetime.now(timezone.utc).isoformat(),
            "scope": "ROUND2_SOURCE_HOLDOUT_RAW_NATIVE256_D4_TTA8",
            "tag": args.tag, "checkpoint": str(checkpoint),
            "checkpoint_sha256": expected_sha, "canonical_state_sha256": canonical_sha,
            "config": str(CONFIG), "config_sha256": sha256_file(CONFIG),
            "dataset": str(image_dir), "count": 1399,
            "ordered_stems_sha256": sha256_lines(names),
            "source_split_manifest": str(inventory / "SOURCE_AUDIT_SPLIT.json"),
            "protocol": "D4 TTA8 exact inverse at native [8,256,256]; FP32 view mean; autocast off; raw/no Bstar",
            "stored": "fp16 [8,256,256] raw mean logits",
            "d4_pixel_map": "uint8 0..8 disagreement count at native grid" if d4_dir else None,
            "fp16_quantization": {"pixels": quant_pixels, "argmax_disagree_pixels": quant_diff,
                                    "argmax_disagree_frac": quant_diff / quant_pixels},
            "d4_disagreement": rows, "live_vs_cache_sanity": sanity,
            "physical_gpu": int(physical), "device": args.device,
            "runtime_sec": round(time.time()-t0, 1), "code_sha256": sha256_file(Path(__file__)),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "test2_ground_truth_used": False, "platform_submission_performed": False}
    write_json(out / "META.json", meta)
    write_json(out / "FORWARD_DONE.json", {"tag": args.tag, "count": 1399,
                                             "checkpoint_sha256": expected_sha,
                                             "runtime_sec": meta["runtime_sec"]})
    print(json.dumps({"out": str(out), "quantization": meta["fp16_quantization"],
                      "sanity": sanity}, indent=2))


if __name__ == "__main__":
    main()
