#!/usr/bin/env python3
"""s62 model-specific calibration: multi-axis combination + Agricultural step-down.

Follows directly from the 54-61 single-axis probe, where all four axes came back
monotone around the Stage23 champion vector:

    Agricultural  -0.4 -> +0.0810pp   (crosses the pre-registered +0.05pp bar)
    Barren        +0.4 -> +0.0330pp
    Vegetation    -0.4 -> +0.0283pp
    Water         -0.4 -> +0.0138pp

Two packages are produced:

  s62_combo4      all four favourable directions applied at once (-0.4 on
                  Agricultural / Vegetation / Water, +0.4 on Barren).
                  Naive additivity would give 0.686610 + 0.1561 = 0.688171;
                  the point of the package is to measure the interaction.
  s62_agri_m08    Agricultural -0.8 on its own, the pre-registered step after
                  the -0.4 probe cleared its threshold.

One forward pass per image; both packages (plus a champion control) come out of
the same 8-view TTA logits, so the marginal cost over the control is zero.

Writes only under --outdir.
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
CHAMPION = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
ZERO = [0.0] * NUM_CLASSES
CLASSES = ["Background", "Building", "Road", "Water",
           "Barren", "Vegetation", "Agricultural", "Vehicle"]


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


def adjust(vector, deltas):
    v = list(vector)
    for idx, d in deltas.items():
        v[idx] = round(v[idx] + d, 6)
    return v


def predict(mean_logits_t, vector):
    import torch
    bias = torch.tensor(vector, dtype=torch.float32, device=mean_logits_t.device)
    bias = bias.view(NUM_CLASSES, 1, 1)
    return (mean_logits_t + bias).argmax(dim=0).add(1).to(torch.uint8).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        raise SystemExit(f"outdir exists: {outdir}")
    outdir.mkdir(parents=True)
    names = [l.strip() for l in Path(args.names).read_text().split() if l.strip()]

    assert_gpu(args.device)
    import torch
    if "/home/jhy/gpu2_day_queue" not in sys.path:
        sys.path.insert(0, "/home/jhy/gpu2_day_queue")
    from tta_anatomy_v2 import build_model, forward_views
    SNAP = "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1"
    if SNAP not in sys.path:
        sys.path.insert(0, SNAP)
    from PIL import Image
    from src.transforms import normalize_to_tensor

    # index -> class: 3 Water, 4 Barren, 5 Vegetation, 6 Agricultural
    COMBO4 = adjust(CHAMPION, {6: -0.4, 5: -0.4, 3: -0.4, 4: +0.4})
    AGRI_M08 = adjust(CHAMPION, {6: -0.8})

    variants = [
        {"name": "s62_ref_champion", "kind": "control", "vector": list(CHAMPION),
         "note": "control; reproduces submitted 28 (0.686610)"},
        {"name": "s62_combo4", "kind": "combo", "vector": COMBO4,
         "deltas": {"Agricultural": -0.4, "Vegetation": -0.4, "Water": -0.4,
                    "Barren": +0.4},
         "naive_additive_expectation": 0.688171},
        {"name": "s62_agri_m08", "kind": "single_axis", "vector": AGRI_M08,
         "deltas": {"Agricultural": -0.8}},
    ]

    meta = {
        "scope": "DIAGNOSTIC_ONLY",
        "purpose": "s62 model-specific calibration: 4-axis combination and Agri -0.8",
        "started": datetime.now(timezone.utc).isoformat(),
        "body": "s62",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "champion": CHAMPION,
        "axis_semantics": "index i = train class i = official class i+1",
        "variants": variants,
    }
    (outdir / "RUN_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    model = build_model(args.config, args.checkpoint, args.device)

    handles, digests = {}, {}
    for v in variants:
        p = outdir / f"{v['name']}.zip"
        handles[v["name"]] = (p, zipfile.ZipFile(p, "x", compression=zipfile.ZIP_STORED))
        digests[v["name"]] = hashlib.sha256()

    t0 = time.time()
    for i, name in enumerate(names, 1):
        img = np.asarray(Image.open(Path(args.test_dir) / name).convert("RGB")).copy()
        px = normalize_to_tensor(img).unsqueeze(0).to(args.device, non_blocking=True)
        with torch.inference_mode():
            views = forward_views(model, px, 1)
            if not isinstance(views, torch.Tensor) or views.dtype != torch.float32:
                raise ValueError("engine contract violated")
            mean = views.mean(dim=0, dtype=torch.float32)
            for v in variants:
                arr = predict(mean, v["vector"])
                payload = io.BytesIO()
                Image.fromarray(arr, mode="L").save(payload, format="PNG")
                data = payload.getvalue()
                handles[v["name"]][1].writestr(name, data)
                digests[v["name"]].update(name.encode() + b"\0" + data)
            del views, mean, px
        if i % 100 == 0:
            print(f"  [{i}/{len(names)}] {time.time() - t0:.0f}s", flush=True)

    for _, h in handles.values():
        h.close()

    results = {}
    for v in variants:
        p, _ = handles[v["name"]]
        results[v["name"]] = {
            "kind": v["kind"], "vector": v["vector"], "note": v.get("note"),
            "deltas": v.get("deltas"),
            "naive_additive_expectation": v.get("naive_additive_expectation"),
            "zip_path": str(p), "zip_sha256": sha256_file(p),
            "entries": len(names),
            "png_content_sha256": digests[v["name"]].hexdigest(),
        }
    (outdir / "PACK_RESULT.json").write_text(
        json.dumps({"finished": datetime.now(timezone.utc).isoformat(),
                    "images": len(names), "packages": results},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done {len(variants)} packages in {time.time() - t0:.0f}s")
    for k, r in results.items():
        print(f"  {k:20s} {r['png_content_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
