#!/usr/bin/env python3
"""ATDA-71 finalize: fold b_star into the classifier bias and verify.

The winning model is a composed checkpoint (theta261 + alpha*delta) whose Bstar
submission PNGs have already been produced by _atda_export.py. This script:

  1. folds b_star into decode_head.classifier.bias (exact linear operator),
  2. verifies tensor-level that ONLY classifier.bias changed, and that its
     delta equals b_star up to fp32 rounding,
  3. forwards the folded model once over all 500 test1 images (D4 TTA8, no
     post-hoc bias) and compares argmax against the pre-fold Bstar zip's PNGs,
     pixel by pixel.

Folding is mathematically exact: mean_views(upsample(conv(x)+b)) =
mean_views(upsample(conv(x))) + b because b is per-class constant and both
upsampling and view-averaging are linear. The only residual difference is
fp32 non-associativity, which can flip an utterly negligible number of
boundary pixels; the script reports the exact count rather than pretending it
is zero.

Physical GPU2 only. Writes only under --outdir.
"""
from __future__ import annotations

import argparse
import hashlib
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
THETA261 = Path("/home/jhy/rare_diag_20260915/phase0_candidates_20260918/"
                "A_cleanSEM_M_k100_th209.pt")
THETA261_SHA256 = "9ffdd0beb8480a1efd361503d637d3d53cf716c808537c551fefae09f7fe2786"
THETA261_SCORE = 0.702836
CHAMPION_CONFIG = SNAP / "stage62_runs/stage62_w1_resume55b_pseudo500_seed42/config.yaml"
TEST = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集/images")
BSTAR = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]
CLASSIFIER_BIAS = "decode_head.classifier.bias"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path):
    import torch
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            c = ck.get(key)
            if isinstance(c, dict) and c and all(isinstance(v, torch.Tensor) for v in c.values()):
                return c
        if ck and all(isinstance(v, torch.Tensor) for v in ck.values()):
            return ck
    raise SystemExit(f"no tensor state dict in {path}")


def fold_state(state):
    import torch
    folded = {k: v.clone() for k, v in state.items()}
    bstar = torch.tensor(BSTAR, dtype=folded[CLASSIFIER_BIAS].dtype)
    folded[CLASSIFIER_BIAS].add_(bstar)
    return folded


def verify_tensor_level(pre, post):
    """Only classifier.bias changed; its delta equals b_star up to fp32 eps."""
    import torch
    changed = [k for k in pre if k != CLASSIFIER_BIAS and not torch.equal(post[k], pre[k])]
    delta = (post[CLASSIFIER_BIAS] - pre[CLASSIFIER_BIAS]).detach().cpu().numpy()
    want = np.asarray(BSTAR, dtype=delta.dtype)
    max_abs = float(np.abs(delta - want).max())
    return {
        "only_classifier_bias_changed": not changed,
        "unexpected_changed_tensors": changed[:10],
        "bias_delta_max_abs_vs_bstar": round(max_abs, 10),
        "bias_delta_is_bstar_within_fp32": max_abs < 1.0e-6,
    }


def read_zip_pngs(zpath: Path):
    from PIL import Image
    import io
    with zipfile.ZipFile(zpath) as zf:
        names = sorted(zf.namelist(), key=lambda n: int(Path(n).stem.split("_")[-1]))
        for name in names:
            arr = np.asarray(Image.open(io.BytesIO(zf.read(name))))
            yield name, arr


def forward_post(model, device):
    import torch
    from PIL import Image

    from src.transforms import normalize_to_tensor
    from tta_anatomy_v2 import forward_views

    names = sorted((p.name for p in TEST.glob("*.png")),
                   key=lambda n: int(Path(n).stem.split("_")[-1]))
    with torch.inference_mode():
        for name in names:
            with Image.open(TEST / name) as im:
                img = np.asarray(im.convert("RGB")).copy()
            px = normalize_to_tensor(img).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=False):
                sc = forward_views(model, px, 1).mean(dim=0, dtype=torch.float32)
            pred = sc.argmax(0).cpu().numpy().astype(np.uint8) + 1  # official 1..8
            yield name, pred
            del px, sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="composed pre-fold checkpoint")
    ap.add_argument("--prefold-zip", required=True, help="Bstar zip already produced")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise SystemExit("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("require CUDA_DEVICE_ORDER=PCI_BUS_ID")

    for p in ("/home/jhy/gpu2_day_queue", str(SNAP)):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from tta_anatomy_v2 import build_model

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)

    pre_state = load_state(ckpt)
    folded_state = fold_state(pre_state)
    tensor_check = verify_tensor_level(pre_state, folded_state)
    print("tensor-level:", json.dumps(tensor_check), flush=True)

    folded_path = outdir / (ckpt.stem + "_folded.pt")
    torch.save({"model": folded_state, "step": 0,
                "atda_final": {
                    "source": str(ckpt), "source_sha256": sha256_file(ckpt),
                    "base_theta261": str(THETA261),
                    "base_theta261_sha256": THETA261_SHA256,
                    "base_platform_score": THETA261_SCORE,
                    "folded_bstar": BSTAR,
                    "folded_into": CLASSIFIER_BIAS,
                }}, folded_path)

    model_post = build_model(CHAMPION_CONFIG, folded_path, args.device)

    total = 0
    mismatched = 0
    n_img = 0
    t0 = time.time()
    zip_gen = read_zip_pngs(Path(args.prefold_zip))
    for (name_pre, pre_pred), (name_post, post_pred) in zip(zip_gen,
                                                            forward_post(model_post, args.device)):
        assert name_pre == name_post
        n_img += 1
        total += pre_pred.size
        mismatched += int((pre_pred != post_pred).sum())
        if n_img % 100 == 0:
            print(f"  {n_img}/500 compared", flush=True)
    del model_post
    torch.cuda.empty_cache()

    frac = mismatched / total
    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "scope": "ATDA71_FINALIZE_BSTAR_FOLD",
        "winner_checkpoint": str(ckpt),
        "winner_checkpoint_sha256": sha256_file(ckpt),
        "prefold_zip": str(args.prefold_zip),
        "folded_checkpoint": str(folded_path),
        "folded_checkpoint_sha256": sha256_file(folded_path),
        "base_theta261": str(THETA261),
        "base_theta261_sha256": THETA261_SHA256,
        "base_platform_score": THETA261_SCORE,
        "bstar": BSTAR,
        "folded_into": CLASSIFIER_BIAS,
        "tensor_level_check": tensor_check,
        "pixel_level_check": {
            "images": n_img,
            "pixels": total,
            "mismatched_pixels": mismatched,
            "mismatch_frac": frac,
            "mismatch_pct": round(100.0 * frac, 8),
            "pixel_identical": mismatched == 0,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (outdir / "FINALIZE_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # hard gate on the real invariant: only classifier.bias moved, by b_star.
    if not (tensor_check["only_classifier_bias_changed"]
            and tensor_check["bias_delta_is_bstar_within_fp32"]):
        raise SystemExit("FOLD TENSOR-LEVEL CHECK FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
