#!/usr/bin/env python3
"""FINAL INVARIANT: is the cached FP16 map the *distribution* we mean to distil?

argmax agreement (99.9981%) shows the decisions match, but many different
distributions share an argmax. KD learns the full distribution, so it must be
checked directly.

The specific failure this guards against:

    correct : softmax( mean_i(logits_i) / T )      <- mean over teachers in LOGIT space
    wrong   : mean_i( softmax(logits_i / T) )      <- mean over teachers in PROB space

Both have near-identical argmax, so only a distribution-level test catches it.

Teachers are loaded ONE AT A TIME so peak GPU memory stays near a single model --
this runs while the KD job holds the GPU, and must not OOM it.

Read-only with respect to the training run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

NUM_CLASSES = 8
PHYSICAL_GPU_INDEX = 2
SNAP = "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft-dir", required=True)
    ap.add_argument("--teachers", required=True, help="JSON {tag:{config,checkpoint}}")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise RuntimeError("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("require CUDA_DEVICE_ORDER=PCI_BUS_ID")

    import torch
    if "/home/jhy/gpu2_day_queue" not in sys.path:
        sys.path.insert(0, "/home/jhy/gpu2_day_queue")
    if SNAP not in sys.path:
        sys.path.insert(0, SNAP)
    from PIL import Image
    from tta_anatomy_v2 import build_model, forward_views
    from src.transforms import normalize_to_tensor

    prob_dir = Path(args.soft_dir) / "probs"
    # test_names.txt entries already carry the .png suffix
    all_names = [l.strip() for l in Path(args.names).read_text().split() if l.strip()]
    all_names = [Path(n).stem for n in all_names]
    # deterministic spread across the set, not just the first N
    step = max(1, len(all_names) // args.count)
    names = all_names[::step][: args.count]
    print(f"checking {len(names)} images: {names[:5]} ...", flush=True)

    teachers = json.loads(Path(args.teachers).read_text())
    T = float(args.temperature)

    imgs = {}
    for n in names:
        with Image.open(Path(args.test_dir) / f"{n}.png") as h:
            imgs[n] = np.asarray(h.convert("RGB")).copy()

    # accumulate mean LOGITS across teachers, loading one model at a time
    acc = {n: None for n in names}
    for tag, spec in teachers.items():
        print(f"  teacher {tag} ...", flush=True)
        model = build_model(spec["config"], spec["checkpoint"], args.device)
        with torch.inference_mode():
            for n in names:
                px = normalize_to_tensor(imgs[n]).unsqueeze(0).to(args.device)
                views = forward_views(model, px, 1)          # [8,C,H,W]
                mean = views.mean(dim=0, dtype=torch.float32)
                acc[n] = mean if acc[n] is None else acc[n] + mean
                del views, mean, px
        del model
        torch.cuda.empty_cache()

    nt = len(teachers)
    summary = {"count": len(names), "teachers": list(teachers.keys()), "temperature": T}
    per_image = []
    maxabs_all, meanabs_all, kl_all = [], [], []
    for n in names:
        ens = acc[n] / float(nt)                             # mean logits
        q_direct = torch.softmax(ens / T, dim=0).cpu().numpy().astype(np.float64)

        cached = np.load(prob_dir / f"{n}.npy")
        q_cache = cached.astype(np.float64)
        if cached.dtype == np.uint8:
            q_cache = q_cache / 255.0
        q_cache = q_cache.transpose(2, 0, 1)                 # [H,W,C] -> [C,H,W]

        diff = np.abs(q_direct - q_cache)
        kl = float(np.sum(q_direct * (np.log(np.clip(q_direct, 1e-12, None))
                                      - np.log(np.clip(q_cache, 1e-12, None)))))
        per_image.append({"name": n, "max_abs": float(diff.max()),
                          "mean_abs": float(diff.mean()), "kl": kl})
        maxabs_all.append(diff.max())
        meanabs_all.append(diff.mean())
        kl_all.append(kl)
        del acc[n]
    torch.cuda.empty_cache()

    # also quantify the "wrong" variant so we can see it would have been caught
    print("\n--- aggregate ---")
    print(f"max |q_direct - q_cache|  : max={max(maxabs_all):.6f}  "
          f"mean_over_images={np.mean(maxabs_all):.6f}")
    print(f"mean|q_direct - q_cache|  : {np.mean(meanabs_all):.8f}")
    print(f"KL(q_direct || q_cache)   : {np.mean(kl_all):.8e}")

    verdict = "PASS"
    if np.mean(maxabs_all) > 1e-2:
        verdict = "FAIL (distribution differs more than fp16 rounding)"
    elif np.mean(kl_all) > 1e-4:
        verdict = "SUSPECT (KL larger than expected for fp16)"
    print(f"\nRESULT: {verdict}")

    summary.update({"per_image": per_image,
                    "max_abs_max": float(max(maxabs_all)),
                    "max_abs_mean": float(np.mean(maxabs_all)),
                    "mean_abs": float(np.mean(meanabs_all)),
                    "kl_mean": float(np.mean(kl_all)),
                    "verdict": verdict})
    out = Path(args.soft_dir) / "DISTRIBUTION_CHECK.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
