#!/usr/bin/env python3
"""Post-hoc diagnostic for the soft-KD student checkpoints.

Answers the question GPT raised: did the student actually absorb the other two
teachers, or did it just stay at its own initialisation (s62)?

For each checkpoint we measure, over the 500 test1 images:

  agree_student   = argmax(student)  == argmax(teacher ensemble)
  agree_init      = argmax(s62 init) == argmax(teacher ensemble)

If agree_student barely moves above agree_init, the KD transferred nothing.
We also report the student's mean max-probability (confidence) so that a
"student got sharper" effect is visible separately from "student moved onto the
ensemble".

Read-only: loads checkpoints, writes one JSON under --outdir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PHYSICAL_GPU_INDEX = 2
SNAP = "/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="KD run dir with checkpoint_step*.pt")
    ap.add_argument("--init-checkpoint", required=True)
    ap.add_argument("--soft-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(PHYSICAL_GPU_INDEX):
        raise RuntimeError("require CUDA_VISIBLE_DEVICES=2")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("require CUDA_DEVICE_ORDER=PCI_BUS_ID")

    import torch
    import torch.nn.functional as F
    for p in (SNAP, "/home/jhy/gpu2_day_queue"):
        if p not in sys.path:
            sys.path.insert(0, p)
    from PIL import Image
    from src.models import build_model, load_model_checkpoint
    from src.transforms import normalize_to_tensor

    prob_dir = Path(args.soft_dir) / "probs"
    names = sorted(p.name for p in prob_dir.glob("*.npy"))[: args.limit]

    # teacher ensemble argmax, from the cache
    ens = []
    for n in names:
        a = np.load(prob_dir / n).astype(np.float32)
        if a.max() > 1.5:  # uint8 cache
            a = a / 255.0
        ens.append(a.argmax(axis=2))
    ens = np.stack(ens)  # [N,H,W]

    ckpts = sorted(Path(args.run_dir).glob("checkpoint_step*.pt"),
                   key=lambda p: int(p.stem.replace("checkpoint_step", "")))
    todo = [(Path(args.init_checkpoint), "init_s62")] + [(c, c.stem) for c in ckpts]

    results = []
    for path, label in todo:
        model = build_model(args.config).to(args.device)
        load_model_checkpoint(model, path, args.device)
        model.eval()
        agree_sum = 0.0
        conf_sum = 0.0
        with torch.inference_mode():
            for i, n in enumerate(names):
                img = np.asarray(Image.open(Path(args.test_dir) / (Path(n).stem + ".png"))
                                 .convert("RGB")).copy()
                px = normalize_to_tensor(img).unsqueeze(0).to(args.device)
                logits = model(pixel_values=px).logits
                logits = F.interpolate(logits, size=ens.shape[-2:], mode="bilinear",
                                       align_corners=False)
                pred = logits.argmax(1)[0].cpu().numpy()
                agree_sum += float((pred == ens[i]).mean())
                conf_sum += float(F.softmax(logits, dim=1).max(1).values.mean())
        n_used = len(names)
        rec = {"label": label, "path": str(path),
               "agree_with_ensemble": agree_sum / n_used,
               "mean_confidence": conf_sum / n_used}
        results.append(rec)
        print(f"{label:22s} agree={rec['agree_with_ensemble']:.4f} "
              f"conf={rec['mean_confidence']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    out = Path(args.run_dir) / "ENSEMBLE_AGREEMENT.json"
    out.write_text(json.dumps({"limit": len(names), "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
