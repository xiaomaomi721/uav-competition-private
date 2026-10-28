"""Spatial-structure causal test against an already-trained checkpoint.

Inference only: loads a finished tiny-overfit checkpoint, reconstructs the
encoder+head, and evaluates the fixed image normally and under three spatial
permutations of the mask logits.

What is held constant : class logits, every query mask's exact value multiset,
                        the image, the metric.
What is destroyed     : only WHERE each mask value sits, via ONE shared random
                        permutation applied to the H*W axis of every query.

If mIoU stays high under permutation, the score never depended on mask spatial
structure and the tiny-overfit pass is an artefact. If it collapses, the high
score is carried by mask layout.

The permutations are random, so the CHECKPOINT is also evaluated once with no
permutation in the same process, on the same weights, as the reference point.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, traceback
from pathlib import Path

import torch
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])
    ap.add_argument("--crop", type=int, default=1024)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or \
       os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("GPU2 mapping required")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.semantic import semantic_scores
    from b3_m2f_head.train_entry import (build_encoder, build_batch, forward_logits,
                                         mean_iou_from_counts, precision_context,
                                         DATA_ROOT, IGNORE)

    report = {"checkpoint": a.checkpoint, "status": "ERROR", "crop": a.crop,
              "seeds": list(a.seeds)}
    try:
        ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        report["checkpoint_sha256"] = hashlib.sha256(
            Path(a.checkpoint).read_bytes()).hexdigest()
        report["checkpoint_step"] = ckpt.get("step")
        report["ckpt_config"] = {k: v for k, v in (ckpt.get("config") or {}).items()
                                 if k in ("stage", "batch_size", "crop", "amp",
                                          "steps", "num_queries", "hidden_dim",
                                          "layers", "fixed_subset")}

        encoder = build_encoder("cuda:0")
        head = B3M2FStyleHead().cuda()
        cfg = ckpt["config"]
        head.load_state_dict(ckpt["head"], strict=True)
        encoder.load_state_dict(ckpt["model"], strict=True)
        encoder.eval(); head.eval()
        torch.set_grad_enabled(False)

        # Prove the reload is faithful: the ratio probes do not need the model at
        # all, but a wrong load would silently invalidate the comparison.
        report["head_tensors"] = len(ckpt["head"])
        report["encoder_tensors"] = len(ckpt["model"])

        eval_stems = None
        for candidate in (ckpt["config"].get("eval_stems"), None):
            if candidate:
                eval_stems = candidate
        if not eval_stems:
            splits = Path("/home/jhy/deploy_exp18/"
                          "exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/splits/train.txt")
            stems = [l.strip() for l in splits.read_text().split() if l.strip()]
            eval_stems = stems[:int(cfg.get("batch_size", 1))]
        report["evaluation_images"] = len(eval_stems)
        assert len(eval_stems) == 1, f"expected a one-image overfit, got {eval_stems}"

        px, lb = build_batch(DATA_ROOT, eval_stems, a.crop, "cuda:0", 42)
        lb = lb.to("cuda:0")
        report["image_sha256"] = hashlib.sha256(px.cpu().numpy().tobytes()).hexdigest()

        def confusion(permute_seed=None):
            with precision_context(True):
                o = forward_logits(encoder, px, head)
            ml = o["mask_logits"]
            if permute_seed is not None:
                flat = ml.flatten(2)
                gen = torch.Generator(device=flat.device); gen.manual_seed(int(permute_seed))
                order = torch.randperm(flat.shape[-1], generator=gen, device=flat.device)
                ml = flat[..., order].view_as(ml)
            scores = semantic_scores(o["class_logits"].float(), ml.float(),
                                     upsample_to=lb.shape[-2:])
            pred = scores.argmax(1) + 1
            gt = torch.where(lb == IGNORE, torch.zeros_like(lb), lb + 1)
            idx = gt.reshape(-1).long() * 9 + pred.reshape(-1).long()
            return torch.bincount(idx, minlength=81).reshape(9, 9).double()

        cm = confusion(None)
        miou, per_class = mean_iou_from_counts(cm)
        report["original"] = {"miou": miou, "per_class_iou": per_class,
                              "pixel_accuracy": float((torch.diagonal(cm).sum() / cm.sum()).item()),
                              "confusion": cm.long().tolist()}
        print(f"[original] mIoU={miou:.4f}", flush=True)

        report["permuted"] = []
        for seed in a.seeds:
            cm_p = confusion(seed)
            m, pc = mean_iou_from_counts(cm_p)
            report["permuted"].append({
                "seed": int(seed), "miou": m, "per_class_iou": pc,
                "pixel_accuracy": float((torch.diagonal(cm_p).sum() / cm_p.sum()).item()),
                "confusion": cm_p.long().tolist()})
            print(f"[permuted seed {seed}] mIoU={m:.4f}", flush=True)

        best = max(t["miou"] for t in report["permuted"])
        report["drop"] = report["original"]["miou"] - best
        report["verdict"] = ("COLLAPSES_UNDER_PERMUTATION" if report["drop"] > 0.5
                             else "SURVIVES_PERMUTATION")
        report["status"] = "COMPLETE"
    except Exception:
        report["error"] = traceback.format_exc()
    (outdir / "RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                        encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("original", "permuted", "error")}))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
