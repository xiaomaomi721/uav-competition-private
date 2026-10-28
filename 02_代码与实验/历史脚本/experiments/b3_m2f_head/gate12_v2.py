"""Gate 1/2: numerical scale and precision behaviour of criterion v2.

Read-only with respect to every frozen file: it imports loss_v2 and the frozen
head/train helpers, and never touches loss.py, head.py or train_entry.py.

Gate 1 (--precision fp32) answers: does mask BCE still scale with the point
count? Reported alongside the per-mask BCE BEFORE the /num_masks division and
the effective valid mass, so the value can be checked against the weighted
point-mean statistics rather than against a hard-coded expectation.

Gate 2 runs the same fixed graph at fp32 / autocast / scaled and requires, for
every mode: finite forward and losses, all finite parameter gradients, an
optimizer step that actually ran, and encoder/head parameters that actually
moved. GradScaler scale before/after and whether the step was skipped are
recorded.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, traceback
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=("fp32", "autocast", "scaled"), required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or \
       os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("GPU2 mapping required")
    root = Path(a.outdir)
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    from b3_m2f_head import loss_v2 as V2
    from b3_m2f_head.head import B3M2FStyleHead
    from b3_m2f_head.train_entry import (build_encoder, build_batch,
                                         to_device_set_targets, build_weight_dict,
                                         DATA_ROOT)

    torch.manual_seed(42); np.random.seed(42); torch.backends.cudnn.benchmark = False
    report = {"precision": a.precision, "status": "ERROR", "seed": 42,
              "criterion": "loss_v2.SetCriterionV2(bce_reduction='reference')"}
    try:
        encoder = build_encoder("cuda:0").train()
        head = B3M2FStyleHead().cuda().train()
        params = list(encoder.parameters()) + list(head.parameters())
        named = ([("encoder." + n, p) for n, p in encoder.named_parameters()] +
                 [("head." + n, p) for n, p in head.named_parameters()])
        h = hashlib.sha256()
        for n, p in named:
            h.update(n.encode()); h.update(p.detach().cpu().numpy().tobytes())
        report["init_sha256"] = h.hexdigest()

        criterion = V2.SetCriterionV2(num_classes=8,
                                      weight_dict=build_weight_dict(6, 1.0),
                                      bce_reduction="reference").cuda()
        optimizer = torch.optim.AdamW(
            [{"params": list(encoder.parameters()), "lr": 1e-5},
             {"params": list(head.parameters()), "lr": 1e-3}], weight_decay=0.01)
        steps_seen = []
        optimizer.register_step_post_hook(lambda *args, **kw: steps_seen.append(1))
        scaler = torch.amp.GradScaler("cuda", enabled=a.precision == "scaled")

        pixel, label = build_batch(DATA_ROOT, ["0001"], 1024, "cuda:0", 42)
        report["image_sha256"] = hashlib.sha256(pixel.cpu().numpy().tobytes()).hexdigest()
        report["label_sha256"] = hashlib.sha256(label.numpy().tobytes()).hexdigest()

        torch.manual_seed(142); torch.cuda.manual_seed_all(142)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=a.precision != "fp32"):
            out = head(encoder(pixel_values=pixel, output_hidden_states=True).hidden_states)
        aux = tuple({"class_logits": x["class_logits"].float(),
                     "mask_logits": x["mask_logits"].float()}
                    for x in out["aux_outputs"])
        targets, pts, w, vm = V2.build_set_targets(label, 8)
        targets, pts, w, vm = to_device_set_targets(targets, pts, w, vm, "cuda:0")
        report["P"] = int(pts.shape[1])
        report["num_masks"] = criterion._num_masks(targets)
        report["valid_mass_fraction"] = float(w.mean())

        # ---- scale probe: per-mask BCE before the /num_masks division --------
        # Reported in BOTH reductions so the value can be checked against the
        # effective valid-point statistics rather than against a fixed constant:
        #   reference : mean over all P points, weights ignored (== installed)
        #   points    : sum(bce*w)/sum(w), the ignore-aware weighted point mean
        probe = {}
        with torch.no_grad():
            rows = []
            for i, t in enumerate(targets):
                if t["class_labels"].numel() == 0:
                    continue
                mlog = out["mask_logits"][i].float()
                idx = criterion.matcher(out["class_logits"].float()[i:i+1],
                                        mlog[None], [t], pts[i:i+1], w[i:i+1],
                                        vm[i:i+1])[0]
                src, tgt = idx
                if src.numel() == 0:
                    continue
                n_pairs = src.numel()
                ppt = pts[i:i+1].expand(n_pairs, -1, -1)          # (M,P,2)
                s_out = V2.sample_point(mlog[src][:, None], ppt)   # (M,1,P)
                s_tgt, s_w = V2.sample_target_masks(
                    t["mask_labels"][tgt][:, None],
                    vm[i:i+1].expand(n_pairs, -1, -1), ppt)
                ref = V2.SetCriterionV2.mask_bce_per_mask(s_out, s_tgt, None, "reference")
                awd = V2.SetCriterionV2.mask_bce_per_mask(s_out, s_tgt, s_w, "points")
                for j in range(n_pairs):
                    rows.append({"image": i, "pair": j,
                                 "per_mask_bce_reference": float(ref[j]),
                                 "per_mask_bce_points": float(awd[j]),
                                 "valid_mass": float(s_w[j].sum()),
                                 "valid_mass_fraction": float(s_w[j].mean())})
            if rows:
                probe = {
                    "rows": rows,
                    "per_mask_bce_reference_mean": float(np.mean(
                        [r["per_mask_bce_reference"] for r in rows])),
                    "per_mask_bce_points_mean": float(np.mean(
                        [r["per_mask_bce_points"] for r in rows])),
                    "valid_mass_fraction_mean": float(np.mean(
                        [r["valid_mass_fraction"] for r in rows])),
                    "num_pairs": len(rows),
                }
        report["per_mask_bce_probe"] = probe

        losses = criterion(out["class_logits"].float(), out["mask_logits"].float(),
                           targets, pts, w, aux, vm)
        total = criterion.total(losses)
        report["losses"] = {k: float(v.detach()) for k, v in losses.items()}
        report["total"] = float(total.detach())
        report["forward_finite"] = bool(torch.isfinite(total))
        report["losses_finite"] = all(bool(torch.isfinite(v)) for v in losses.values())
        if not report["forward_finite"] or not report["losses_finite"]:
            raise RuntimeError("non-finite forward/loss")

        # Pair BEFORE/AFTER positionally, not by name: keying a dict on the
        # parameter name collides when two entries share a suffix, which then
        # subtracts tensors of different shapes. Shape is recorded so a genuine
        # shape change would be visible rather than crashing.
        watched = [(n, p) for n, p in named
                   if n.endswith("class_embed.weight")
                   or n.endswith("mask_embed.2.weight")
                   or n == "encoder.stages.0.patch_embeddings.proj.weight"]
        before = [(n, p.detach().clone()) for n, p in watched]
        scale_before = scaler.get_scale()
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nonfinite = [n for n, p in named
                     if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        missing = [n for n, p in named if p.grad is None]
        grad_norm = None
        if not nonfinite:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for _, p in named], 1e9))
        report["grad_norm_unclipped"] = grad_norm
        report["nonfinite_gradients"] = nonfinite
        report["missing_gradients"] = missing
        focus = dict(named)["head.mask_embed.2.bias"].grad
        report["focus_grad"] = None if focus is None else {
            "finite": bool(torch.isfinite(focus).all()),
            "inf": int(torch.isinf(focus).sum()),
            "nan": int(torch.isnan(focus).sum()),
            "finite_max_abs": float(focus[torch.isfinite(focus)].abs().max())
            if bool(torch.isfinite(focus).any()) else None}

        previous = len(steps_seen)
        if not nonfinite:
            scaler.step(optimizer)
        scaler.update()
        report["scale_before"] = scale_before
        report["scale_after"] = scaler.get_scale()
        report["optimizer_called"] = len(steps_seen) > previous
        report["parameter_max_delta"] = [
            {"name": n, "shape": list(p.shape),
             "max_delta": float((p.detach() - old).abs().max()),
             "changed": bool(float((p.detach() - old).abs().max()) > 0.0)}
            for (n, p), (_, old) in zip(watched, before)]
        report["peak_allocated_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        report["peak_reserved_MiB"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        report["source_hashes"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py") if p.name != "__init__.py"}
        report["status"] = "COMPLETE"
    except Exception:
        report["error"] = traceback.format_exc()
    (root / "RESULT.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
