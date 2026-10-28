"""Gate 3: does GradScaler recover on its own after the initial overflow?

Five consecutive autocast + GradScaler steps on the same fixed batch. Nothing is
tuned: no LR change, no loss-weight change, no batch change, no clipping, no
scaler-config change. The scaler's own skip/backoff behaviour is what is under
test, so `scaler.step` is always called and allowed to skip.

Success = two consecutive steps where the optimizer actually ran with all-finite
gradients. If the scale falls substantially and steps still cannot run, the run
stops and reports rather than switching precision strategies on its own.
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys, traceback
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=5)
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
    report = {"gate": "scaler_recovery", "steps_requested": a.steps,
              "status": "ERROR", "seed": 42, "records": [],
              "criterion": "loss_v2.SetCriterionV2(bce_reduction='reference')",
              "notes": "no LR/weight/batch/clip/scaler-config change; scaler.step "
                       "always called and allowed to skip"}
    try:
        encoder = build_encoder("cuda:0").train()
        head = B3M2FStyleHead().cuda().train()
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
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        pixel, label = build_batch(DATA_ROOT, ["0001"], 1024, "cuda:0", 42)
        report["image_sha256"] = hashlib.sha256(pixel.cpu().numpy().tobytes()).hexdigest()
        report["label_sha256"] = hashlib.sha256(label.numpy().tobytes()).hexdigest()

        watched = [("head.mask_embed.2.bias", dict(named)["head.mask_embed.2.bias"])]
        watched.append(("head.class_embed.weight", dict(named)["head.class_embed.weight"]))
        watched.append(("encoder.stages.0.patch_embeddings.proj.weight",
                        dict(named)["encoder.stages.0.patch_embeddings.proj.weight"]))
        focus = dict(named)["head.mask_embed.2.bias"]

        consecutive_ok = 0
        best_consecutive = 0
        for step in range(1, a.steps + 1):
            torch.manual_seed(142 + step); torch.cuda.manual_seed_all(142 + step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                out = head(encoder(pixel_values=pixel,
                                   output_hidden_states=True).hidden_states)
            aux = tuple({"class_logits": x["class_logits"].float(),
                         "mask_logits": x["mask_logits"].float()}
                        for x in out["aux_outputs"])
            targets, pts, w, vm = V2.build_set_targets(label, 8)
            targets, pts, w, vm = to_device_set_targets(targets, pts, w, vm, "cuda:0")
            losses = criterion(out["class_logits"].float(), out["mask_logits"].float(),
                               targets, pts, w, aux, vm)
            total = criterion.total(losses)

            scale_before = scaler.get_scale()
            fwd_finite = bool(torch.isfinite(total))
            losses_finite = all(bool(torch.isfinite(v)) for v in losses.values())
            before = [(n, p.detach().clone()) for n, p in watched]

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nonfinite = [n for n, p in named
                         if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            fg = focus.grad
            previous = len(steps_seen)
            scaler.step(optimizer)
            scaler.update()
            ran = len(steps_seen) > previous

            delta = [{"name": n, "shape": list(p.shape),
                      "max_delta": float((p.detach() - old).abs().max()),
                      "changed": bool(float((p.detach() - old).abs().max()) > 0.0)}
                     for (n, p), (_, old) in zip(watched, before)]
            rec = {"step": step,
                   "scale_before": scale_before, "scale_after": scaler.get_scale(),
                   "forward_finite": fwd_finite, "losses_finite": losses_finite,
                   "total": float(total.detach()),
                   "nonfinite_gradient_count": len(nonfinite),
                   "optimizer_step_ran": ran,
                   "focus_grad": None if fg is None else {
                       "finite": bool(torch.isfinite(fg).all()),
                       "inf": int(torch.isinf(fg).sum()),
                       "nan": int(torch.isnan(fg).sum()),
                       "finite_max_abs": float(fg[torch.isfinite(fg)].abs().max())
                       if bool(torch.isfinite(fg).any()) else None},
                   "parameter_delta": delta}
            report["records"].append(rec)
            print(json.dumps(rec), flush=True)

            if ran and len(nonfinite) == 0:
                consecutive_ok += 1
                best_consecutive = max(best_consecutive, consecutive_ok)
            else:
                consecutive_ok = 0
            del out, aux, losses, total, before, delta

        report["best_consecutive_clean_steps"] = best_consecutive
        report["final_scale"] = scaler.get_scale()
        report["recovered"] = best_consecutive >= 2
        report["status"] = "COMPLETE"
    except Exception:
        report["error"] = traceback.format_exc()
    (root / "RESULT.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
