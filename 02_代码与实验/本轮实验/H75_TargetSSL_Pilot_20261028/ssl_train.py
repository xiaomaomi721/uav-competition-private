"""H75 TargetSSL pilot: C0 (source-only) / C1 (RGB-MIM) / C2 (feature consistency).

Registered rules:
  - trainable: segformer.stages.0/1 only (+ training-only aux head for C1)
  - stage2/stage3/decode_head frozen (params AND buffers audited)
  - target images used ONLY for image reconstruction / feature regression.
    No target labels, no pseudo-labels, no teacher logits.
  - lambda chosen ONCE from a pilot batch so ||grad_aux|| / ||grad_src|| ~ 0.2-0.3
"""
import argparse
from ssl_common import *

PATCH = 32
MASK_RATIO = 0.4
BASE_LR = 1e-5
HEAD_LR = 5e-7
WD = 0.01


def make_masks(seed):
    gh = gw = 1024 // PATCH
    n = gh * gw
    k = max(1, int(round(MASK_RATIO * n)))
    g = np.random.default_rng(seed)
    idx = g.permutation(n)[:k]
    grid = np.zeros(n, dtype=bool); grid[idx] = True
    grid = grid.reshape(gh, gw)
    px = np.kron(grid, np.ones((PATCH, PATCH), dtype=bool))
    return (torch.from_numpy(px)[None, None].float().cuda(),
            torch.from_numpy(grid.astype(np.float32)).cuda())


def token_mask_from_grid(grid):
    """grid: [gh,gw] patch-level mask. stage1 is 128x128; 1024/128=8 px per token,
    32-px patch => 4x4 tokens."""
    return F.interpolate(grid[None, None], size=(128, 128), mode="nearest")[0, 0]


def build_pairs(n):
    src_ids = [x.strip() for x in (SRC / "splits" / "train.txt").read_text().split() if x.strip()]
    assert len(src_ids) == 5597
    rows = target_rows()
    assert len(rows) == 3594
    sr = np.random.default_rng(SEED + 10001).permutation(len(src_ids))[:n]
    tr = np.random.default_rng(SEED + 10002).permutation(len(rows))[:n]
    return [(src_ids[int(s)], rows[int(t)]) for s, t in zip(sr, tr)]


def make_student():
    m = build_model()
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith(STAGE01))
    return m.cuda()


def aux_forward(model, arm, x_masked, teacher=None):
    """Returns (aux_loss, aux_log). Shared stage0/1 forward."""
    hs, logits = stage_features(model, x_masked)
    f1 = hs[1]  # 128x128x128
    if arm == "C1":
        return f1, logits
    return f1, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["C0", "C1", "C2"], required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--pilot", action="store_true", help="only measure gradient norms")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    guard(); seed_all()
    steps = a.smoke if a.smoke else a.steps
    out = Path(a.out or (WORK / ("pilot" if a.pilot else f"train_{a.arm}")))
    out.mkdir(parents=True, exist_ok=False)

    from src.label_mapping import official_to_train
    from src.transforms import TrainTransform
    from src.rare_crop import RareClassCropper

    student = make_student()
    teacher = None
    if a.arm == "C2":
        teacher = build_model().cuda().eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    head = RGBHead().cuda() if a.arm == "C1" else None
    params = [p for n, p in student.named_parameters() if n.startswith(STAGE01)]
    groups = [{"params": params, "lr": BASE_LR}]
    if head is not None:
        groups.append({"params": list(head.parameters()), "lr": HEAD_LR})
    opt = torch.optim.AdamW(groups, weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    tf = TrainTransform(crop_size=1024, config=get_cfg()["transforms"],
                        rare_cropper=RareClassCropper(1024, .5, (7, 4, 6), .25),
                        ignore_index=255)
    pairs = build_pairs(steps)
    snap = frozen_snapshot(student)

    meta = {"arm": a.arm, "steps": steps, "smoke": bool(a.smoke), "pilot": bool(a.pilot),
            "h75_sha256": H75_SHA, "trainable_scope": list(STAGE01),
            "frozen_scope": list(FROZEN), "mask_ratio": MASK_RATIO, "patch": PATCH,
            "base_lr": BASE_LR, "head_lr": HEAD_LR, "wd": WD, "seed": SEED,
            "lambda": a.lam, "gpu_index": EXPECTED_GPU_INDEX,
            "trainable_params": int(sum(p.numel() for p in student.parameters() if p.requires_grad)),
            "aux_head_params": int(sum(p.numel() for p in head.parameters())) if head else 0}
    dump(out / "RUN_CONFIG.json", meta)
    emit(stage="start", **{k: meta[k] for k in ("arm", "steps", "trainable_params", "lambda")})

    b = torch.tensor(BSTAR).cuda()[None, :, None, None]
    hist = open(out / "history.jsonl", "w", buffering=1)
    t0 = time.time()
    pilot_src, pilot_aux = [], []
    try:
        for i, (stem, row) in enumerate(pairs, 1):
            if i % 20 == 1:
                guard()
            rgb = read_rgb(SOURCE_DATA / "train" / "images" / f"{stem}.png")
            with Image.open(SOURCE_DATA / "train" / "masks" / f"{stem}.png") as f:
                lab = official_to_train(np.asarray(f).copy())
            sx, sy = tf(rgb, lab, rng=random.Random(SEED + 100000 + i))
            sx = sx.unsqueeze(0).cuda(); sy = sy.unsqueeze(0).long().cuda()

            tgt_rgb = read_rgb(TARGET_IMAGES / row["input_name"])
            tx = normalize_to_tensor(tgt_rgb).unsqueeze(0).cuda()
            tgt01 = torch.from_numpy(tgt_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
            px_mask, grid = make_masks(SEED + 500000 + i)
            tx_masked = tx * (1.0 - px_mask)

            # ---- source CE ----
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                sz = student(pixel_values=sx).logits
            sz = F.interpolate(sz.float(), size=sy.shape[-2:], mode="bilinear", align_corners=False)
            l_src = F.cross_entropy(sz, sy, ignore_index=255)
            g_src = torch.autograd.grad(l_src, params, retain_graph=False, allow_unused=True)
            n_src = float(torch.sqrt(sum((g.float() ** 2).sum() for g in g_src if g is not None)))

            # ---- aux ----
            l_aux = None
            n_aux = 0.0
            if a.arm != "C0":
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    hs, _ = stage_features(student, tx_masked)
                f1_masked = hs[1].float()
                if a.arm == "C1":
                    rec = head(f1_masked)
                    m3 = px_mask.expand(-1, 3, -1, -1)
                    l_aux = ((rec - tgt01).abs() * m3).sum() / (3.0 * m3.sum().clamp_min(1.0))
                else:
                    with torch.no_grad():
                        with torch.amp.autocast("cuda"):
                            hs_ref, _ = stage_features(teacher, tx)
                        f1_ref = hs_ref[1].float().detach()
                    tm = token_mask_from_grid(grid).unsqueeze(0).unsqueeze(0)
                    a_n = F.normalize(f1_masked, dim=1)
                    b_n = F.normalize(f1_ref, dim=1)
                    cos = (a_n * b_n).sum(1, keepdim=True)
                    l_aux = ((1.0 - cos) * tm).sum() / tm.sum().clamp_min(1.0)
                l_aux = l_aux.float()
                aux_params = params + (list(head.parameters()) if head else [])
                g_aux = torch.autograd.grad(l_aux, aux_params, retain_graph=False, allow_unused=True)
                n_aux = float(torch.sqrt(sum((g.float() ** 2).sum() for g in g_aux if g is not None)))
                pilot_src.append(n_src); pilot_aux.append(n_aux)

            if a.pilot:
                if i >= 10:
                    break
                continue

            # ---- combined step ----
            opt.zero_grad(set_to_none=True)
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = (BASE_LR if gi == 0 else HEAD_LR) * max(0.0, 1.0 - (i - 1) / 10000)
            with torch.amp.autocast("cuda"):
                sz2 = student(pixel_values=sx).logits
            sz2 = F.interpolate(sz2.float(), size=sy.shape[-2:], mode="bilinear", align_corners=False)
            l_src2 = F.cross_entropy(sz2, sy, ignore_index=255)
            total = l_src2
            if a.arm != "C0" and a.lam:
                with torch.amp.autocast("cuda"):
                    hs2, _ = stage_features(student, tx_masked)
                f1m = hs2[1].float()
                if a.arm == "C1":
                    rec = head(f1m)
                    m3 = px_mask.expand(-1, 3, -1, -1)
                    l_aux2 = ((rec - tgt01).abs() * m3).sum() / (3.0 * m3.sum().clamp_min(1.0))
                else:
                    with torch.no_grad():
                        with torch.amp.autocast("cuda"):
                            hsr, _ = stage_features(teacher, tx)
                        f1r = hsr[1].float().detach()
                    tm = token_mask_from_grid(grid).unsqueeze(0).unsqueeze(0)
                    cos = (F.normalize(f1m, dim=1) * F.normalize(f1r, dim=1)).sum(1, keepdim=True)
                    l_aux2 = ((1.0 - cos) * tm).sum() / tm.sum().clamp_min(1.0)
                warm = min(1.0, i / 50.0)
                total = total + a.lam * warm * l_aux2.float()

            scaler.scale(total).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(params + (list(head.parameters()) if head else []), 1.0))
            sb = scaler.get_scale()
            scaler.step(opt); scaler.update()
            stepped = scaler.get_scale() >= sb

            row_out = {"step": i, "arm": a.arm, "loss_src": round(float(l_src2), 6),
                       "loss_aux": round(float(l_aux2), 6) if (a.arm != "C0" and a.lam) else None,
                       "loss_total": round(float(total), 6), "grad_norm": round(gn, 5),
                       "stepped": stepped, "elapsed": round(time.time() - t0, 1),
                       "mem_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)}
            hist.write(json.dumps(row_out) + "\n")
            if i % 25 == 0 or i == steps:
                emit(stage="train", **row_out)
            if i in (100, 300) or i == steps:
                torch.save({"model": student.state_dict(), "step": i, "arm": a.arm,
                            "h75_sha256": H75_SHA, "lambda": a.lam,
                            "aux_head": head.state_dict() if head else None},
                           out / f"checkpoint_step{i}.pt")

        if a.pilot:
            ratio = (float(np.mean(pilot_aux)) / max(1e-9, float(np.mean(pilot_src))))
            res = {"mean_grad_src": float(np.mean(pilot_src)),
                   "mean_grad_aux": float(np.mean(pilot_aux)),
                   "ratio_aux_over_src": ratio,
                   "n": len(pilot_src), "arm": a.arm,
                   "lambda_for_0.25": 0.25 / max(1e-9, ratio)}
            dump(out / "PILOT_GRAD.json", res)
            print(json.dumps(res, indent=2))
            return

        bad = frozen_diff(student, snap)
        summary = {**meta, "status": "COMPLETED", "completed_steps": steps,
                   "runtime_sec": round(time.time() - t0, 1),
                   "frozen_diff_after_training": bad,
                   "peak_mem_mib": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)}
        dump(out / "SUMMARY.json", summary)
        emit(stage="complete", arm=a.arm, frozen_diff=len(bad))
    finally:
        hist.close()


if __name__ == "__main__":
    main()
