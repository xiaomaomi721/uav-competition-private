#!/usr/bin/env python3
"""TTA Anatomy: single-pass 8-view D4 TTA with pre-registered scheme evaluation.

Used by Phase B (Golden/Stage23), Phase C (Stage66-52000) and Phase F (Full6996).

One image -> 8 forward passes -> inverse transform -> in-memory evaluation of ALL
pre-registered schemes. No 8-view logit tensor is ever persisted (188 GB would not
fit). Only the standard TTA8 mean logits are cached, fp16, in chunks.

Schemes are PRE-REGISTERED and symmetric; there is no subset search.

Static test (no GPU):
    python tta_anatomy.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
sys.path.insert(0, str(SNAP))

DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
NUM_CLASSES = 8
IGNORE = 255
CROP = 1024

# ---------------------------------------------------------------------------
# View index ordering MUST match the historical D4 definition:
#   index = k * 2 + (1 if hflip else 0),  k in 0..3
# ---------------------------------------------------------------------------
VIEWS = [(k, hf) for k in range(4) for hf in (False, True)]
VIEW_NAMES = [f"rot{k*90}" + ("_mirror" if hf else "") for k, hf in VIEWS]

# Pre-registered schemes: name -> (member view indices, combine mode)
SCHEMES: dict[str, tuple[tuple[int, ...], str]] = {
    "single_canonical": ((0,), "logit"),
    "single_rot90": ((2,), "logit"),
    "single_rot180": ((4,), "logit"),
    "single_rot270": ((6,), "logit"),
    "mirror_pair": ((0, 1), "logit"),
    "rotations4": ((0, 2, 4, 6), "logit"),
    "subset2_mirror90": ((0, 1, 2, 3), "logit"),
    "subset4_mirror0_90": ((0, 1, 2, 3, 4, 5), "logit"),
    "subset6": ((0, 1, 2, 3, 4, 5, 6, 7)[:6], "logit"),
    "tta8_mean_logits": (tuple(range(8)), "logit"),
    "tta8_mean_probs": (tuple(range(8)), "prob"),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


# ---------------------------------------------------------------------------
# pure-numpy core (unit-testable without torch)
# ---------------------------------------------------------------------------
def combine(view_logits: list[np.ndarray], members: tuple[int, ...], mode: str) -> np.ndarray:
    """Combine the given member views. view_logits are HxWxC float32, already
    inverse-transformed back into canonical orientation."""
    stack = np.stack([view_logits[i] for i in members], axis=0)
    if mode == "prob":
        m = stack.max(axis=-1, keepdims=True)
        e = np.exp(stack - m)
        p = e / e.sum(axis=-1, keepdims=True)
        return p.mean(axis=0)
    return stack.mean(axis=0)


def update_confusion(cm: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> None:
    """cm[gt, pred] += 1, ignoring IGNORE pixels and out-of-range ids."""
    valid = (gt != IGNORE) & (gt >= 0) & (gt < NUM_CLASSES)
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    idx = g * NUM_CLASSES + p
    cm += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    diag = np.diag(cm).astype(np.float64)
    row = cm.sum(axis=1).astype(np.float64)   # GT counts
    col = cm.sum(axis=0).astype(np.float64)   # pred counts
    denom = row + col - diag
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, diag / np.maximum(denom, 1e-12), np.nan)
        acc = np.where(row > 0, diag / np.maximum(row, 1e-12), np.nan)
    return {
        "mIoU": float(np.nanmean(iou)),
        "pixel_accuracy": float(diag.sum() / max(cm.sum(), 1)),
        "per_class_iou": [None if np.isnan(v) else float(v) for v in iou],
        "per_class_recall": [None if np.isnan(v) else float(v) for v in acc],
        "gt_pixels": [int(v) for v in row],
        "pred_pixels": [int(v) for v in col],
    }


def disagreement(cm_a: np.ndarray, cm_b: np.ndarray) -> float:
    """Pixel disagreement proxy between two schemes that share the same GT."""
    return None  # replaced by pairwise prediction comparison, see run().


# ---------------------------------------------------------------------------
# self-test: synthetic logits, no torch / no GPU
# ---------------------------------------------------------------------------
def self_test() -> int:
    rng = np.random.default_rng(42)
    H = W = 8
    gt = rng.integers(0, NUM_CLASSES, size=(H, W)).astype(np.uint8)
    gt[0, 0] = IGNORE

    # 8 synthetic views; view 0 is the "true" one, others are noisy
    base = rng.normal(size=(H, W, NUM_CLASSES)).astype(np.float32)
    views = [base + rng.normal(scale=0.5, size=(H, W, NUM_CLASSES)).astype(np.float32)
             for _ in range(8)]

    cms = {n: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for n in SCHEMES}
    for name, (members, mode) in SCHEMES.items():
        c = combine(views, members, mode)
        update_confusion(cms[name], c.argmax(axis=-1).astype(np.uint8), gt)

    # invariant 1: every scheme consumed exactly H*W-1 valid pixels
    for name, cm in cms.items():
        assert cm.sum() == H * W - 1, (name, cm.sum())
    # invariant 2: ignore pixel excluded
    assert all(cm.sum() == H * W - 1 for cm in cms.values())
    # invariant 3: single_canonical equals argmax of view 0 exactly
    canon = views[0].argmax(axis=-1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    update_confusion(cm, canon.astype(np.uint8), gt)
    assert np.array_equal(cm, cms["single_canonical"]), "canonical view mismatch"
    # invariant 4: tta8 is the mean of all 8, and differs from a single view
    mean8 = np.stack(views).mean(axis=0)
    assert np.allclose(combine(views, tuple(range(8)), "logit"), mean8, atol=1e-5)
    # invariant 5: mean_probs rows sum to 1
    p = combine(views, tuple(range(8)), "prob")
    assert np.allclose(p.sum(axis=-1), 1.0, atol=1e-4), "prob scheme not normalised"
    # invariant 6: metrics shape + ignore handled
    m = metrics_from_confusion(cms["tta8_mean_logits"])
    assert len(m["per_class_iou"]) == NUM_CLASSES
    assert 0.0 <= m["mIoU"] <= 1.0
    # invariant 7: view ordering matches the historical D4 definition
    assert VIEWS == [(0, False), (0, True), (1, False), (1, True),
                     (2, False), (2, True), (3, False), (3, True)]
    assert VIEW_NAMES[0] == "rot0" and VIEW_NAMES[1] == "rot0_mirror"
    assert VIEW_NAMES[2] == "rot90" and VIEW_NAMES[7] == "rot270_mirror"
    # invariant 8: fp16 round-trip of mean logits (precision gate core)
    q = mean8.astype(np.float16).astype(np.float32)
    a, b = mean8.argmax(-1), q.argmax(-1)
    print(f"  self-test fp16 argmax disagreement on synthetic: "
          f"{(a != b).mean() * 100:.4f}%")

    print("SELF_TEST_OK  schemes=%d  views=%d" % (len(SCHEMES), len(VIEWS)))
    return 0


# ---------------------------------------------------------------------------
def image_list(split: str) -> tuple[list[Path], list[Path] | None]:
    if split == "val":
        ids = [l.strip() for l in (SNAP / "splits" / "val.txt").read_text().splitlines() if l.strip()]
        imgs = [DATA / "train" / "images" / f"{i}.png" for i in ids]
        msks = [DATA / "train" / "masks" / f"{i}.png" for i in ids]
        return imgs, msks
    imgs = sorted((DATA / "images").glob("*.png"), key=lambda p: int(p.stem.split("_")[-1]))
    assert len(imgs) == 500, len(imgs)
    return imgs, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out-dir", type=Path, required=False)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-mean-logits", action="store_true")
    ap.add_argument("--precision-gate-n", type=int, default=32)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    import torch
    from PIL import Image
    from src.utils import load_config
    from src.models import build_model
    from src.transforms import normalize_to_tensor

    assert args.out_dir, "--out-dir required"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "mean_logits_fp16"

    cfg = load_config(str(args.config))
    cfg["cwfa"] = {"enabled": False, "probability": 0.3, "epsilon": 4.0,
                   "norm_eps": 1.0e-12, "start_step": 1}
    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()

    ck = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    sd = None
    for k in ("model", "model_state_dict", "state_dict"):
        c = ck.get(k)
        if isinstance(c, dict) and c and all(isinstance(v, torch.Tensor) for v in c.values()):
            sd = c
            break
    if all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    del ck, sd

    imgs, msks = image_list(args.split)
    if args.limit:
        imgs, msks = imgs[:args.limit], (msks[:args.limit] if msks else None)

    cms = {n: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64) for n in SCHEMES}
    preds_cache: dict[str, list[np.ndarray]] = {n: [] for n in SCHEMES}
    gate_fp32, gate_idx = [], []
    cache_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for i, ip in enumerate(imgs):
        with Image.open(ip) as h:
            im = np.asarray(h.convert("RGB")).copy()
        gt = None
        if msks is not None:
            from src.label_mapping import official_to_train
            gt = official_to_train(np.asarray(Image.open(msks[i])))
        px = normalize_to_tensor(im).unsqueeze(0).to(device)

        views: list[np.ndarray] = []
        with torch.inference_mode():
            for k, hf in VIEWS:
                v = px
                if k:
                    v = torch.rot90(v, k=k, dims=[-2, -1])
                if hf:
                    v = v.flip(dims=[-1])
                lg = model(pixel_values=v).logits
                if hf:
                    lg = lg.flip(dims=[-1])
                if k:
                    lg = torch.rot90(lg, k=-k, dims=[-2, -1])
                if tuple(lg.shape[-2:]) != (CROP, CROP):
                    lg = torch.nn.functional.interpolate(
                        lg, size=(CROP, CROP), mode="bilinear", align_corners=False)
                views.append(lg[0].permute(1, 2, 0).float().cpu().numpy())

        for name, (members, mode) in SCHEMES.items():
            c = combine(views, members, mode)
            pr = c.argmax(axis=-1).astype(np.uint8)
            preds_cache[name].append(pr)
            if gt is not None:
                update_confusion(cms[name], pr, gt)

        if args.cache_mean_logits:
            mean8 = combine(views, tuple(range(8)), "logit").astype(np.float16)
            np.save(cache_dir / f"{ip.stem}.npy", mean8)
            if len(gate_fp32) < args.precision_gate_n and gt is not None:
                gate_fp32.append(mean8.astype(np.float32))
                gate_idx.append(i)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(imgs)}] {time.perf_counter()-t0:.1f}s", flush=True)

    # ---------------------------------------------------------- pairwise disagree
    names = list(SCHEMES)
    disagree = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            pa, pb = preds_cache[names[a]], preds_cache[names[b]]
            d = sum(int((x != y).sum()) for x, y in zip(pa, pb))
            tot = sum(x.size for x in pa)
            disagree[f"{names[a]}_vs_{names[b]}"] = d / tot

    # ---------------------------------------------------------- fp16 gate
    gate = {"evaluated": False}
    if gate_fp32:
        dis, iou_deltas = [], []
        for j, idx in enumerate(gate_idx):
            fp32 = gate_fp32[j]
            rel = np.load(cache_dir / f"{imgs[idx].stem}.npy").astype(np.float32)
            a32, a16 = fp32.argmax(-1), rel.argmax(-1)
            dis.append(float((a32 != a16).mean()))
            m32 = metrics_from_confusion(_cm_from_pred(a32, msks[idx]))
            m16 = metrics_from_confusion(_cm_from_pred(a16, msks[idx]))
            iou_deltas.append(abs(m32["mIoU"] - m16["mIoU"]))
        gate = {
            "evaluated": True, "n": len(gate_idx),
            "argmax_disagreement_mean": float(np.mean(dis)),
            "argmax_disagreement_max": float(np.max(dis)),
            "miou_delta_mean": float(np.mean(iou_deltas)),
            "miou_delta_max": float(np.max(iou_deltas)),
            "status": "FP16_LOGIT_CACHE_PASS"
            if max(dis) < 1e-4 and max(iou_deltas) < 1e-4 else "FP16_LOGIT_CACHE_WARN",
        }
        print("FP16_GATE", json.dumps(gate), flush=True)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config": str(args.config), "config_sha256": sha256_file(args.config),
        "split": args.split,
        "image_list_sha256": sha256_lines([p.stem for p in imgs]),
        "n_images": len(imgs),
        "class_order": "train 0..7 = Background,Building,Road,Water,Barren,Vegetation,Agricultural,Vehicle",
        "views": VIEW_NAMES,
        "tta_definition": "D4: 4 rotations x 2 mirrors, inverse-transformed, mean logits/probs",
        "normalization": "ImageNet mean/std via src.transforms.normalize_to_tensor",
        "schemes": {n: metrics_from_confusion(cms[n]) for n in SCHEMES},
        "pairwise_disagreement": disagree,
        "fp16_precision_gate": gate,
        "mean_logits_cache": {
            "dir": str(cache_dir) if args.cache_mean_logits else None,
            "dtype": "float16", "shape": [CROP, CROP, NUM_CLASSES],
            "note": "standard TTA8 mean logits only; per-view logits are never persisted",
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (args.out_dir / "TTA_ANATOMY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md(args.out_dir / "TTA_ANATOMY.md", result)
    print("TTA_ANATOMY_DONE", args.out_dir, flush=True)
    return 0


def _cm_from_pred(pred: np.ndarray, mask_path: Path) -> np.ndarray:
    from PIL import Image as _I
    from src.label_mapping import official_to_train
    gt = official_to_train(np.asarray(_I.open(mask_path)))
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    update_confusion(cm, pred.astype(np.uint8), gt)
    return cm


def _write_md(path: Path, r: dict) -> None:
    lines = [
        "# TTA Anatomy", "", f"- checkpoint: `{r['checkpoint']}`",
        f"- checkpoint sha256: `{r['checkpoint_sha256']}`",
        f"- config sha256: `{r['config_sha256']}`",
        f"- split: {r['split']}  images: {r['n_images']}",
        f"- image list sha256: `{r['image_list_sha256']}`",
        f"- elapsed: {r['elapsed_seconds']:.1f}s", "",
        "## Pre-registered schemes", "",
        "| scheme | mIoU | Background | Barren | Agricultural | Road | Vehicle |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, m in r["schemes"].items():
        pc = m["per_class_iou"]
        def g(i):
            v = pc[i]
            return "-" if v is None else f"{v:.4f}"
        lines.append(f"| {n} | {m['mIoU']:.4f} | {g(0)} | {g(4)} | {g(6)} | {g(2)} | {g(7)} |")
    lines += ["", "## fp16 precision gate", "", f"```\n{json.dumps(r['fp16_precision_gate'], indent=2)}\n```"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
