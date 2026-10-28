#!/usr/bin/env python3
"""Phase D - constrained calibration diagnostics. CPU only, streams the cache.

Never performs an 8-D free search. Only Background(0) / Barren(4) /
Agricultural(6) are varied, around the historical champion vectors, plus a tiny
optional Water(3) sensitivity axis. Ranking prioritises cross-domain stability
(D0/D1/D2) over a single full-val maximum.

Streams the fp16 mean-logits cache one image at a time; the full ~24 GiB cache is
never resident in RAM.

Static test (no cache, no GPU):
    python constrained_calibration.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SNAP = Path("/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1")
DAYS = Path("/home/jhy/gpu2_day_queue")
DATA = Path("/home/jhy/datasets/2026-低空图像语义分割赛道-训练集")
NUM = 8
IGNORE = 255

# The two historical vectors. NEITHER is claimed to be 69.281290's exact bias.
FIXED8D_EXP41_STYLE = [-0.2, 0.1, 0.0709, 0.7, 3.6, 0.4, 3.3215, -0.5]
LOCKED8D_69_276229 = [-0.2, -0.0384, 0.0709, 0.7, 3.6, 0.1932, 3.3215, -0.5]

B, R, A, W = 0, 4, 6, 3   # Background, Barren, Agricultural, Water


def _v(bg=None, barren=None, agri=None, water=None, base=FIXED8D_EXP41_STYLE):
    v = list(base)
    if bg is not None:
        v[B] = bg
    if barren is not None:
        v[R] = barren
    if agri is not None:
        v[A] = agri
    if water is not None:
        v[W] = water
    return v


def candidates() -> list[dict]:
    """Pre-registered, deliberately small. No automatic expansion."""
    out = [
        {"name": "A_FIXED8D_EXP41_STYLE", "vec": list(FIXED8D_EXP41_STYLE),
         "group": "A_baseline_history"},
        {"name": "B_LOCKED8D_69_276229", "vec": list(LOCKED8D_69_276229),
         "group": "A_baseline_history"},
    ]
    # B: Barren only, narrow around 3.6
    for br in (3.2, 3.4, 3.6, 3.8):
        out.append({"name": f"B_barren_{br}", "vec": _v(barren=br),
                    "group": "B_barren_only"})
    # C: Background + Barren small 2-D
    for bg in (-0.3, -0.2, -0.1):
        for br in (3.4, 3.6, 3.8):
            out.append({"name": f"C_bg{bg}_barren{br}", "vec": _v(bg=bg, barren=br),
                        "group": "C_bg_barren_2d"})
    # D: Background + Barren + Agricultural constrained 3-D
    for bg in (-0.3, -0.2, -0.1):
        for br in (3.4, 3.6, 3.8):
            for ag in (3.1, 3.3215, 3.5):
                out.append({"name": f"D_bg{bg}_barren{br}_agri{ag}",
                            "vec": _v(bg=bg, barren=br, agri=ag),
                            "group": "D_bg_barren_agri_3d"})
    # E: Water sensitivity only (4th axis, no grid)
    for wa in (0.6, 0.7, 0.8):
        out.append({"name": f"E_water_{wa}", "vec": _v(water=wa),
                    "group": "E_water_sensitivity"})
    return out


def accumulate(cm: np.ndarray, logits: np.ndarray, bias: np.ndarray, gt: np.ndarray) -> None:
    pred = (logits + bias.reshape(1, 1, NUM)).argmax(axis=-1).astype(np.int64)
    valid = (gt != IGNORE) & (gt >= 0) & (gt < NUM)
    g = gt[valid].astype(np.int64)
    p = pred[valid]
    cm += np.bincount(g * NUM + p, minlength=NUM * NUM).reshape(NUM, NUM)


def miou_of(cm: np.ndarray) -> float:
    d = np.diag(cm).astype(np.float64)
    den = cm.sum(1) + cm.sum(0) - d
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(den > 0, d / np.maximum(den, 1e-12), np.nan)
    return float(np.nanmean(iou))


def domains() -> dict[str, int]:
    p = SNAP / "diagnostics" / "champion_loss_audit_stage4" / "image_domain_assignments.csv"
    if not p.is_file():
        return {}
    with p.open() as f:
        return {r["image_name"]: int(r["domain_id"]) for r in csv.DictReader(f)}


def rank(records: list[dict]) -> list[dict]:
    """Cross-domain stability first, then full-val."""
    for r in records:
        d = [r.get("D0", float("nan")), r.get("D1", float("nan")), r.get("D2", float("nan"))]
        base = r["baseline_full_val"]
        deltas = [x - base for x in d if not np.isnan(x)]
        r["domain_delta_spread"] = float(max(deltas) - min(deltas)) if deltas else None
        r["domain_delta_min"] = float(min(deltas)) if deltas else None
        r["all_domains_improve"] = bool(deltas) and all(x > 0 for x in deltas)
        r["full_val_delta"] = r["full_val"] - base
    def key(r):
        return (
            0 if r["all_domains_improve"] else 1,
            -(r["domain_delta_min"] if r["domain_delta_min"] is not None else -9),
            -r["full_val_delta"],
        )
    return sorted(records, key=key)


def run(model_tag: str, cache_dir: Path, out_dir: Path, limit: int = 0) -> dict:
    stems = [l.strip() for l in (SNAP / "splits" / "val.txt").read_text().splitlines() if l.strip()]
    if limit:
        stems = stems[:limit]
    dom = domains()
    cands = candidates()
    biased = {c["name"]: np.zeros((NUM, NUM), dtype=np.int64) for c in cands}
    zero_cm = np.zeros((NUM, NUM), dtype=np.int64)
    per_dom = {c["name"]: {0: np.zeros((NUM, NUM), np.int64),
                           1: np.zeros((NUM, NUM), np.int64),
                           2: np.zeros((NUM, NUM), np.int64)} for c in cands}
    dom_counts = {0: 0, 1: 0, 2: 0, -1: 0}

    from PIL import Image
    from src.label_mapping import official_to_train

    t0 = time.perf_counter()
    for i, stem in enumerate(stems):
        f = cache_dir / f"{stem}.npy"
        if not f.is_file():
            continue
        lg = np.load(f).astype(np.float32)
        gt = official_to_train(np.asarray(Image.open(DATA / "train" / "masks" / f"{stem}.png")))
        d = dom.get(f"{stem}.png", dom.get(stem, -1))
        dom_counts[d] = dom_counts.get(d, 0) + 1

        accumulate(zero_cm, lg, np.zeros(NUM, np.float32), gt)
        for c in cands:
            bias = np.asarray(c["vec"], dtype=np.float32)
            accumulate(biased[c["name"]], lg, bias, gt)
            if d in (0, 1, 2):
                accumulate(per_dom[c["name"]][d], lg, bias, gt)
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(stems)}] {time.perf_counter()-t0:.0f}s", flush=True)

    base_full = miou_of(zero_cm)
    records = []
    for c in cands:
        rec = {"name": c["name"], "group": c["group"], "calibration_vector": c["vec"],
               "full_val": miou_of(biased[c["name"]]), "baseline_full_val": base_full,
               "full_val_delta": None}
        for d in (0, 1, 2):
            rec[f"D{d}"] = miou_of(per_dom[c["name"]][d]) if per_dom[c["name"]][d].sum() else float("nan")
        records.append(rec)
    ranked = rank(records)

    golden = [r for r in ranked if r["group"] != "Z"][:3]
    res = {
        "model_tag": model_tag,
        "cache_dir": str(cache_dir),
        "n_images_used": int(zero_cm.sum() > 0 and len([s for s in stems if (cache_dir/f'{s}.npy').is_file()])),
        "domain_counts": dom_counts,
        "domain_source": str(SNAP / "diagnostics/champion_loss_audit_stage4/image_domain_assignments.csv"),
        "baseline_zero_bias_full_val": base_full,
        "historical_vectors": {
            "FIXED8D_EXP41_STYLE": FIXED8D_EXP41_STYLE,
            "LOCKED8D_69_276229": LOCKED8D_69_276229,
        },
        "attribution_warning": (
            "NEITHER historical vector is claimed to be the exact 69.281290 bias. "
            "That identity remains unresolved (GOLDEN_CALIBRATION_IDENTITY_UNRESOLVED)."),
        "ranking_rule": "cross-D0/D1/D2 direction consistency first, then full-val delta",
        "all_candidates": ranked,
        "top3": golden,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"CALIBRATION_CONSTRAINED_REPORT_{model_tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def self_test() -> int:
    rng = np.random.default_rng(0)
    # NOTE: do not name these W/R/B/A -- those are module-level class-index
    # constants (Water/Barren/Background/Agricultural) and shadowing them
    # silently corrupts the axis-containment assertion below.
    HH = WD = 32
    gt = rng.integers(0, NUM, size=(HH, WD)).astype(np.uint8)
    gt[0, 0] = IGNORE
    logits = rng.normal(size=(HH, WD, NUM)).astype(np.float32)
    cm0 = np.zeros((NUM, NUM), np.int64)
    accumulate(cm0, logits, np.zeros(NUM, np.float32), gt)
    assert cm0.sum() == HH * WD - 1
    # a bias that strongly favours Barren must raise Barren predicted pixels
    big = np.zeros(NUM, np.float32); big[R] = 10.0
    cm1 = np.zeros((NUM, NUM), np.int64)
    accumulate(cm1, logits, big, gt)
    assert cm1[:, R].sum() > cm0[:, R].sum(), "bias did not shift predictions"
    # metric bounds
    assert 0.0 <= miou_of(cm0) <= 1.0
    assert 0.0 <= miou_of(cm1) <= 1.0
    # candidate set is bounded and excludes a free 8-D search
    cs = candidates()
    assert 2 + 4 + 9 + 27 + 3 == len(cs), len(cs)
    # The two baseline rows are independent historical vectors and legitimately
    # differ from each other. Only the *perturbation* groups must stay on the
    # allowed axes.
    varied = {i for c in cs if c["group"] != "A_baseline_history"
              for i in range(NUM) if c["vec"][i] != FIXED8D_EXP41_STYLE[i]}
    assert varied <= {B, W, R, A}, f"unexpected axes varied: {varied}"
    base_names = {c["name"] for c in cs if c["group"] == "A_baseline_history"}
    assert base_names == {"A_FIXED8D_EXP41_STYLE", "B_LOCKED8D_69_276229"}, base_names
    # ranking puts stable candidates first
    recs = [
        {"name": "stable", "full_val": 0.80, "baseline_full_val": 0.79,
         "D0": 0.80, "D1": 0.80, "D2": 0.80},
        {"name": "spiky", "full_val": 0.85, "baseline_full_val": 0.79,
         "D0": 0.85, "D1": 0.70, "D2": 0.86},
    ]
    assert rank(recs)[0]["name"] == "stable", "ranking rule not applied"
    print(f"SELF_TEST_OK  candidates={len(cs)}  axes_varied={sorted(varied)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=DAYS / "calibration")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nice", action="store_true", help="lower CPU priority")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    if args.nice:
        try:
            os.nice(19)
        except Exception:
            pass
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "2")

    res = run(args.model_tag, args.cache_dir, args.out_dir, args.limit)
    print(json.dumps({"top3": [r["name"] for r in res["top3"]],
                      "baseline": res["baseline_zero_bias_full_val"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
