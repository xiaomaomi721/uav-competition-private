#!/usr/bin/env python3
"""Error-morphology audit: what SHAPE is our remaining error?

Motivation
----------
We know the aggregate numbers (zero-bias ~0.680, calibrated ~0.693) but not the
GEOMETRY of what we still get wrong. That geometry decides which mechanism is
worth building next:

  * errors in large connected components   -> region-level reasoning is missing
                                             (argues for a mask-classification head)
  * errors hugging class boundaries        -> boundary representation / high-res
                                             decoder is what is missing
  * many small interior specks             -> small-object recall
  * diffuse, uncorrelated errors           -> probably class prior / calibration

Two independent views are produced:

  A. GT error morphology  (val split, predictions vs ground truth)
     This is the real error signal: prediction != GT.

  B. Test-domain disagreement morphology  (test1, model vs model, ZERO bias)
     Only says the models differ -- not who is right -- but it reflects the
     TARGET domain, which the val split does not. Deliberately uses zero-bias
     raw predictions so that calibration does not manufacture whole-class
     region switches.

CPU only. Writes one JSON under --outdir.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

NUM_CLASSES = 8
OFFICIAL_NAMES = {1: "Background", 2: "Building", 3: "Road", 4: "Water",
                  5: "Barren", 6: "Vegetation", 7: "Agricultural", 8: "Vehicle"}


def read_zip_preds(zip_path: Path) -> dict[str, np.ndarray]:
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            if n.lower().endswith(".png"):
                from PIL import Image
                with Image.open(io.BytesIO(z.read(n))) as im:
                    out[Path(n).stem] = np.asarray(im).copy()
    return out


def component_stats(mask: np.ndarray) -> dict:
    """Area distribution of the connected components of a boolean mask."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return {"n_components": 0, "total_px": 0, "p50": 0, "p90": 0, "p99": 0,
                "max": 0, "frac_in_ge64": 0.0, "frac_in_ge256": 0.0,
                "frac_in_ge1024": 0.0, "frac_in_ge4096": 0.0}
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes > 0]
    total = int(sizes.sum())
    order = np.sort(sizes)
    def q(p):
        return int(order[min(len(order) - 1, int(round(p / 100.0 * (len(order) - 1))))])
    def frac(lo):
        return float(sizes[sizes >= lo].sum()) / total
    return {"n_components": int(n), "total_px": total,
            "p50": q(50), "p90": q(90), "p99": q(99), "max": int(sizes.max()),
            "frac_in_ge64": frac(64), "frac_in_ge256": frac(256),
            "frac_in_ge1024": frac(1024), "frac_in_ge4096": frac(4096)}


def boundary_band(gt: np.ndarray, width: int) -> np.ndarray:
    """Pixels within `width` px of a class boundary in gt."""
    edges = np.zeros_like(gt, dtype=bool)
    edges[:, 1:] |= gt[:, 1:] != gt[:, :-1]
    edges[1:, :] |= gt[1:, :] != gt[:-1, :]
    if not edges.any():
        return edges
    dist = ndimage.distance_transform_edt(~edges)
    return dist <= width


def audit_gt(preds: dict[str, np.ndarray], gt_dir: Path,
             stems: list[str], boundary_widths=(3, 5)) -> dict:
    agg = {w: {"err_px": 0, "in_band": 0} for w in boundary_widths}
    comp_err = {w: [] for w in boundary_widths}
    interior_comp: list[dict] = []
    pair_comp: dict[str, list[dict]] = defaultdict(list)
    per_class = defaultdict(lambda: {"iou_num": 0, "iou_den": 0, "err_px": 0, "gt_px": 0})
    total_err = 0
    total_px = 0
    used = 0

    for stem in stems:
        if stem not in preds:
            continue
        gt_path = gt_dir / f"{stem}.png"
        if not gt_path.is_file():
            continue
        from PIL import Image
        with Image.open(gt_path) as h:
            gt = np.asarray(h).copy()
        pred = preds[stem]
        if pred.shape != gt.shape:
            continue
        err = pred != gt
        total_err += int(err.sum())
        total_px += int(err.size)
        used += 1
        if not err.any():
            continue

        for w in boundary_widths:
            band = boundary_band(gt, w)
            inband = int((err & band).sum())
            agg[w]["err_px"] += int(err.sum())
            agg[w]["in_band"] += inband
            interior = err & ~band
            if interior.any():
                comp_err[w].append(component_stats(interior))

        # interior-only components at width=5 (the "region" question)
        band5 = boundary_band(gt, 5)
        interior5 = err & ~band5
        if interior5.any():
            interior_comp.append(component_stats(interior5))

        # per class-pair components (confusion structure)
        for cid in range(1, NUM_CLASSES + 1):
            m = gt == cid
            if not m.any():
                continue
            pc = per_class[cid]
            pc["gt_px"] += int(m.sum())
            inter = int((pred[m] == cid).sum())
            pc["iou_num"] += inter
            pc["iou_den"] += int(m.sum()) + int((pred == cid).sum()) - inter
            pc["err_px"] += int((m & err).sum())

        wrong = err
        if wrong.any():
            lab, n = ndimage.label(wrong, structure=np.ones((3, 3), dtype=int))
            if n:
                gtv = gt[wrong]
                prv = pred[wrong]
                labv = lab[wrong]
                sizes = np.bincount(labv)
                # dominant (gt,pred) pair per component
                order = np.argsort(-sizes[1:]) + 1
                for comp in order[:200]:
                    sel = labv == comp
                    g_, p_ = gtv[sel], prv[sel]
                    if g_.size < 32:
                        continue
                    key = f"{OFFICIAL_NAMES[int(np.bincount(g_).argmax())]}->" \
                          f"{OFFICIAL_NAMES[int(np.bincount(p_).argmax())]}"
                    pair_comp[key].append(int(g_.size))

    summary = {
        "images_used": used,
        "total_error_px": total_err,
        "total_px": total_px,
        "pixel_error_rate": total_err / total_px if total_px else None,
    }
    for w in boundary_widths:
        a = agg[w]
        summary[f"boundary_band_{w}px"] = {
            "error_px": a["err_px"],
            "in_band_px": a["in_band"],
            "frac_of_error_near_boundary": a["in_band"] / a["err_px"] if a["err_px"] else None,
            "interior_error_px": a["err_px"] - a["in_band"],
            "frac_of_error_interior": 1 - (a["in_band"] / a["err_px"]) if a["err_px"] else None,
        }

    def merge(dicts):
        if not dicts:
            return None
        keys = dicts[0].keys()
        return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

    summary["interior_components_band5"] = merge(interior_comp)
    summary["per_class"] = {
        OFFICIAL_NAMES[c]: {
            "gt_px": v["gt_px"],
            "err_px": v["err_px"],
            "iou": v["iou_num"] / v["iou_den"] if v["iou_den"] else None,
            "err_rate_within_class": v["err_px"] / v["gt_px"] if v["gt_px"] else None,
        } for c, v in sorted(per_class.items())
    }
    top = sorted(pair_comp.items(), key=lambda kv: -sum(kv[1]))[:15]
    summary["top_confusion_pairs"] = [
        {"pair": k, "n_components": len(v), "total_px": int(sum(v)),
         "median_component_px": int(np.median(v)) if v else 0} for k, v in top
    ]
    return summary


def audit_disagreement(preds_a: dict, preds_b: dict, tag: str) -> dict:
    common = sorted(set(preds_a) & set(preds_b))
    total = 0
    diff = 0
    comps = []
    per_pair = defaultdict(int)
    for stem in common:
        a, b = preds_a[stem], preds_b[stem]
        if a.shape != b.shape:
            continue
        d = a != b
        total += int(d.size)
        diff += int(d.sum())
        if not d.any():
            continue
        comps.append(component_stats(d))
        lab, n = ndimage.label(d, structure=np.ones((3, 3), dtype=int))
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            av, bv = a[d], b[d]
            for ca in np.unique(av):
                sel = av == ca
                if sel.sum() < 256:
                    continue
                cb = int(np.bincount(bv[sel]).argmax())
                per_pair[f"{OFFICIAL_NAMES.get(int(ca), ca)}->"
                         f"{OFFICIAL_NAMES.get(int(cb), cb)}"] += int(sel.sum())
    merged = None
    if comps:
        merged = {k: float(np.mean([c[k] for c in comps])) for k in comps[0]}
    return {"comparison": tag, "images": len(common),
            "disagreement_rate": diff / total if total else None,
            "component_stats": merged,
            "top_switches": sorted(per_pair.items(), key=lambda kv: -kv[1])[:8]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--val-split", required=True, help="txt of stems (no extension ok)")
    ap.add_argument("--pred-zip", action="append", default=[],
                    help="tag=path/to.zip (repeatable)")
    ap.add_argument("--disagree", action="append", default=[],
                    help="tagA,tagB (repeatable)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_dir)

    stems = [Path(l.strip()).stem for l in Path(args.val_split).read_text().split()
             if l.strip()]
    print(f"val stems: {len(stems)}", flush=True)

    preds = {}
    for spec in args.pred_zip:
        tag, _, path = spec.partition("=")
        preds[tag] = read_zip_preds(Path(path))
        print(f"  loaded {tag}: {len(preds[tag])} masks", flush=True)

    report = {"val_stems": len(stems), "boundary_widths_px": [3, 5]}
    for tag, p in preds.items():
        print(f"auditing GT error for {tag} ...", flush=True)
        report[f"gt_error_{tag}"] = audit_gt(p, gt_dir, stems)

    for spec in args.disagree:
        a, _, b = spec.partition(",")
        if a in preds and b in preds:
            print(f"auditing disagreement {a} vs {b} ...", flush=True)
            report.setdefault("test_disagreement", []).append(
                audit_disagreement(preds[a], preds[b], f"{a}_vs_{b}"))

    out = outdir / "ERROR_MORPHOLOGY.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    for tag in preds:
        s = report[f"gt_error_{tag}"]
        b5 = s["boundary_band_5px"]
        ic = s.get("interior_components_band5") or {}
        print(f"\n=== {tag} ===")
        print(f"  pixel error rate        : {s['pixel_error_rate']:.4%}")
        print(f"  error near boundary(5px): {b5['frac_of_error_near_boundary']:.2%}")
        print(f"  error INTERIOR          : {b5['frac_of_error_interior']:.2%}")
        if ic:
            print(f"  interior comps: n={ic['n_components']:.0f} p50={ic['p50']:.0f} "
                  f"p90={ic['p90']:.0f} p99={ic['p99']:.0f}")
            print(f"    frac of interior err in >=1024px comps: {ic['frac_in_ge1024']:.2%}")
            print(f"    frac of interior err in >=4096px comps: {ic['frac_in_ge4096']:.2%}")
        print("  worst classes by err_rate:")
        pc = sorted(s["per_class"].items(),
                    key=lambda kv: -(kv[1]["err_rate_within_class"] or 0))[:4]
        for name, v in pc:
            print(f"    {name:12s} IoU={v['iou']:.4f} err_rate={v['err_rate_within_class']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
