#!/usr/bin/env python3
"""CPU-only, pooled GT-error and test-disagreement morphology.

No torch/model imports. Prediction sources must be declared separately by domain.
Areas are pooled across images; connected components never span image borders.
Morphology describes errors/disagreements, not their cause or a head's efficacy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage

NAMES = {1: "Background", 2: "Building", 3: "Road", 4: "Water",
         5: "Barren", 6: "Vegetation", 7: "Agricultural", 8: "Vehicle"}
STRUCTURE = np.ones((3, 3), dtype=np.uint8)
THRESHOLDS = (64, 256, 1024, 4096)


def ratio(a, b):
    return float(a / b) if b else None


class Areas:
    """Exact area histogram: bounded by image size, no component truncation."""
    def __init__(self):
        self.hist = Counter()
        self.images = 0
        self.images_nonempty = 0

    def add(self, mask):
        self.images += 1
        if not mask.any():
            return
        self.images_nonempty += 1
        lab, _ = ndimage.label(mask, structure=STRUCTURE)
        sizes = np.bincount(lab.ravel())[1:]
        values, counts = np.unique(sizes, return_counts=True)
        self.hist.update({int(v): int(c) for v, c in zip(values, counts)})

    def result(self):
        n = sum(self.hist.values())
        px = sum(a * c for a, c in self.hist.items())
        ordered = sorted(self.hist)
        cumulative = np.cumsum([self.hist[a] for a in ordered])

        def quantile(q):
            if not n:
                return None
            # numpy's default linear interpolation, computed from the histogram.
            index = (n - 1) * q
            lo, hi = int(np.floor(index)), int(np.ceil(index))
            a = ordered[int(np.searchsorted(cumulative, lo, side="right"))]
            b = ordered[int(np.searchsorted(cumulative, hi, side="right"))]
            return float(a + (b - a) * (index - lo))

        return {
            "images": self.images, "images_nonempty": self.images_nonempty,
            "n_components": n, "total_px": px,
            "p50": quantile(.5), "p90": quantile(.9), "p99": quantile(.99),
            "max": max(ordered) if n else None,
            "pixel_share_by_min_component_area": {
                str(t): ratio(sum(a * c for a, c in self.hist.items() if a >= t), px)
                for t in THRESHOLDS},
        }


class Masks:
    def __init__(self, path, encoding="official"):
        self.path = Path(path)
        self.encoding = encoding
        self.z = zipfile.ZipFile(self.path) if self.path.is_file() else None
        paths = self.z.namelist() if self.z else [str(p) for p in self.path.rglob("*.png")]
        self.entries = {}
        for p in paths:
            if not p.lower().endswith(".png"):
                continue
            stem = Path(p).stem
            if stem in self.entries:
                raise ValueError(f"duplicate prediction stem: {stem} in {path}")
            self.entries[stem] = p
        if not self.entries:
            raise ValueError(f"no masks in {path}")

    def read(self, stem):
        entry = self.entries[stem]
        source = io.BytesIO(self.z.read(entry)) if self.z else entry
        with Image.open(source) as im:
            mask = np.asarray(im).copy()
        if mask.ndim != 2:
            raise ValueError(f"not a 2D class mask: {entry}")
        if self.encoding == "train":
            if mask.min() < 0 or mask.max() > 7:
                raise ValueError(f"expected train IDs 0..7: {entry}")
            mask = mask.astype(np.uint8) + 1
        elif self.encoding == "official_gtmasked":
            if mask.min() < 0 or mask.max() > 8:
                raise ValueError(f"expected official IDs 0..8: {entry}")
        elif mask.min() < 1 or mask.max() > 8:
            raise ValueError(f"expected official prediction IDs 1..8: {entry}")
        return mask

    def receipt(self):
        result = {"path": str(self.path), "count": len(self.entries),
                  "encoding": self.encoding}
        if self.z:
            h = hashlib.sha256()
            with self.path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            result["zip_sha256"] = h.hexdigest()
        return result


def boundary_distance(labels, valid):
    """Symmetric 4-neighbour label transitions, excluding Ignore interfaces.

    Both pixel centres adjoining a transition are seeds. Distance is Euclidean;
    image borders are not seeds. A 5px band is <=5 from these seeds, not exactly
    5px from the subpixel interface. Ignore pixels never contribute to counts.
    """
    edge = np.zeros(labels.shape, dtype=bool)
    horizontal = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    vertical = valid[1:, :] & valid[:-1, :] & (labels[1:, :] != labels[:-1, :])
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return ndimage.distance_transform_edt(~edge) if edge.any() else np.full(labels.shape, np.inf)


def confusion(a, b, valid):
    codes = (a[valid].astype(np.int64) - 1) * 8 + b[valid].astype(np.int64) - 1
    return np.bincount(codes, minlength=64).reshape(8, 8)


def audit(stems, load_a, load_b, mode, coverage):
    all_areas = Areas()
    interior = {w: Areas() for w in (3, 5)}
    pair_areas, pair_interior = {}, {}
    cm = np.zeros((8, 8), dtype=np.int64)
    bands = {w: Counter() for w in (3, 5)}
    per_image = []
    for i, stem in enumerate(stems, 1):
        a, b = load_a(stem), load_b(stem)
        if a.ndim != 2 or a.shape != b.shape:
            raise ValueError(f"shape mismatch at {stem}: {a.shape}, {b.shape}")
        if a.min() < 0 or a.max() > 8:
            raise ValueError(f"expected official GT IDs 0..8: {stem}")
        valid = a != 0 if mode == "gt_error" else np.ones(a.shape, dtype=bool)
        if np.any(valid & ((b < 1) | (b > 8))):
            raise ValueError(f"prediction outside IDs 1..8 on valid pixels: {stem}")
        delta = valid & (a != b)
        image_cm = confusion(a, b, valid)
        cm += image_cm
        all_areas.add(delta)
        distance_a = boundary_distance(a, valid)
        distance_b = boundary_distance(b, valid) if mode != "gt_error" else None
        current_interior = {}
        for w in bands:
            # Test uses union of BOTH prediction boundary bands, NOT a GT band.
            band = distance_a <= w
            if distance_b is not None:
                band |= distance_b <= w
            band &= valid
            inside = delta & ~band
            current_interior[w] = inside
            interior[w].add(inside)
            bands[w].update(valid_px=int(valid.sum()), band_px=int(band.sum()),
                            delta_px=int(delta.sum()), band_delta_px=int((delta & band).sum()))
        # Every true directed class-pair gets its own connected-component mask.
        rows, cols = np.nonzero(image_cm)
        for row, col in zip(rows, cols):
            if row == col:
                continue
            key = f"{NAMES[int(row)+1]}->{NAMES[int(col)+1]}"
            mask = valid & (a == row + 1) & (b == col + 1)
            pair_areas.setdefault(key, Areas()).add(mask)
            if mode == "gt_error":
                pair_interior.setdefault(key, Areas()).add(mask & current_interior[5])
        per_image.append({"stem": stem, "valid_px": int(valid.sum()),
                          "delta_px": int(delta.sum()),
                          "interior5_delta_px": int(current_interior[5].sum())})
        if i % 50 == 0:
            print(f"  {mode}: {i}/{len(stems)}", flush=True)

    total = int(cm.sum())
    delta_px = total - int(np.trace(cm))
    result = {"status": "complete" if stems else "unavailable_no_overlap",
              "mode": mode, "coverage": coverage, "images_used": len(stems),
              "valid_px": total, "delta_px": delta_px,
              "delta_rate": ratio(delta_px, total), "components": all_areas.result(),
              "boundary_reference": "GT" if mode == "gt_error" else "union_of_A_and_B_prediction_boundaries",
              "boundary": {}, "confusion_rows_A_cols_B_ids1_to8": cm.tolist(),
              "directed_pairs": {}, "per_image": per_image}
    for w, v in bands.items():
        near, bandpx = v["band_delta_px"], v["band_px"]
        far, interiorpx = v["delta_px"] - near, v["valid_px"] - bandpx
        result["boundary"][str(w)] = {
            "band_px": bandpx, "band_delta_px": near,
            "share_of_delta_in_band": ratio(near, delta_px),
            "share_of_valid_in_band": ratio(bandpx, total),
            "delta_rate_in_band": ratio(near, bandpx),
            "interior_px": interiorpx, "interior_delta_px": far,
            "share_of_delta_interior": ratio(far, delta_px),
            "delta_rate_interior": ratio(far, interiorpx),
            "interior_components": interior[w].result()}
    for key, areas in pair_areas.items():
        entry = {"components": areas.result()}
        entry["share_of_all_delta"] = ratio(entry["components"]["total_px"], delta_px)
        if key in pair_interior:
            entry["interior5_components"] = pair_interior[key].result()
        result["directed_pairs"][key] = entry
    assert all_areas.result()["total_px"] == delta_px
    assert sum(p["components"]["total_px"] for p in result["directed_pairs"].values()) == delta_px
    if mode == "gt_error":
        result["per_class"] = {}
        for i in range(8):
            tp = int(cm[i, i])
            gt, pred = int(cm[i, :].sum()), int(cm[:, i].sum())
            result["per_class"][NAMES[i + 1]] = {
                "gt_px": gt, "pred_px": pred, "tp": tp, "fn": gt - tp, "fp": pred - tp,
                "iou": ratio(tp, gt + pred - tp), "gt_error_rate": ratio(gt - tp, gt)}
    return result


def self_test():
    a = Areas()
    a.add(np.ones((2, 2), dtype=bool))
    a.add(np.ones((1, 100), dtype=bool))
    a.add(np.zeros((2, 2), dtype=bool))
    r = a.result()
    assert r["p50"] == 52 and r["total_px"] == 104 and r["n_components"] == 2
    assert r["pixel_share_by_min_component_area"]["64"] == 100 / 104
    assert np.isclose(r["p90"], np.quantile([4, 100], .9))
    gt = {"x": np.array([[0, 1, 1], [2, 2, 1]], dtype=np.uint8),
          "correct": np.ones((2, 3), dtype=np.uint8)}
    pred = {"x": np.array([[8, 1, 3], [1, 2, 1]], dtype=np.uint8),
            "correct": gt["correct"].copy()}
    r = audit(list(gt), gt.__getitem__, pred.__getitem__, "gt_error", {})
    assert r["valid_px"] == 11 and r["delta_px"] == 2
    assert r["per_class"]["Road"]["fp"] == 1
    assert r["per_class"]["Background"]["iou"] == 8 / 10
    assert r["directed_pairs"]["Background->Road"]["components"]["total_px"] == 1
    labels = np.array([[1, 1, 2, 2]], dtype=np.uint8)
    d = boundary_distance(labels, np.ones_like(labels, dtype=bool))
    assert d.tolist() == [[1., 0., 0., 1.]]
    labels = np.array([[1, 0, 2]], dtype=np.uint8)
    assert np.isinf(boundary_distance(labels, labels != 0)).all()
    d = audit(list(gt), pred.__getitem__, pred.__getitem__, "test_disagreement", {})
    assert d["delta_px"] == 0 and d["components"]["p50"] is None
    empty = audit([], None, None, "gt_error", {})
    assert empty["status"] == "unavailable_no_overlap"
    print("SELF_TEST_PASS: pooling, FP on absent class, perfect images, Ignore, pairs, symmetric boundaries, empty coverage")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir")
    parser.add_argument("--gt-dir")
    parser.add_argument("--val-split")
    parser.add_argument("--val", action="append", default=[], help="tag=directory_or_zip (official IDs)")
    parser.add_argument("--val-train-ids", action="append", default=[], help="tag=directory_or_zip (train IDs 0..7)")
    parser.add_argument("--val-gtmasked", action="append", default=[], help="official IDs; 0 permitted ONLY where GT is Ignore")
    parser.add_argument("--test", action="append", default=[], help="tag=directory_or_zip, ZERO bias, official IDs")
    parser.add_argument("--compare", action="append", default=[], help="A,B test model tags")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU audit requires CUDA_VISIBLE_DEVICES='' explicitly")
    if args.self_test:
        self_test()
        return
    if not args.outdir:
        parser.error("--outdir required")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 2, "cpu_only": True,
              "method": {"connectivity": 8, "component_pooling": "all components across images, linear quantiles",
                         "pixel_shares": "sum areas above threshold / all delta pixels; NOT per-image mean",
                         "pairs": "directed exact class-pair masks; no component-size cutoff or cap",
                         "boundary": "symmetric 4-neighbour seeds, Euclidean <=3/5 pixels, Ignore interfaces excluded",
                         "interpretation": "test differences have no correctness labels; shape does not establish cause"},
              "gt_error": {}, "test_disagreement": {}, "sources": {}}
    if args.val or args.val_train_ids or args.val_gtmasked:
        if not args.gt_dir or not args.val_split:
            parser.error("GT analysis requires --gt-dir and --val-split")
        stems = [Path(s).stem for s in Path(args.val_split).read_text().split()]
        if len(stems) != len(set(stems)):
            raise ValueError("duplicate validation stems")
        for specs, encoding in ((args.val, "official"), (args.val_train_ids, "train"),
                                (args.val_gtmasked, "official_gtmasked")):
            for spec in specs:
                tag, path = spec.split("=", 1)
                masks = Masks(path, encoding)
                missing_pred = sorted(set(stems) - masks.entries.keys())
                missing_gt = [s for s in stems if not (Path(args.gt_dir) / (s + ".png")).is_file()]
                used = sorted(set(stems) & masks.entries.keys() - set(missing_gt))
                def load_gt(stem):
                    with Image.open(Path(args.gt_dir) / (stem + ".png")) as im:
                        return np.asarray(im).copy()
                coverage = {"requested": len(stems), "missing_pred": missing_pred,
                            "missing_gt": missing_gt, "extra_pred": sorted(masks.entries.keys() - set(stems))}
                print(f"GT {tag}: {len(used)}/{len(stems)}", flush=True)
                report["gt_error"][tag] = audit(used, load_gt, masks.read, "gt_error", coverage)
                report["sources"]["val:" + tag] = masks.receipt()
    else:
        report["gt_status"] = "unavailable_no_verified_validation_predictions_supplied"
    tests = {}
    for spec in args.test:
        tag, path = spec.split("=", 1)
        tests[tag] = Masks(path)
        if any(not s.startswith("test1_") for s in tests[tag].entries):
            raise ValueError("test source includes non-test1 stems")
        report["sources"]["test:" + tag] = tests[tag].receipt()
    for spec in args.compare:
        a, b = spec.split(",")
        common = sorted(tests[a].entries.keys() & tests[b].entries.keys())
        coverage = {"A_count": len(tests[a].entries), "B_count": len(tests[b].entries),
                    "A_only": sorted(tests[a].entries.keys() - tests[b].entries.keys()),
                    "B_only": sorted(tests[b].entries.keys() - tests[a].entries.keys())}
        print(f"TEST {a} vs {b}: {len(common)}", flush=True)
        report["test_disagreement"][a + "_vs_" + b] = audit(
            common, tests[a].read, tests[b].read, "test_disagreement", coverage)
    output = outdir / "ERROR_MORPHOLOGY.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"MORPHOLOGY_DONE {output}", flush=True)


if __name__ == "__main__":
    main()
