#!/usr/bin/env python3
"""Phase 0: forensic asset inventory, identity chain and frozen pilot split.

CPU/read-only with respect to models and caches: this script hashes and reads
existing assets but performs no model forward and no training.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _r2_contracts import (
    ATDA, CONFIG, DATA, EXPECTED_FILE_SHA, OUT, ROUND2, SEED, SNAP,
    THETA261, THETA278, THETA278_FOLDED, ATDA8K, canonical_state_sha,
    compare_states, load_state, sha256_file, sha256_lines, write_json,
)

B2 = Path("/home/jhy/rare_diag_20260915/b2_val_20260917")
M2F = Path("/home/jhy/rare_diag_20260915/m2f_gpu_20260915")


def file_record(path: Path) -> dict:
    return {"path": str(path), "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None}


def dir_record(path: Path, pattern: str = "*") -> dict:
    files = sorted(p for p in path.glob(pattern) if p.is_file()) if path.is_dir() else []
    return {"path": str(path), "exists": path.is_dir(), "count": len(files),
            "first": files[0].name if files else None,
            "last": files[-1].name if files else None}


def cache_record(path: Path, count: int, prefix: str) -> dict:
    scores = path / "scores"
    names = sorted((p.name for p in scores.glob("*.npy")),
                   key=lambda x: int(Path(x).stem.split("_")[-1])) if scores.is_dir() else []
    expected = [f"{prefix}_{i}.npy" for i in range(1, count + 1)]
    meta = path / "META.json"
    return {"path": str(path), "scores": len(names), "exact_names": names == expected,
            "meta": file_record(meta),
            "meta_content": json.loads(meta.read_text()) if meta.is_file() else None}


def checkpoint_identity() -> dict:
    paths = {"theta261": THETA261, "atda8k": ATDA8K, "theta278": THETA278,
             "theta278_folded": THETA278_FOLDED}
    states, records = {}, {}
    for tag, path in paths.items():
        state, wrapper = load_state(path)
        states[tag] = state
        records[tag] = {**file_record(path), "expected_file_sha256": EXPECTED_FILE_SHA[tag],
                        "file_sha_matches": sha256_file(path) == EXPECTED_FILE_SHA[tag],
                        "canonical_state_sha256": canonical_state_sha(state),
                        "tensor_count": len(state),
                        "wrapper_keys": sorted(wrapper) if isinstance(wrapper, dict) else None}
    base = states["theta261"]
    train = states["atda8k"]
    composed = states["theta278"]
    folded = states["theta278_folded"]
    composition = {"same_keys": set(base) == set(train) == set(composed)}
    max_err, bad, out_scope = 0.0, [], []
    import torch
    for name in base:
        if base[name].is_floating_point():
            want = base[name].float() + 1.25 * (train[name].float() - base[name].float())
            err = float((want.to(composed[name].dtype) - composed[name]).float().abs().max())
            max_err = max(max_err, err)
            if err != 0.0:
                bad.append(name)
        elif not torch.equal(base[name], composed[name]):
            bad.append(name)
        if not (name.startswith("segformer.stages.2.") or name.startswith("segformer.stages.3.")) and not torch.equal(base[name], composed[name]):
            out_scope.append(name)
    composition.update({"formula": "theta261 + 1.25*(atda8k-theta261)",
                        "max_abs_error": max_err, "mismatched_tensors": bad,
                        "out_of_stage2_stage3_changed": out_scope,
                        "exact": not bad})
    folded_cmp = compare_states(composed, folded)
    expected_fold_only = folded_cmp.get("changed_names") == ["decode_head.classifier.bias"]
    return {"records": records,
            "lineage": {"train8k_vs_theta261": compare_states(base, train),
                        "composition_check": composition,
                        "prefold_vs_folded": folded_cmp,
                        "folded_only_classifier_bias": expected_fold_only},
            "authoritative_theta278": {
                "meaning": "scored prefold model; old Bstar remains external deployment operator",
                "path": str(THETA278), "file_sha256": records["theta278"]["sha256"],
                "canonical_state_sha256": records["theta278"]["canonical_state_sha256"],
                "online_package_sha256": "7e5018549c2b3fb76f076fb62d3553951aeffef4aacfd01eeb8c57dde917edb0",
                "online_score_percent": 70.569757}}


def source_split_and_pilot() -> dict:
    train = [x.strip() for x in (SNAP / "splits/train.txt").read_text().split() if x.strip()]
    val = [x.strip() for x in (SNAP / "splits/val.txt").read_text().split() if x.strip()]
    if len(train) != 5597 or len(val) != 1399 or set(train) & set(val) or len(set(train) | set(val)) != 6996:
        raise RuntimeError("source split identity failed")
    gt_dir = B2 / "gt"
    by_stem = {}
    for stem in val:
        arr = np.load(gt_dir / f"{stem}.npy", allow_pickle=False)
        legal = np.isin(arr, list(range(8)) + [255])
        if not bool(legal.all()):
            raise RuntimeError(f"illegal GT labels: {stem}")
        valid = arr != 255
        counts = np.bincount(arr[valid], minlength=8)
        by_stem[stem] = counts
    # Deterministic greedy multilabel coverage: rarity-weighted round-robin score,
    # no model outputs are touched.  Fixed tie-break is a seed-keyed SHA.
    total_presence = np.sum(np.stack([x > 0 for x in by_stem.values()]), axis=0)
    selected, remaining = [], set(val)
    target = 300
    covered = np.zeros(8, dtype=np.int64)
    while len(selected) < target:
        best = None
        for stem in remaining:
            present = by_stem[stem] > 0
            rarity = np.sum(present / np.maximum(total_presence, 1))
            deficit = np.sum(present / (covered + 1))
            tie = sha256_lines([str(SEED), stem])
            key = (deficit + rarity, tie)
            if best is None or key > best[0]:
                best = (key, stem, present)
        _, stem, present = best
        selected.append(stem); remaining.remove(stem); covered += present.astype(np.int64)
    h300 = selected
    r1099 = [s for s in val if s not in set(h300)]
    return {"train_count": len(train), "val_count": len(val), "disjoint": True,
            "union_count": len(set(train) | set(val)),
            "train_sha256": sha256_lines(train), "val_sha256": sha256_lines(val),
            "selection_policy": "GT-only deterministic rarity/coverage greedy; seed 20260922; no model outputs",
            "H300": h300, "H300_sha256": sha256_lines(h300),
            "R1099": r1099, "R1099_sha256": sha256_lines(r1099),
            "H300_class_image_presence": covered.tolist()}


def main() -> None:
    phase = OUT / "00_inventory"
    phase.mkdir(parents=True, exist_ok=False)
    identity = checkpoint_identity()
    write_json(phase / "THETA278_IDENTITY.json", identity)
    split = source_split_and_pilot()
    write_json(phase / "SOURCE_AUDIT_SPLIT.json", split)

    caches = {
        "theta261_test1_reference": {"path": str(ATDA / "ref261_cache"),
                                      "scores": dir_record(ATDA / "ref261_cache/scores", "*.npy"),
                                      "meta": file_record(ATDA / "ref261_cache/FORWARD_META.json")},
        "theta278_test1": cache_record(ROUND2 / "cache_278_test1_raw", 500, "test1"),
        "theta261_test2": cache_record(ROUND2 / "cache_261_test2_raw", 1300, "test2"),
        "theta278_test2": cache_record(ROUND2 / "cache_278_test2_raw", 1300, "test2"),
        "b2_source_holdout": {"gt": dir_record(B2 / "gt", "*.npy"),
                              "logits_champ": dir_record(B2 / "logits_champ", "*.npy"),
                              "logits_barrenAgri": dir_record(B2 / "logits_barrenAgri", "*.npy"),
                              "meta": file_record(B2 / "B2_META.json"),
                              "reuse_note": "GT reusable; logits are single-view full-res and are not theta261/theta278 D4 deployment caches"},
    }
    features = {
        "deep_source6996": file_record(ROUND2 / "deep_features_exp18/source6996.npz"),
        "deep_test1": file_record(ROUND2 / "deep_features_exp18/test1.npz"),
        "deep_test2": file_record(ROUND2 / "deep_features_exp18/test2.npz"),
        "deep_meta": file_record(ROUND2 / "deep_features_exp18/META.json"),
        "image_pca30": file_record(ROUND2 / "image_space_bootstrap/features_pca30.npz"),
        "feature_report": file_record(ROUND2 / "feature_space/FEATURE_SPACE_REPORT.json"),
    }
    donors = {
        "barrenAgri": {"checkpoint": file_record(SNAP / "stage59_runs/stage59_w1_exp59_b3_cwfa_ce_regionC_barrenAgri_seed42/best_miou.pt"),
                        "source_cache": caches["b2_source_holdout"]["logits_barrenAgri"],
                        "protocol": "existing single-view full-resolution; preliminary audit only"},
        "M2F56": {"checkpoint": file_record(M2F / "formal_run1b/scores/score_iter56000.pt"),
                   "multiscale_config": file_record(M2F / "test1_multiscale_v1/INFERENCE_CONFIG.json"),
                   "source_holdout_cache_found": False},
        "Exp18_source_only": {"checkpoint": file_record(SNAP / "stage18_runs/stage18_w1_exp18_b3_1024_full_ce_formal_seed42/best_miou.pt"),
                              "source_holdout_cache_found": False},
    }
    geometry = {"historical_m2f_multiscale": dir_record(M2F / "test1_multiscale_v1"),
                "overlap_tile_cache_found": False,
                "note": "historical M2F multiscale is a different model and was negative; protocol reference only"}
    fourier = {"existing_radial_features": features["image_pca30"],
               "FDA_transform_found": False, "FDA_training_allowed": False}
    forward_budget = {
        "frozen": True, "holdout_forwards": {"theta261": 1, "theta278": 1},
        "geometry_H300": ["G1_fixed_768_overlap_plus_whole", "G2_fixed_1p25_plus_whole"],
        "geometry_full1399_max": 1, "donor_holdout_forwards_max": 3,
        "DART_variants": ["D0", "D1", "D2_if_gate_pass"],
        "DART_checkpoints": [1000, 2000], "max_steps": 2000,
        "forbidden": ["regenerate valid test1/test2 caches", "DART 4k/8k/16k",
                      "geometry grid", "more than 3 donors", "FDA training",
                      "test2 package", "platform submission"]}
    write_json(phase / "FORWARD_BUDGET.json", forward_budget)
    registry = {"created": datetime.now(timezone.utc).isoformat(), "caches": caches}
    write_json(phase / "CACHE_REGISTRY.json", registry)
    inventory = {"created": datetime.now(timezone.utc).isoformat(),
                 "scope": "ROUND2_MULTIPROBE_PHASE0_NO_FORWARD",
                 "identity": identity, "source_split": split, "caches": caches,
                 "features": features, "donors": donors, "geometry": geometry,
                 "fourier": fourier, "forward_budget": forward_budget,
                 "environment": {"python": sys.version, "platform": platform.platform()},
                 "test2_ground_truth_used": False, "platform_submission_performed": False}
    write_json(phase / "ASSET_INVENTORY.json", inventory)

    lines = ["# PILOT ASSET INVENTORY", "", "## Identity",
             f"- theta261: `{identity['records']['theta261']['sha256']}`",
             f"- ATDA 8k training checkpoint: `{identity['records']['atda8k']['sha256']}`",
             f"- authoritative theta278 = theta261 + 1.25×(8k−theta261): `{identity['records']['theta278']['sha256']}`",
             f"- theta278 canonical tensor SHA: `{identity['records']['theta278']['canonical_state_sha256']}`",
             f"- composition exact: **{identity['lineage']['composition_check']['exact']}**",
             "- `c3e87…` and `5f8a…` are different lineage stages, not conflicting aliases.", "",
             "## Reusable caches",
             "- Reuse existing theta261/theta278 test1/test2 native D4 caches; no repeated forward.",
             "- Reuse verified 1399 GT from b2_val; old b2 logits are single-view, older models, and cannot stand in for 261/278 D4 caches.",
             "- Reuse Exp18 source/test1/test2 pooled features and image PCA features.", "",
             "## Missing assets authorized by forward budget",
             "- Exact theta261 and theta278 source-holdout D4 native caches.",
             "- Fixed G1/G2 H300 geometry forwards.",
             "- At most three provenance-approved missing donor holdout caches after geometry.", "",
             "## Historical evidence",
             "- M2F true input multiscale exists and was negative; it is protocol reference, not reusable theta278 geometry evidence.",
             "- No compatible overlap-tile cache and no deterministic FDA transform were found.", "",
             "## Frozen pilot split",
             f"- H300 SHA: `{split['H300_sha256']}`; class image presence={split['H300_class_image_presence']}",
             f"- R1099 SHA: `{split['R1099_sha256']}`", "",
             "## Hard guards", "- No platform submission.", "- No test2 GT.",
             "- No DART beyond 2k.", "- No large grid or automatic continuation."]
    (phase / "PILOT_ASSET_INVENTORY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(phase), "theta278": identity["authoritative_theta278"],
                      "H300_sha": split["H300_sha256"], "R1099_sha": split["R1099_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
