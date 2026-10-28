#!/usr/bin/env python3
"""Apply the frozen source gate to test2 and enforce no-GT support checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import shbr_common as common


def load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if not required <= set(archive.files):
            raise RuntimeError(f"missing fields in {path}: {required - set(archive.files)}")
        return {key: archive[key] for key in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--frozen-gate", type=Path, required=True)
    parser.add_argument("--source-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = common.load_json(args.config)
    paths = common.ServerPaths.from_config(settings)
    source_decision = common.load_json(args.source_decision)
    if source_decision.get("status") != "PROCEED_SOURCE_GATE":
        raise RuntimeError(f"source gate did not pass: {source_decision.get('status')}")
    frozen = common.load_json(args.frozen_gate)
    provided_sha = frozen.pop("gate_payload_sha256", None)
    calculated_sha = common.canonical_json_sha256(frozen)
    if provided_sha != calculated_sha or provided_sha != source_decision.get("frozen_gate_sha256"):
        raise RuntimeError("frozen gate receipt SHA mismatch")
    rules = [common.GateRule.from_mapping(item) for item in frozen["rules"]]
    if not 1 <= len(rules) <= 2:
        raise RuntimeError(f"invalid frozen rule count: {len(rules)}")

    names = [Path(name).stem for name in common.exact_test2_names()]
    total_pixels = len(names) * 256 * 256
    selected_pixels = 0
    support_images = 0
    per_rule_pixels = [0 for _ in rules]
    per_rule_images = [0 for _ in rules]
    rows = []
    bias = torch.from_numpy(common.BSTAR)[:, None, None]
    for index, stem in enumerate(names, 1):
        raw_path = paths.a05_target_cache / f"{stem}.npy"
        raw = np.load(raw_path, allow_pickle=False)
        if raw.shape != (8, 256, 256) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise RuntimeError(f"invalid A05 target cache: {raw_path}")
        probabilities = (torch.from_numpy(raw) + bias).softmax(0)
        confidence, a05_prediction = probabilities.max(0)
        blind = torch.zeros_like(a05_prediction, dtype=torch.bool)
        for class_id in common.BLIND_CLASSES:
            blind |= a05_prediction == class_id
        selected = (confidence >= common.CONF_NORMAL) | (blind & (confidence >= common.CONF_BLIND))
        armb = load_npz(args.cache_root / "target_armb" / f"{stem}.npz", {"prediction"})
        donor = load_npz(
            args.cache_root / "target_donor" / f"{stem}.npz",
            {"probabilities", "prediction", "agreement", "margin"},
        )
        combined, individual = common.apply_gate_rules(
            rules,
            a05_prediction=a05_prediction.numpy().astype(np.uint8),
            armb_prediction=armb["prediction"].astype(np.uint8),
            donor_prediction=donor["prediction"].astype(np.uint8),
            a05_selected=selected.numpy(),
            donor_margin=donor["margin"].astype(np.float32),
            donor_agreement=donor["agreement"].astype(np.uint8),
        )
        if np.any(combined & selected.numpy()):
            raise RuntimeError(f"G overlaps M: {stem}")
        count = int(combined.sum())
        selected_pixels += count
        support_images += int(count > 0)
        for rule_index, mask in enumerate(individual):
            rule_count = int(mask.sum())
            per_rule_pixels[rule_index] += rule_count
            per_rule_images[rule_index] += int(rule_count > 0)
        donor_probabilities = donor["probabilities"].astype(np.float32)
        if donor_probabilities.shape != (8, 256, 256) or not np.isfinite(donor_probabilities).all():
            raise RuntimeError(f"invalid donor target probability cache: {stem}")
        sums = donor_probabilities.sum(0)
        if not np.allclose(sums, 1.0, atol=2e-3, rtol=0):
            raise RuntimeError(f"donor target probabilities do not sum to one: {stem}")
        np.savez_compressed(
            args.output_dir / f"{stem}.npz",
            donor_probabilities=donor["probabilities"].astype(np.float16),
            gate=combined.astype(np.uint8),
        )
        rows.append({"stem": stem, "gate_pixels": count, "per_rule_pixels": [int(mask.sum()) for mask in individual]})
        if index % 100 == 0:
            print(json.dumps({"target_gate": index, "total": len(names)}), flush=True)

    coverage = selected_pixels / total_pixels
    source_coverage = float(source_decision["R1099_gate_coverage"])
    ratio = coverage / source_coverage if source_coverage > 0 else float("inf")
    failures = []
    if support_images < 50:
        failures.append("support_images<50")
    if not 0.00002 <= coverage <= 0.0075:
        failures.append("coverage_outside_0.002_to_0.75_percent")
    if not 0.25 <= ratio <= 4.0:
        failures.append("target_source_coverage_ratio_outside_0.25_to_4")
    for index, images in enumerate(per_rule_images):
        if images < 20:
            failures.append(f"rule_{index + 1}_support_images<20")
    manifest = common.content_manifest(args.output_dir)
    report = {
        "schema_version": 1,
        "warning": "NO TEST2 GT: support/distribution checks only, not accuracy evidence.",
        "status": "PROCEED_TARGET_SUPPORT" if not failures else "STOP_TARGET_OUT_OF_SUPPORT",
        "frozen_gate_sha256": provided_sha,
        "images": len(names),
        "support_images": support_images,
        "gate_pixels": selected_pixels,
        "total_native_grid_pixels": total_pixels,
        "gate_coverage": coverage,
        "R1099_gate_coverage": source_coverage,
        "target_source_coverage_ratio": ratio,
        "per_rule_images": per_rule_images,
        "per_rule_pixels": per_rule_pixels,
        "failures": failures,
        "cache_manifest": manifest,
        "rows": rows,
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    common.write_json(args.output_dir / "TARGET_SUPPORT.json", report)
    print(json.dumps({key: report[key] for key in ("status", "support_images", "gate_coverage", "failures")}, indent=2))


if __name__ == "__main__":
    main()
