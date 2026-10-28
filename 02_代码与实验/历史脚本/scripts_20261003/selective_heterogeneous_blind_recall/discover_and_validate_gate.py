#!/usr/bin/env python3
"""Discover SHBR rules on H300, freeze them, then inspect R1099 once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import shbr_common as common


class AuditRecord:
    def __init__(self, stem: str, gt: np.ndarray, a05: dict, armb: dict, donor: dict):
        self.stem = stem
        self.gt = gt
        self.a05_prediction = a05["prediction"].astype(np.uint8)
        self.armb_prediction = armb["prediction"].astype(np.uint8)
        self.donor_prediction = donor["prediction"].astype(np.uint8)
        self.a05_selected = a05["selected"].astype(bool)
        self.donor_margin = donor["margin"].astype(np.float32)
        self.donor_agreement = donor["agreement"].astype(np.uint8)
        shapes = {
            tuple(item.shape)
            for item in (
                self.gt,
                self.a05_prediction,
                self.armb_prediction,
                self.donor_prediction,
                self.a05_selected,
                self.donor_margin,
                self.donor_agreement,
            )
        }
        if shapes != {(256, 256)}:
            raise RuntimeError(f"cache shape mismatch for {stem}: {shapes}")

    def gate(self, rule: common.GateRule) -> np.ndarray:
        return common.apply_gate_rule(
            rule,
            self.a05_prediction,
            self.armb_prediction,
            self.donor_prediction,
            self.a05_selected,
            self.donor_margin,
            self.donor_agreement,
        )

    def combined_gate(self, rules: list[common.GateRule]) -> tuple[np.ndarray, list[np.ndarray]]:
        return common.apply_gate_rules(
            rules,
            a05_prediction=self.a05_prediction,
            armb_prediction=self.armb_prediction,
            donor_prediction=self.donor_prediction,
            a05_selected=self.a05_selected,
            donor_margin=self.donor_margin,
            donor_agreement=self.donor_agreement,
        )


def load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"missing cache: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if not required <= set(archive.files):
            raise RuntimeError(f"cache fields missing in {path}: {sorted(required - set(archive.files))}")
        return {key: archive[key] for key in required}


def load_records(stems: list[str], paths: common.ServerPaths, caches: Path, gt_encoding: str) -> list[AuditRecord]:
    records = []
    for index, stem in enumerate(stems, 1):
        with Image.open(paths.source_masks / f"{stem}.png") as image:
            resized = image.resize((256, 256), resample=Image.Resampling.NEAREST)
            raw = np.asarray(resized).copy()
        gt = common.canonical_gt_to_train(raw, gt_encoding)
        a05 = load_npz(caches / "source_a05" / f"{stem}.npz", {"prediction", "selected"})
        armb = load_npz(caches / "source_armb" / f"{stem}.npz", {"prediction"})
        donor = load_npz(caches / "source_donor" / f"{stem}.npz", {"prediction", "agreement", "margin"})
        records.append(AuditRecord(stem, gt, a05, armb, donor))
        if index % 100 == 0:
            print(json.dumps({"loaded": index, "total": len(stems)}), flush=True)
    return records


def metric_records(records: list[AuditRecord], rules: list[common.GateRule]):
    return [
        (record.gt, record.armb_prediction, record.donor_prediction, record.combined_gate(rules)[0])
        for record in records
    ]


def rule_passes_development(metrics: dict) -> bool:
    return (
        metrics["support_images"] >= 10
        and metrics["selected_pixels"] >= 256
        and metrics["selected_pixel_donor_accuracy"] >= 0.60
        and metrics["correction_damage_ratio"] >= 2.0
        and metrics["delta_miou"] > 0
    )


def check_gate(metrics: dict, bootstrap: dict) -> tuple[bool, list[str]]:
    failures = []
    if metrics["support_images"] < 20:
        failures.append("support_images<20")
    if metrics["selected_pixels"] < 512:
        failures.append("selected_pixels<512")
    if metrics["selected_pixel_donor_accuracy"] < 0.60:
        failures.append("donor_accuracy<0.60")
    if metrics["correction_damage_ratio"] < 1.5:
        failures.append("correction_damage_ratio<1.5")
    if metrics["delta_miou"] <= 0:
        failures.append("delta_miou<=0")
    if bootstrap["ci95"][0] <= 0:
        failures.append("bootstrap_ci95_lower<=0")
    deltas = np.asarray(metrics["per_class_delta_iou"], dtype=np.float64)
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0 or float(finite.min()) < -0.002:
        failures.append("per_class_iou_drop>0.20pp")
    return not failures, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = common.load_json(args.config)
    paths = common.ServerPaths.from_config(settings)
    split = common.load_json(paths.source_audit_split)
    h300_stems = [str(item) for item in split["H300"]]
    r1099_stems = [str(item) for item in split["R1099"]]
    gt_encoding = settings["source_gt_encoding"]

    h300 = load_records(h300_stems, paths, args.cache_root, gt_encoding)
    margin_thresholds: dict[int, list[float]] = {}
    for donor_class in common.BLIND_CLASSES:
        values = []
        for record in h300:
            eligible = (
                (~record.a05_selected)
                & (record.donor_prediction == donor_class)
                & (record.a05_prediction != donor_class)
                & (record.armb_prediction != donor_class)
            )
            if eligible.any():
                values.append(record.donor_margin[eligible])
        if values:
            merged = np.concatenate(values)
            margin_thresholds[donor_class] = [float(item) for item in np.quantile(merged, [0.70, 0.80, 0.90])]

    oracle_records = []
    for record in h300:
        oracle = (
            (~record.a05_selected)
            & np.isin(record.donor_prediction, common.BLIND_CLASSES)
            & (record.a05_prediction != record.donor_prediction)
            & (record.armb_prediction != record.donor_prediction)
            & (record.donor_prediction == record.gt)
        )
        oracle_records.append((record.gt, record.armb_prediction, record.donor_prediction, oracle))
    oracle_metrics = common.replacement_metrics(oracle_records)

    candidates = []
    for donor_class, thresholds in margin_thresholds.items():
        origins = sorted(
            {
                int(origin)
                for record in h300
                for origin in np.unique(record.armb_prediction[record.donor_prediction == donor_class])
                if int(origin) != donor_class
            }
        )
        for origin in origins:
            for agreement in (6, 7, 8):
                for threshold in thresholds:
                    for area in (8, 32, 128):
                        rule = common.GateRule(origin, donor_class, agreement, threshold, area)
                        metrics = common.replacement_metrics(metric_records(h300, [rule]))
                        if rule_passes_development(metrics):
                            candidates.append({"rule": rule, "metrics": metrics})
    candidates.sort(
        key=lambda item: (
            -item["metrics"]["delta_miou"],
            -item["rule"].min_agreement,
            -item["rule"].margin_threshold,
            -item["rule"].min_component_area,
            item["metrics"]["selected_pixels"],
            item["rule"].origin_class,
            item["rule"].donor_class,
        )
    )

    chosen: list[common.GateRule] = []
    current_delta = 0.0
    for item in candidates:
        rule = item["rule"]
        if any(existing.donor_class == rule.donor_class for existing in chosen):
            continue
        trial = chosen + [rule]
        trial_metrics = common.replacement_metrics(metric_records(h300, trial))
        if trial_metrics["delta_miou"] > current_delta:
            chosen = trial
            current_delta = trial_metrics["delta_miou"]
        if len(chosen) == 2:
            break

    frozen_gate = {
        "schema_version": 1,
        "formula_version": common.FORMULA_VERSION,
        "development_split": "H300",
        "development_split_sha256": common.sha256_lines(h300_stems),
        "check_split_not_opened_during_selection": True,
        "margin_quantiles": {common.CLASS_NAMES[key]: value for key, value in margin_thresholds.items()},
        "rules": [rule.to_dict() for rule in chosen],
        "rule_count": len(chosen),
    }
    frozen_gate["gate_payload_sha256"] = common.canonical_json_sha256(frozen_gate)
    common.write_json(args.output_dir / "FROZEN_GATE.json", frozen_gate)

    development = {
        "oracle": oracle_metrics,
        "passing_candidate_count": len(candidates),
        "top_candidates": [
            {"rule": item["rule"].to_dict(), "metrics": item["metrics"]} for item in candidates[:50]
        ],
        "chosen_combined": common.replacement_metrics(metric_records(h300, chosen)) if chosen else None,
    }
    if not chosen:
        decision = {
            "status": "STOP_NO_COMPLEMENTARITY",
            "reason": "No H300 rule passed the pre-registered development constraints.",
            "test2_ground_truth_used": False,
            "platform_submission_performed": False,
        }
        common.write_json(args.output_dir / "SOURCE_COMPLEMENTARITY.json", {"development": development})
        common.write_json(args.output_dir / "GATE_DECISION.json", decision)
        print(json.dumps(decision, indent=2))
        return

    # The frozen gate is on disk before the held-out R1099 cache is loaded.
    r1099 = load_records(r1099_stems, paths, args.cache_root, gt_encoding)
    check_rows = []
    all_pass = True
    for label, rules in [(f"rule_{index + 1}", [rule]) for index, rule in enumerate(chosen)] + [("combined", chosen)]:
        records = metric_records(r1099, rules)
        metrics = common.replacement_metrics(records)
        bootstrap = common.bootstrap_delta_miou(records, repetitions=2000, seed=common.SEED)
        passed, failures = check_gate(metrics, bootstrap)
        all_pass &= passed
        check_rows.append(
            {
                "label": label,
                "rules": [rule.to_dict() for rule in rules],
                "metrics": metrics,
                "bootstrap": bootstrap,
                "passed": passed,
                "failures": failures,
            }
        )
    combined_metrics = check_rows[-1]["metrics"]
    source_report = {
        "schema_version": 1,
        "warning": "Source GT donor gate evidence controls implementation risk; it does not predict test2 accuracy.",
        "development": development,
        "frozen_gate_sha256": frozen_gate["gate_payload_sha256"],
        "check_split": "R1099",
        "check_split_sha256": common.sha256_lines(r1099_stems),
        "check_results": check_rows,
        "R1099_gate_coverage": combined_metrics["selected_pixels"] / (len(r1099) * 256 * 256),
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    common.write_json(args.output_dir / "SOURCE_COMPLEMENTARITY.json", source_report)
    decision = {
        "status": "PROCEED_SOURCE_GATE" if all_pass else "STOP_NO_DEPLOYABLE_GATE",
        "frozen_gate_sha256": frozen_gate["gate_payload_sha256"],
        "all_rule_and_combined_checks_passed": all_pass,
        "failed_checks": [row["label"] for row in check_rows if not row["passed"]],
        "R1099_gate_coverage": source_report["R1099_gate_coverage"],
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    common.write_json(args.output_dir / "GATE_DECISION.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
