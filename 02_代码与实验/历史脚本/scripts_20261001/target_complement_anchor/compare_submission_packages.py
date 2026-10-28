#!/usr/bin/env python3
"""Compare two compliant submission ZIPs without target ground truth.

The output is descriptive behavior only.  It must never be called accuracy,
used to auto-submit, or used to choose among multiple endpoints.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

import tca_common as common

CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]


def read_mask(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    payload = archive.read(name)
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        array = np.asarray(image)
        if image.mode != "L" or image.size != (1024, 1024) or array.dtype != np.uint8:
            raise RuntimeError(f"bad PNG contract: {name}")
        if int(array.min()) < 1 or int(array.max()) > 8:
            raise RuntimeError(f"label range violation: {name}")
        return array.astype(np.int16) - 1


def validate_names(archive: zipfile.ZipFile) -> list[str]:
    expected = common.exact_test2_names()
    names = archive.namelist()
    if names != expected or len(names) != len(set(names)) or archive.testzip() is not None:
        raise RuntimeError("submission ZIP identity/order/CRC contract failed")
    if any("/" in name or "\\" in name for name in names):
        raise RuntimeError("submission ZIP has nested members")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    transition = np.zeros((8, 8), dtype=np.int64)
    reference_counts = np.zeros(8, dtype=np.int64)
    candidate_counts = np.zeros(8, dtype=np.int64)
    per_image = []
    total_pixels = 0
    changed_pixels = 0

    with zipfile.ZipFile(args.reference) as reference, zipfile.ZipFile(args.candidate) as candidate:
        names = validate_names(reference)
        if validate_names(candidate) != names:
            raise RuntimeError("reference/candidate member mismatch")
        for name in names:
            ref = read_mask(reference, name)
            cand = read_mask(candidate, name)
            reference_counts += np.bincount(ref.ravel(), minlength=8)[:8]
            candidate_counts += np.bincount(cand.ravel(), minlength=8)[:8]
            changed = ref != cand
            count = int(changed.sum())
            total_pixels += ref.size
            changed_pixels += count
            if count:
                transition += np.bincount(
                    (ref[changed].astype(np.int64) * 8 + cand[changed]).ravel(),
                    minlength=64,
                ).reshape(8, 8)
            per_image.append(
                {
                    "name": name,
                    "changed_pixels": count,
                    "changed_fraction": count / ref.size,
                }
            )

    per_image.sort(key=lambda row: row["changed_pixels"], reverse=True)
    denominator = int(reference_counts.sum())
    ref_share = reference_counts / denominator
    cand_share = candidate_counts / denominator
    transitions = []
    for source in range(8):
        for destination in range(8):
            count = int(transition[source, destination])
            if source != destination and count:
                transitions.append(
                    {
                        "from_id": source,
                        "from_class": CLASS_NAMES[source],
                        "to_id": destination,
                        "to_class": CLASS_NAMES[destination],
                        "pixels": count,
                    }
                )
    transitions.sort(key=lambda row: row["pixels"], reverse=True)

    report = {
        "schema_version": 1,
        "warning": "NO TARGET GROUND TRUTH: descriptive behavior, not accuracy evidence",
        "reference": str(args.reference),
        "reference_sha256": common.sha256_file(args.reference),
        "candidate": str(args.candidate),
        "candidate_sha256": common.sha256_file(args.candidate),
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "changed_fraction": changed_pixels / total_pixels,
        "identical_images": sum(row["changed_pixels"] == 0 for row in per_image),
        "class_shares": {
            CLASS_NAMES[index]: {
                "reference_percent": float(ref_share[index] * 100),
                "candidate_percent": float(cand_share[index] * 100),
                "delta_pp": float((cand_share[index] - ref_share[index]) * 100),
            }
            for index in range(8)
        },
        "transition_matrix_train_ids_0_to_7": transition.tolist(),
        "top_transitions": transitions[:32],
        "top_changed_images": per_image[:50],
        "test2_ground_truth_used": False,
        "platform_submission_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("warning", "changed_pixels", "changed_fraction", "identical_images")}, indent=2))


if __name__ == "__main__":
    main()
