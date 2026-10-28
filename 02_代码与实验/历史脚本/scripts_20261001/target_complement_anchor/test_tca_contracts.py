#!/usr/bin/env python3
from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import package_tca
import tca_common as common


class TcaContracts(unittest.TestCase):
    def test_exact_names_are_numeric_not_lexicographic(self):
        names = common.exact_test2_names()
        self.assertEqual(len(names), 1300)
        self.assertEqual(names[:4], ["test2_1.png", "test2_2.png", "test2_3.png", "test2_4.png"])
        self.assertEqual(names[-1], "test2_1300.png")
        lexical = sorted(names)
        self.assertEqual(lexical[:4], ["test2_1.png", "test2_10.png", "test2_100.png", "test2_1000.png"])
        self.assertNotEqual(names, lexical)

    def test_d4_roundtrip_non_square_and_unique(self):
        tensor = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
        seen = set()
        for view in range(8):
            transformed = common.d4_forward(tensor, view)
            restored = common.d4_inverse(transformed, view)
            self.assertTrue(torch.equal(restored, tensor))
            key = (tuple(transformed.shape), transformed.numpy().tobytes())
            self.assertNotIn(key, seen)
            seen.add(key)
        self.assertEqual(len(seen), 8)

    def test_teacher_mask_boundaries_and_complement(self):
        probabilities = torch.full((1, 8, 1, 4), 0.02)
        # non-blind at exactly 0.98 -> selected
        probabilities[0, :, 0, 0] = (0.02 / 7)
        probabilities[0, 1, 0, 0] = 0.98
        # blind at exactly 0.50 -> selected
        probabilities[0, :, 0, 1] = 0.50 / 7
        probabilities[0, 3, 0, 1] = 0.50
        # non-blind below 0.98 -> complement
        probabilities[0, :, 0, 2] = 0.021 / 7
        probabilities[0, 2, 0, 2] = 0.979
        # blind below 0.50 -> complement
        probabilities[0, :, 0, 3] = 0.501 / 7
        probabilities[0, 4, 0, 3] = 0.499

        _, predicted, selected, complement = common.teacher_masks(probabilities)
        self.assertEqual(predicted.flatten().tolist(), [1, 3, 2, 4])
        self.assertEqual(selected.flatten().tolist(), [True, True, False, False])
        self.assertTrue(torch.equal(complement, ~selected))
        self.assertTrue(torch.all(selected ^ complement))

    def test_masked_means_and_empty_fail_closed(self):
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        selected = torch.tensor([[True, False, True, False]])
        complement = ~selected
        self.assertAlmostEqual(float(common.masked_mean(values, selected, "selected")), 2.0)
        self.assertAlmostEqual(float(common.masked_mean(values, complement, "complement")), 3.0)
        with self.assertRaises(RuntimeError):
            common.masked_mean(values, torch.zeros_like(selected), "empty")

    def test_per_pixel_kl_matches_manual_and_handles_zero_teacher_probability(self):
        teacher = torch.tensor([[[[0.7]], [[0.3]]]], dtype=torch.float32)
        student = torch.tensor([[[[0.4]], [[0.6]]]], dtype=torch.float32)
        actual = common.per_pixel_kl(teacher, student.log())
        expected = 0.7 * np.log(0.7 / 0.4) + 0.3 * np.log(0.3 / 0.6)
        self.assertAlmostEqual(float(actual), float(expected), places=6)

        one_hot_teacher = torch.tensor([[[[1.0]], [[0.0]]]], dtype=torch.float32)
        zero_safe = common.per_pixel_kl(one_hot_teacher, student.log())
        self.assertTrue(torch.isfinite(zero_safe).all())
        self.assertAlmostEqual(float(zero_safe), float(-np.log(0.4)), places=6)

    def test_freeze_mu_uses_median_and_clamp(self):
        result = common.freeze_mu([2.0, 4.0, 8.0], [4.0, 4.0, 4.0], target_grad_ratio=0.10)
        self.assertAlmostEqual(result["median_main_over_rest"], 1.0)
        self.assertAlmostEqual(result["mu"], 0.10)
        low = common.freeze_mu([0.01], [10.0], target_grad_ratio=0.10)
        self.assertEqual(low["mu"], 0.02)
        self.assertTrue(low["clamped"])
        high = common.freeze_mu([100.0], [1.0], target_grad_ratio=0.10)
        self.assertEqual(high["mu"], 0.15)
        self.assertTrue(high["clamped"])

    def test_receipt_binding_rejects_mismatch(self):
        receipt = {
            "formula_version": common.FORMULA_VERSION,
            "black_region_training_exclusion": False,
            "full_teacher_grid_partition": True,
            "a05_sha256": "a05",
            "teacher_cache_content_manifest_sha256": "cache",
            "tca_common_sha256": "common",
            "mu": 0.08,
        }
        receipt["receipt_payload_sha256"] = common.canonical_json_sha256(receipt)
        self.assertEqual(common.validate_calibration_receipt(receipt, "a05", "cache", "common"), 0.08)
        broken = dict(receipt)
        broken["teacher_cache_content_manifest_sha256"] = "other"
        broken.pop("receipt_payload_sha256")
        broken["receipt_payload_sha256"] = common.canonical_json_sha256(broken)
        with self.assertRaises(RuntimeError):
            common.validate_calibration_receipt(broken, "a05", "cache", "common")

        old_v3 = dict(receipt)
        old_v3["formula_version"] = (
            "tca-v3-xlogy-zero-safe-black-core-excluded-selected-plus-gradient-calibrated-complement"
        )
        old_v3.pop("receipt_payload_sha256")
        old_v3["receipt_payload_sha256"] = common.canonical_json_sha256(old_v3)
        with self.assertRaises(RuntimeError):
            common.validate_calibration_receipt(old_v3, "a05", "cache", "common")

    def _png_payload(self, value: int = 1, mode: str = "L", size=(1024, 1024)) -> bytes:
        array = np.full((size[1], size[0]), value, dtype=np.uint8)
        image = Image.fromarray(array, mode=mode)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def test_zip_audit_accepts_exact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.zip"
            payload = self._png_payload()
            with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
                for name in common.exact_test2_names():
                    archive.writestr(name, payload)
            result = package_tca.audit_zip(path)
            self.assertEqual(result["count"], 1300)
            self.assertTrue(result["all_png_contracts_pass"])

    def test_zip_audit_rejects_lexicographic_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.zip"
            payload = self._png_payload()
            with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
                for name in sorted(common.exact_test2_names()):
                    archive.writestr(name, payload)
            with self.assertRaises(RuntimeError):
                package_tca.audit_zip(path)

    def test_training_source_contains_no_source_dataset_branch(self):
        source = Path(__file__).with_name("train_tca.py").read_text(encoding="utf-8")
        forbidden = ["from src.dataset", "RareClassCropper", "cross_entropy("]
        for token in forbidden:
            self.assertNotIn(token, source)

    def test_v4_uses_full_teacher_grid_without_black_exclusion(self):
        root = Path(__file__).parent
        for name in ("tca_common.py", "calibrate_tca_mu.py", "train_tca.py"):
            source = (root / name).read_text(encoding="utf-8")
            for token in ("black_core", "valid_nonblack", "valid_teacher_masks", "scipy"):
                self.assertNotIn(token, source, f"{token} leaked into {name}")
        train_source = (root / "train_tca.py").read_text(encoding="utf-8")
        calibration_source = (root / "calibrate_tca_mu.py").read_text(encoding="utf-8")
        package_source = (root / "package_tca.py").read_text(encoding="utf-8")
        for source in (train_source, calibration_source, package_source):
            self.assertIn('"black_region_training_exclusion": False', source)
            self.assertIn('"full_teacher_grid_partition": True', source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
