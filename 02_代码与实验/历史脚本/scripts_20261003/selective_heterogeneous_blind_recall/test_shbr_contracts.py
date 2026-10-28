#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

import shbr_common as common


class ShbrContracts(unittest.TestCase):
    def test_numeric_test2_names(self):
        names = common.exact_test2_names()
        self.assertEqual(len(names), 1300)
        self.assertEqual(names[:3], ["test2_1.png", "test2_2.png", "test2_3.png"])
        self.assertEqual(names[-1], "test2_1300.png")
        self.assertNotEqual(names, sorted(names))

    def test_d4_roundtrip(self):
        value = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
        views = set()
        for view in range(8):
            transformed = common.d4_forward(value, view)
            restored = common.d4_inverse(transformed, view)
            self.assertTrue(torch.equal(restored, value))
            views.add((tuple(transformed.shape), transformed.numpy().tobytes()))
        self.assertEqual(len(views), 8)

    def test_explicit_gt_mapping_preserves_ignore(self):
        train = np.asarray([[0, 7, 255]], dtype=np.uint8)
        self.assertEqual(common.canonical_gt_to_train(train, "train_ids_0_7_ignore255").tolist(), [[0, 7, 255]])
        official = np.asarray([[1, 8, 255]], dtype=np.uint8)
        self.assertEqual(
            common.canonical_gt_to_train(official, "official_ids_1_8_ignore255").tolist(), [[0, 7, 255]]
        )
        with self.assertRaises(ValueError):
            common.canonical_gt_to_train(official, "train_ids_0_7_ignore255")
        with self.assertRaises(ValueError):
            common.canonical_gt_to_train(train, "guess")

    def test_teacher_mask_thresholds(self):
        probability = torch.full((1, 8, 1, 4), 0.0)
        probability[0, 1, 0, 0] = 0.98
        probability[0, 0, 0, 0] = 0.02
        probability[0, :, 0, 1] = 0.50 / 7
        probability[0, 3, 0, 1] = 0.50
        probability[0, 2, 0, 2] = 0.979
        probability[0, 0, 0, 2] = 0.021
        probability[0, :, 0, 3] = 0.501 / 7
        probability[0, 4, 0, 3] = 0.499
        _, prediction, selected = common.teacher_selected(probability)
        self.assertEqual(prediction.flatten().tolist(), [1, 3, 2, 4])
        self.assertEqual(selected.flatten().tolist(), [True, True, False, False])

    def test_zero_safe_kl(self):
        teacher = torch.tensor([[[[1.0]], [[0.0]]]])
        student = torch.tensor([[[[0.4]], [[0.6]]]])
        actual = common.per_pixel_kl(teacher, student.log())
        self.assertTrue(torch.isfinite(actual).all())
        self.assertAlmostEqual(float(actual), float(-np.log(0.4)), places=6)

    def test_loss_uses_M_denominator_and_only_treat_gets_donor(self):
        source = torch.tensor(2.0)
        a05 = torch.tensor([[1.0, 3.0, 99.0, 99.0]])
        donor = torch.tensor([[99.0, 99.0, 4.0, 8.0]])
        selected = torch.tensor([[True, True, False, False]])
        gate = torch.tensor([[False, False, True, True]])
        control = common.shbr_losses(source, a05, selected, donor, gate, 1.0, "control")
        treat = common.shbr_losses(source, a05, selected, donor, gate, 1.0, "treat")
        self.assertAlmostEqual(float(control["main"]), 2.0)
        self.assertAlmostEqual(float(control["donor"]), 6.0)  # sum(G)=12 divided by |M|=2
        self.assertAlmostEqual(float(control["total"]), 2.2)
        self.assertAlmostEqual(float(treat["total"]), 3.7)
        self.assertEqual(float(control["donor_coefficient"]), 0.0)
        self.assertEqual(float(treat["donor_coefficient"]), 0.25)
        with self.assertRaises(RuntimeError):
            common.shbr_losses(source, a05, selected, donor, selected, 1.0, "treat")

    def test_four_connected_component_area(self):
        mask = np.asarray(
            [
                [1, 1, 0, 1],
                [1, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=bool,
        )
        areas = common.component_area_map(mask)
        self.assertEqual(areas.tolist(), [[3, 3, 0, 2], [3, 0, 0, 2], [0, 0, 1, 0]])

    def test_gate_is_selective_and_disjoint(self):
        shape = (4, 4)
        a05 = np.zeros(shape, np.uint8)
        armb = np.zeros(shape, np.uint8)
        donor = np.full(shape, 3, np.uint8)
        selected = np.zeros(shape, bool)
        margin = np.full(shape, 0.8, np.float32)
        agreement = np.full(shape, 8, np.uint8)
        selected[0, 0] = True
        rule = common.GateRule(0, 3, 7, 0.7, 2)
        gate = common.apply_gate_rule(rule, a05, armb, donor, selected, margin, agreement)
        self.assertFalse(gate[0, 0])
        self.assertEqual(int(gate.sum()), 15)
        combined, _ = common.apply_gate_rules(
            [rule],
            a05_prediction=a05,
            armb_prediction=armb,
            donor_prediction=donor,
            a05_selected=selected,
            donor_margin=margin,
            donor_agreement=agreement,
        )
        self.assertFalse(np.any(combined & selected))

    def test_replacement_metrics_and_bootstrap(self):
        gt = np.asarray([[3, 3], [0, 0]], dtype=np.uint8)
        base = np.asarray([[0, 0], [0, 0]], dtype=np.uint8)
        donor = np.asarray([[3, 3], [3, 3]], dtype=np.uint8)
        gate = np.asarray([[1, 1], [0, 0]], dtype=bool)
        records = [(gt, base, donor, gate) for _ in range(4)]
        metrics = common.replacement_metrics(records)
        self.assertEqual(metrics["gain_pixels"], 8)
        self.assertEqual(metrics["damage_pixels"], 0)
        self.assertGreater(metrics["delta_miou"], 0)
        first = common.bootstrap_delta_miou(records, repetitions=20, seed=7)
        second = common.bootstrap_delta_miou(records, repetitions=20, seed=7)
        self.assertEqual(first, second)

    def test_no_bstar_or_upload_in_donor_and_runner_contracts(self):
        root = Path(__file__).parent
        donor_source = (root / "donor_adapter.py").read_text(encoding="utf-8")
        self.assertIn("bstar_applied\": False", donor_source)
        self.assertNotIn("common.BSTAR", donor_source)
        for script in root.glob("*.py"):
            if script.name.startswith("test_"):
                continue
            source = script.read_text(encoding="utf-8").lower()
            self.assertNotIn("platform/upload", source)
            self.assertNotIn("subprocess.run(['kill'", source)

    def test_development_is_frozen_before_check_load(self):
        source = Path(__file__).with_name("discover_and_validate_gate.py").read_text(encoding="utf-8")
        frozen_write = source.index('"FROZEN_GATE.json"')
        check_load = source.index("r1099 = load_records")
        self.assertLess(frozen_write, check_load)

    def test_formula_constants_are_frozen(self):
        self.assertEqual(common.BETA, 0.25)
        self.assertEqual(common.BLIND_CLASSES, (3, 4, 6))
        self.assertEqual(common.MAX_STEPS, 2000)
        self.assertEqual(common.RAMP_STEPS, 500)
        self.assertEqual(common.EXPECTED_CONTROL_ZIP_SHA, "1700504fbc9659bbb00c386a63c5624902fbf0f45ba1cce09bd8f2727be648b4")
        config = common.load_json(Path(__file__).with_name("frozen_config.json"))
        self.assertEqual(config["allowed_physical_gpus"], [2, 3])


if __name__ == "__main__":
    unittest.main(verbosity=2)
