#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import SegmentationMetrics  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_ignore_confusion_iou_miou_and_accuracy(self) -> None:
        target = np.array([[0, 1, 255], [1, 2, 2]], dtype=np.uint8)
        prediction = np.array([[0, 0, 7], [1, 2, 0]], dtype=np.uint8)
        meter = SegmentationMetrics(num_classes=8, ignore_index=255)
        meter.update(prediction, target)
        result = meter.compute()
        self.assertEqual(result["valid_pixel_count"], 5)
        self.assertAlmostEqual(result["pixel_accuracy"], 3 / 5)
        self.assertAlmostEqual(result["per_class_iou"][0], 1 / 3)
        self.assertAlmostEqual(result["per_class_iou"][1], 1 / 2)
        self.assertAlmostEqual(result["per_class_iou"][2], 1 / 2)
        self.assertAlmostEqual(result["mIoU"], (1 / 3 + 1 / 2 + 1 / 2) / 3)
        self.assertAlmostEqual(result["per_class_precision"][0], 1 / 3)
        self.assertAlmostEqual(result["per_class_recall"][1], 1 / 2)
        self.assertEqual(result["per_class_false_positive_pixels"][0], 2)
        self.assertEqual(result["absent_class_ids"], [3, 4, 5, 6, 7])
        expected = np.zeros((8, 8), dtype=np.int64)
        expected[0, 0] = 1
        expected[1, 0] = 1
        expected[1, 1] = 1
        expected[2, 0] = 1
        expected[2, 2] = 1
        np.testing.assert_array_equal(meter.confusion_matrix, expected)

    def test_invalid_non_ignore_target_is_rejected(self) -> None:
        meter = SegmentationMetrics()
        with self.assertRaises(ValueError):
            meter.update(np.array([0]), np.array([9]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
