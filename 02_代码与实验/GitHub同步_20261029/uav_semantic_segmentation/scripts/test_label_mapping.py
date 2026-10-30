#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.label_mapping import official_to_train, train_to_official  # noqa: E402


class LabelMappingTests(unittest.TestCase):
    def test_complete_mapping_dtype_and_round_trip(self) -> None:
        official = np.arange(9, dtype=np.uint8).reshape(3, 3)
        training = official_to_train(official)
        self.assertEqual(training.dtype, np.uint8)
        self.assertEqual(int(training[0, 0]), 255)
        np.testing.assert_array_equal(training.reshape(-1)[1:], np.arange(8, dtype=np.uint8))
        restored = train_to_official(training, allow_ignore=True)
        self.assertEqual(restored.dtype, np.uint8)
        np.testing.assert_array_equal(restored, official)
        np.testing.assert_array_equal(train_to_official(np.arange(8, dtype=np.uint8)), np.arange(1, 9, dtype=np.uint8))

    def test_input_is_not_modified(self) -> None:
        official = np.array([[0, 1, 8]], dtype=np.uint8)
        original = official.copy()
        official_to_train(official)
        np.testing.assert_array_equal(official, original)

    def test_illegal_values_and_dtype_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            official_to_train(np.array([9], dtype=np.uint8))
        with self.assertRaises(ValueError):
            train_to_official(np.array([8], dtype=np.uint8))
        with self.assertRaises(TypeError):
            official_to_train(np.array([0, 1], dtype=np.int64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
