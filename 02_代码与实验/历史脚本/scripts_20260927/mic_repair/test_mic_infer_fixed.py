#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import inspect
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("mic_infer_fixed.py")
SPEC = importlib.util.spec_from_file_location("mic_infer_fixed", MODULE_PATH)
MIC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MIC)


class MicRepairContracts(unittest.TestCase):
    def test_exact_names_are_numeric(self):
        names = MIC.exact_names()
        self.assertEqual(len(names), 1300)
        self.assertEqual(names[:4], ["test2_1.png", "test2_2.png", "test2_3.png", "test2_4.png"])
        self.assertEqual(names[-1], "test2_1300.png")

    def test_legacy_lexicographic_bug_misbinds_1299(self):
        numeric = MIC.exact_names()
        lexical = sorted(numeric)
        fixed_points = sum(left == right for left, right in zip(numeric, lexical))
        self.assertEqual(fixed_points, 1)
        self.assertEqual(lexical[:4], ["test2_1.png", "test2_10.png", "test2_100.png", "test2_1000.png"])

    def test_input_set_requires_exact_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in MIC.exact_names():
                (root / name).touch()
            self.assertEqual(MIC.validate_input_names(root), MIC.exact_names())
            (root / "test2_777.png").unlink()
            with self.assertRaises(RuntimeError):
                MIC.validate_input_names(root)

    def test_d4_contract_is_rotate_then_flip_inverse_flip_then_rotate(self):
        self.assertEqual(MIC.VIEWS, [(rotation, horizontal_flip) for rotation in range(4) for horizontal_flip in (False, True)])
        forward_source = inspect.getsource(MIC.d4_forward)
        inverse_source = inspect.getsource(MIC.d4_inverse)
        self.assertLess(forward_source.index("tensor.rot90"), forward_source.index("tensor.flip"))
        self.assertLess(inverse_source.index("tensor.flip"), inverse_source.index("tensor.rot90"))

    def test_numpy_d4_roundtrip_non_square_and_unique(self):
        tensor = np.arange(2 * 3 * 5 * 7, dtype=np.int64).reshape(2, 3, 5, 7)
        seen = set()
        for rotation, horizontal_flip in MIC.VIEWS:
            transformed = np.rot90(tensor, rotation, axes=(-2, -1)) if rotation else tensor
            if horizontal_flip:
                transformed = np.flip(transformed, axis=-1)
            restored = np.flip(transformed, axis=-1) if horizontal_flip else transformed
            if rotation:
                restored = np.rot90(restored, -rotation, axes=(-2, -1))
            self.assertTrue(np.array_equal(restored, tensor))
            key = (transformed.shape, transformed.tobytes())
            self.assertNotIn(key, seen)
            seen.add(key)
        self.assertEqual(len(seen), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
