#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.transforms import apply_geometric_ops, resize_pair  # noqa: E402
from src.rare_crop import RareClassCropper  # noqa: E402
from src.transforms import TrainTransform  # noqa: E402
import random


class TransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.zeros((8, 8), dtype=np.uint8)
        self.mask[1:4, 2:6] = 3
        self.mask[6:, 6:] = 255
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image[self.mask == 3, 0] = 255

    def assert_region_aligned(self, image: np.ndarray, mask: np.ndarray) -> None:
        np.testing.assert_array_equal(image[..., 0] == 255, mask == 3)

    def test_flip_rotation_and_crop_are_synchronized(self) -> None:
        for kwargs in (
            {"horizontal_flip": True},
            {"vertical_flip": True},
            {"rotate_k": 1},
            {"horizontal_flip": True, "vertical_flip": True, "rotate_k": 3},
        ):
            image, mask = apply_geometric_ops(self.image, self.mask, **kwargs)
            self.assert_region_aligned(image, mask)
        image, mask = apply_geometric_ops(self.image, self.mask, crop_box=(1, 2, 4, 4))
        self.assertEqual(image.shape[:2], (4, 4))
        self.assert_region_aligned(image, mask)

    def test_resize_uses_nearest_for_mask(self) -> None:
        image, mask = resize_pair(self.image, self.mask, (13, 11))
        self.assertEqual(image.shape, (11, 13, 3))
        self.assertEqual(mask.shape, (11, 13))
        self.assertTrue(set(int(x) for x in np.unique(mask)).issubset({0, 3, 255}))

    def test_target_aware_crop_survives_following_geometry(self) -> None:
        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        mask[30:35, 900:905] = 7
        transform = TrainTransform(
            crop_size=512,
            config={
                "horizontal_flip_probability": 1.0,
                "vertical_flip_probability": 1.0,
                "rotate90_probability": 1.0,
                "color_jitter": {"probability": 0.0},
                "domain_aug": {"enabled": False},
            },
            rare_cropper=RareClassCropper(512, rare_probability=1.0),
        )
        _, transformed_mask = transform(image, mask, rng=random.Random(42), force_rare=True)
        self.assertTrue(bool(np.any(transformed_mask.numpy() == 7)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
