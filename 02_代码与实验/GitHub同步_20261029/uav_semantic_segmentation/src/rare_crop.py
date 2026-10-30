"""Rare-class-aware crop selection that never modifies source masks."""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CropStatistics:
    total_crops: int = 0
    random_crops: int = 0
    rare_crops: int = 0
    patches_with_vehicle: int = 0
    patches_with_barren: int = 0
    patches_with_agricultural: int = 0

    def to_dict(self) -> dict[str, int | float]:
        result: dict[str, int | float] = asdict(self)
        denominator = max(self.total_crops, 1)
        result.update(
            {
                "vehicle_patch_ratio": self.patches_with_vehicle / denominator,
                "barren_patch_ratio": self.patches_with_barren / denominator,
                "agricultural_patch_ratio": self.patches_with_agricultural / denominator,
            }
        )
        return result


class RareClassCropper:
    """Choose ordinary or target-aware square crops in training-label space."""

    def __init__(
        self,
        crop_size: int,
        rare_probability: float = 0.30,
        priority_ids: tuple[int, ...] = (7, 4, 6),
        max_offset_fraction: float = 0.25,
    ) -> None:
        self.crop_size = int(crop_size)
        self.rare_probability = float(rare_probability)
        self.priority_ids = tuple(int(x) for x in priority_ids)
        self.max_offset_fraction = float(max_offset_fraction)
        self.statistics = CropStatistics()
        self.last_mode = "uninitialized"
        self.last_target_id: int | None = None

    def reset_statistics(self) -> None:
        self.statistics = CropStatistics()

    def _random_origin(self, height: int, width: int, rng: random.Random) -> tuple[int, int]:
        max_y = max(0, height - self.crop_size)
        max_x = max(0, width - self.crop_size)
        return rng.randint(0, max_y) if max_y else 0, rng.randint(0, max_x) if max_x else 0

    def choose(self, mask: np.ndarray, rng: random.Random, *, force_rare: bool | None = None) -> tuple[int, int]:
        height, width = mask.shape[-2:]
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f"mask {width}x{height} is smaller than crop {self.crop_size}")
        use_rare = force_rare if force_rare is not None else rng.random() < self.rare_probability
        selected_id: int | None = None
        if use_rare:
            for class_id in self.priority_ids:
                if np.any(mask == class_id):
                    selected_id = class_id
                    break
        if selected_id is None:
            top, left = self._random_origin(height, width, rng)
            mode = "random"
        else:
            flat = np.flatnonzero(mask == selected_id)
            selected_flat = int(flat[rng.randrange(len(flat))])
            center_y, center_x = divmod(selected_flat, width)
            offset = int(round(self.crop_size * self.max_offset_fraction))
            center_y += rng.randint(-offset, offset)
            center_x += rng.randint(-offset, offset)
            top = min(max(center_y - self.crop_size // 2, 0), height - self.crop_size)
            left = min(max(center_x - self.crop_size // 2, 0), width - self.crop_size)
            mode = "rare"

        patch = mask[top : top + self.crop_size, left : left + self.crop_size]
        self.statistics.total_crops += 1
        if mode == "rare":
            self.statistics.rare_crops += 1
        else:
            self.statistics.random_crops += 1
        self.statistics.patches_with_vehicle += int(np.any(patch == 7))
        self.statistics.patches_with_barren += int(np.any(patch == 4))
        self.statistics.patches_with_agricultural += int(np.any(patch == 6))
        self.last_mode = mode
        self.last_target_id = selected_id
        return int(top), int(left)
