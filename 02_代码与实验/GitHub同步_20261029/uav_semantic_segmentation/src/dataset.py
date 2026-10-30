"""Dataset loading with read-only official files and in-memory label mapping."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_mapping import official_to_train
from .rare_crop import RareClassCropper
from .transforms import TrainTransform, ValidationTransform
from .utils import read_stems, save_triptych


def deterministic_subset(stems: list[str], maximum: int | None, seed: int) -> list[str]:
    if maximum is None or maximum <= 0 or maximum >= len(stems):
        return list(stems)
    selected = list(stems)
    random.Random(seed).shuffle(selected)
    return selected[:maximum]


class UAVSegmentationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_root: Path,
        stems: list[str] | Path,
        transform: TrainTransform | ValidationTransform,
        *,
        maximum: int | None = None,
        seed: int = 42,
        random_subset: bool = True,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.image_dir = self.dataset_root / "train" / "images"
        self.mask_dir = self.dataset_root / "train" / "masks"
        source_stems = read_stems(stems) if isinstance(stems, Path) else list(stems)
        self.stems = deterministic_subset(source_stems, maximum, seed) if random_subset else (
            source_stems[:maximum] if maximum and maximum > 0 else source_stems
        )
        self.transform = transform
        self.seed = int(seed)
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"expected train/images and train/masks below {self.dataset_root}")

    def __len__(self) -> int:
        return len(self.stems)

    def load_raw(self, stem: str) -> tuple[np.ndarray, np.ndarray]:
        image_path = self.image_dir / f"{stem}.png"
        mask_path = self.mask_dir / f"{stem}.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"missing image/mask pair for stem {stem}")
        with Image.open(image_path) as handle:
            image = np.asarray(handle.convert("RGB")).copy()
        with Image.open(mask_path) as handle:
            official_mask = np.asarray(handle).copy()
        if official_mask.ndim != 2:
            raise ValueError(f"mask must be single-channel: {mask_path}")
        return image, official_to_train(official_mask)

    def __getitem__(self, index: int) -> dict[str, Any]:
        stem = self.stems[index]
        image, mask = self.load_raw(stem)
        pixel_values, labels = self.transform(image, mask)
        return {"pixel_values": pixel_values, "labels": labels, "stem": stem}

    def save_training_previews(
        self,
        output_dir: Path,
        *,
        count: int = 20,
        force_rare: bool = False,
        force_domain: bool = False,
    ) -> list[str]:
        if not isinstance(self.transform, TrainTransform):
            return []
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[str] = []
        if force_rare:
            for stem in self.stems:
                _, mask = self.load_raw(stem)
                if np.any(mask == 7) or np.any(mask == 4):
                    candidates.append(stem)
                    if len(candidates) >= count:
                        break
        if len(candidates) < count:
            for stem in self.stems:
                if stem not in candidates:
                    candidates.append(stem)
                    if len(candidates) >= count:
                        break
        paths: list[str] = []
        for index, stem in enumerate(candidates[:count]):
            image, mask = self.load_raw(stem)
            tensor, transformed_mask = self.transform(
                image,
                mask,
                rng=random.Random(self.seed + 100_000 + index),
                force_rare=True if force_rare else None,
                force_domain=True if force_domain else None,
            )
            from .utils import denormalize_image

            path = output_dir / f"augmentation_{index + 1:02d}_{stem}.png"
            save_triptych(path, denormalize_image(tensor), transformed_mask.numpy())
            paths.append(str(path))
        self.transform.rare_cropper.reset_statistics()
        return paths


def build_datasets(
    config: dict[str, Any],
    dataset_root: Path,
    project_root: Path,
) -> tuple[UAVSegmentationDataset, UAVSegmentationDataset, RareClassCropper]:
    seed = int(config.get("seed", 42))
    crop_size = int(config["crop_size"])
    rare_cfg = config.get("rare_crop", {})
    cropper = RareClassCropper(
        crop_size=crop_size,
        rare_probability=float(rare_cfg.get("probability", 0.30)) if rare_cfg.get("enabled", False) else 0.0,
        priority_ids=tuple(int(x) for x in rare_cfg.get("priority_train_ids", [7, 4, 6])),
        max_offset_fraction=float(rare_cfg.get("max_offset_fraction", 0.25)),
    )
    train_transform = TrainTransform(
        crop_size=crop_size,
        config=config.get("transforms", {}),
        rare_cropper=cropper,
        ignore_index=int(config.get("ignore_index", 255)),
    )
    train_split = project_root / str(config.get("train_split", "splits/train.txt"))
    val_split = project_root / str(config.get("validation_split", "splits/fixed_quick_val.txt"))
    train_dataset = UAVSegmentationDataset(
        dataset_root,
        train_split,
        train_transform,
        maximum=int(config.get("max_train_samples", 0)) or None,
        seed=seed,
        random_subset=True,
    )
    val_dataset = UAVSegmentationDataset(
        dataset_root,
        val_split,
        ValidationTransform(),
        maximum=int(config.get("max_val_samples", 0)) or None,
        seed=seed,
        random_subset=False,
    )
    return train_dataset, val_dataset, cropper
