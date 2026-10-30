"""Synchronized image/mask geometry and conservative image-only augmentation."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from .rare_crop import RareClassCropper
from .utils import IMAGENET_MEAN, IMAGENET_STD


def resize_pair(image: np.ndarray, mask: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Resize image bilinearly and mask with nearest-neighbor interpolation."""
    width, height = int(size[0]), int(size[1])
    resized_image = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))
    resized_mask = np.asarray(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST))
    return resized_image.copy(), resized_mask.copy()


def apply_geometric_ops(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    rotate_k: int = 0,
    crop_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic helper used by both the pipeline and synchronization tests."""
    if horizontal_flip:
        image, mask = np.flip(image, axis=1), np.flip(mask, axis=1)
    if vertical_flip:
        image, mask = np.flip(image, axis=0), np.flip(mask, axis=0)
    if rotate_k % 4:
        image, mask = np.rot90(image, k=rotate_k % 4), np.rot90(mask, k=rotate_k % 4)
    if crop_box is not None:
        top, left, height, width = (int(x) for x in crop_box)
        image = image[top : top + height, left : left + width]
        mask = mask[top : top + height, left : left + width]
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def _pad_to_crop(image: np.ndarray, mask: np.ndarray, crop_size: int, ignore_index: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    pad_h = max(0, crop_size - height)
    pad_w = max(0, crop_size - width)
    if not pad_h and not pad_w:
        return image, mask
    image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=ignore_index)
    return image, mask


def _adjust_hue(image: Image.Image, amount: float) -> Image.Image:
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    shift = int(round(amount * 255.0))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _color_jitter(image: np.ndarray, cfg: dict[str, Any], rng: random.Random) -> np.ndarray:
    pil = Image.fromarray(image)
    operations: list[tuple[str, float]] = []
    for name in ("brightness", "contrast", "saturation"):
        magnitude = float(cfg.get(name, 0.0))
        if magnitude > 0:
            operations.append((name, rng.uniform(1.0 - magnitude, 1.0 + magnitude)))
    rng.shuffle(operations)
    for name, factor in operations:
        if name == "brightness":
            pil = ImageEnhance.Brightness(pil).enhance(factor)
        elif name == "contrast":
            pil = ImageEnhance.Contrast(pil).enhance(factor)
        else:
            pil = ImageEnhance.Color(pil).enhance(factor)
    hue = float(cfg.get("hue", 0.0))
    if hue > 0:
        pil = _adjust_hue(pil, rng.uniform(-hue, hue))
    return np.asarray(pil).copy()


def _domain_image_augment(image: np.ndarray, cfg: dict[str, Any], rng: random.Random) -> np.ndarray:
    image = _color_jitter(image, cfg, rng)
    gamma_range = cfg.get("gamma_range", [0.90, 1.10])
    gamma = rng.uniform(float(gamma_range[0]), float(gamma_range[1]))
    normalized = np.asarray(image, dtype=np.float32) / 255.0
    image = np.clip((normalized**gamma) * 255.0, 0, 255).astype(np.uint8)
    if rng.random() < float(cfg.get("blur_probability", 0.10)):
        radius = rng.uniform(0.2, float(cfg.get("max_blur_radius", 1.0)))
        image = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=radius))).copy()
    if rng.random() < float(cfg.get("noise_probability", 0.10)):
        sigma = rng.uniform(1.0, float(cfg.get("max_noise_sigma", 6.0)))
        generator = np.random.default_rng(rng.randrange(2**32))
        noise = generator.normal(0.0, sigma, size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < float(cfg.get("compression_probability", 0.0)):
        quality = rng.randint(int(cfg.get("jpeg_quality_min", 85)), int(cfg.get("jpeg_quality_max", 98)))
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            image = np.asarray(compressed.convert("RGB")).copy()
    return image


def normalize_to_tensor(image: np.ndarray) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float()


@dataclass
class TrainTransform:
    crop_size: int
    config: dict[str, Any]
    rare_cropper: RareClassCropper
    ignore_index: int = 255

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        rng: random.Random | None = None,
        force_rare: bool | None = None,
        force_domain: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rng = rng or random
        domain_cfg = self.config.get("domain_aug", {})
        domain_enabled = bool(domain_cfg.get("enabled", False))
        apply_domain = force_domain if force_domain is not None else (
            domain_enabled and rng.random() < float(domain_cfg.get("probability", 0.50))
        )

        if apply_domain:
            scale_range = domain_cfg.get("scale_range", [0.90, 1.10])
            scale = rng.uniform(float(scale_range[0]), float(scale_range[1]))
            if abs(scale - 1.0) > 1.0e-6:
                height, width = mask.shape
                image, mask = resize_pair(image, mask, (max(1, round(width * scale)), max(1, round(height * scale))))

        image, mask = _pad_to_crop(image, mask, self.crop_size, self.ignore_index)
        top, left = self.rare_cropper.choose(mask, rng, force_rare=force_rare)
        # Crop in the coordinate system used by the cropper. Applying a flip or
        # rotation before this crop would invalidate the target-aware origin.
        image, mask = apply_geometric_ops(
            image,
            mask,
            crop_box=(top, left, self.crop_size, self.crop_size),
        )
        image, mask = apply_geometric_ops(
            image,
            mask,
            horizontal_flip=rng.random() < float(self.config.get("horizontal_flip_probability", 0.5)),
            vertical_flip=rng.random() < float(self.config.get("vertical_flip_probability", 0.5)),
            rotate_k=rng.randrange(4) if rng.random() < float(self.config.get("rotate90_probability", 0.5)) else 0,
        )

        base_color_cfg = self.config.get("color_jitter", {})
        if rng.random() < float(base_color_cfg.get("probability", 0.5)):
            image = _color_jitter(image, base_color_cfg, rng)
        if apply_domain:
            image = _domain_image_augment(image, domain_cfg, rng)

        values = set(int(x) for x in np.unique(mask))
        illegal = values - set(range(8)) - {self.ignore_index}
        if illegal:
            raise ValueError(f"augmentation introduced invalid mask IDs: {sorted(illegal)}")
        return normalize_to_tensor(image), torch.from_numpy(mask.astype(np.int64, copy=True))


class ValidationTransform:
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        return normalize_to_tensor(image), torch.from_numpy(mask.astype(np.int64, copy=True))
