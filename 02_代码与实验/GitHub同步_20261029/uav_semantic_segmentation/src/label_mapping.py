"""Lossless in-memory mapping between official and training label IDs."""

from __future__ import annotations

import numpy as np

NUM_CLASSES = 8
IGNORE_INDEX = 255
OFFICIAL_IDS = set(range(9))
TRAIN_IDS = set(range(NUM_CLASSES))


def _as_uint8_array(mask: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {array.dtype}")
    if array.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    return array


def official_to_train(mask: np.ndarray) -> np.ndarray:
    """Map official 0..8 to train 255,0..7 without modifying the input."""
    array = _as_uint8_array(mask, "official mask")
    values = set(int(x) for x in np.unique(array))
    illegal = values - OFFICIAL_IDS
    if illegal:
        raise ValueError(f"official mask contains illegal IDs: {sorted(illegal)}")
    mapped = np.full(array.shape, IGNORE_INDEX, dtype=np.uint8)
    valid = array > 0
    mapped[valid] = array[valid] - 1
    return mapped


def train_to_official(mask: np.ndarray, *, allow_ignore: bool = False) -> np.ndarray:
    """Map train 0..7 back to official 1..8.

    Predictions must normally contain only 0..7. ``allow_ignore`` is reserved for
    round-trip tests and diagnostic visualization of targets.
    """
    array = _as_uint8_array(mask, "training mask")
    allowed = TRAIN_IDS | ({IGNORE_INDEX} if allow_ignore else set())
    values = set(int(x) for x in np.unique(array))
    illegal = values - allowed
    if illegal:
        raise ValueError(f"training mask contains illegal IDs: {sorted(illegal)}")
    restored = np.empty(array.shape, dtype=np.uint8)
    valid = array != IGNORE_INDEX
    restored[valid] = array[valid] + 1
    if allow_ignore:
        restored[~valid] = 0
    return restored
