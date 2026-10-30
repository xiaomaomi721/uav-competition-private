"""Reusable training components for the UAV semantic-segmentation project."""

from .label_mapping import IGNORE_INDEX, NUM_CLASSES, official_to_train, train_to_official

__all__ = [
    "IGNORE_INDEX",
    "NUM_CLASSES",
    "official_to_train",
    "train_to_official",
]
