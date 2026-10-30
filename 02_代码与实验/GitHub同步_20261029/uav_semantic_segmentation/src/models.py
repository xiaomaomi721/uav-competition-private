"""SegFormer model construction and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSemanticSegmentation

from .utils import CLASS_NAMES


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_name = str(config["model_name"])
    num_classes = int(config.get("num_classes", 8))
    id2label = {index: CLASS_NAMES[index] for index in range(num_classes)}
    label2id = {name: index for index, name in id2label.items()}
    model = AutoModelForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    if bool(config.get("gradient_checkpointing", False)):
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError(f"model {model_name} does not support gradient checkpointing")
        model.gradient_checkpointing_enable()
    return model


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}
