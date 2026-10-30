"""Small shared utilities; all filesystem paths are handled with pathlib."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLASS_NAMES = ["Background", "Building", "Road", "Water", "Barren", "Vegetation", "Agricultural", "Vehicle"]
PALETTE = np.asarray(
    [
        [128, 128, 128],
        [220, 20, 60],
        [255, 215, 0],
        [30, 144, 255],
        [210, 180, 140],
        [34, 139, 34],
        [154, 205, 50],
        [255, 105, 180],
    ],
    dtype=np.uint8,
)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", value) or re.fullmatch(
        r"[-+]?\d+[eE][-+]?\d+", value
    ):
        return float(value)
    return value.strip("'\"")


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if ":" not in stripped:
            raise ValueError(f"unsupported YAML line {line_number}: {raw_line}")
        key, value = stripped.split(":", 1)
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_scalar(value)
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = _load_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    config = load_yaml(path)
    base_ref = config.pop("_base_", None)
    if base_ref is not None:
        base_path = (path.parent / str(base_ref)).resolve()
        config = deep_merge(load_config(base_path), config)
    config["source_config"] = str(path)
    return config


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def resolve_device(requested: str) -> torch.device:
    import torch

    requested = str(requested).lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def read_stems(path: Path) -> list[str]:
    stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(stems) != len(set(stems)):
        raise ValueError(f"duplicate stems in {path}")
    return stems


def environment_info() -> dict[str, Any]:
    import torch

    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "total_memory_mb": int(properties.total_memory / 1024**2),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    try:
        import transformers

        transformers_version: str | None = transformers.__version__
    except Exception:
        transformers_version = None
    return {
        "generated_at_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "transformers": transformers_version,
        "gpu": gpu,
    }


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    array = (array * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    mask = np.asarray(mask)
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(PALETTE))
    result[valid] = PALETTE[mask[valid].astype(np.int64)]
    result[mask == ignore_index] = (0, 0, 0)
    return result


def save_triptych(path: Path, image: np.ndarray, target: np.ndarray, prediction: np.ndarray | None = None) -> None:
    target_color = colorize_mask(target)
    panels = [image, target_color]
    if prediction is not None:
        panels.append(colorize_mask(prediction))
    canvas = np.concatenate(panels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def run_command_version(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except Exception:
        return None
