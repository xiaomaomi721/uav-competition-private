#!/usr/bin/env python3
"""Evaluate a checkpoint on the fixed validation subset without random augmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import UAVSegmentationDataset  # noqa: E402
from src.inference import evaluate_model  # noqa: E402
from src.models import build_model, load_model_checkpoint  # noqa: E402
from src.transforms import ValidationTransform  # noqa: E402
from src.utils import load_config, resolve_device  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str)
    parser.add_argument("--max-val-samples", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    maximum = args.max_val_samples if args.max_val_samples is not None else int(config.get("max_val_samples", 0))
    dataset = UAVSegmentationDataset(
        args.dataset_root.resolve(),
        PROJECT_ROOT / str(config.get("validation_split", "splits/fixed_quick_val.txt")),
        ValidationTransform(),
        maximum=maximum or None,
        seed=int(config.get("seed", 42)),
        random_subset=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(config.get("num_workers", 0)))
    device = resolve_device(args.device or str(config.get("device", "cuda")))
    model = build_model(config).to(device)
    load_model_checkpoint(model, args.checkpoint.resolve(), device)
    metrics = evaluate_model(
        model,
        loader,
        device,
        amp=bool(config.get("amp", True)),
        output_dir=args.output_dir.resolve(),
        save_artifacts=True,
        visualization_limit=int(config.get("prediction_visualization_count", 8)),
    )
    print(f"mIoU={metrics['mIoU']:.6f} pixel_accuracy={metrics['pixel_accuracy']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
