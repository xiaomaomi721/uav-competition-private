#!/usr/bin/env python3
"""Train any experiment through one resolved YAML configuration."""

from __future__ import annotations

import argparse
import traceback
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainer import run_training_with_oom_retry  # noqa: E402
from src.utils import load_config, resolve_device, save_json, utc_now  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    for cli_name, config_name in (
        ("max_steps", "max_steps"),
        ("max_train_samples", "max_train_samples"),
        ("max_val_samples", "max_val_samples"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            config[config_name] = value
    requested_device = args.device or str(config.get("device", "cuda"))
    device = resolve_device(requested_device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        metrics = run_training_with_oom_retry(
            config,
            args.dataset_root.resolve(),
            output_dir,
            PROJECT_ROOT,
            device,
            resume=args.resume.resolve() if args.resume else None,
        )
        print(f"completed {config['experiment_id']}: mIoU={metrics['mIoU']:.6f}")
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (output_dir / "error.log").write_text(trace, encoding="utf-8")
        save_json(
            output_dir / "failed.json",
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "experiment_id": config.get("experiment_id"),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        print(trace, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
