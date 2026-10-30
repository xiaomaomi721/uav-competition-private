#!/usr/bin/env python3
"""Fail-fast orchestrator for prechecks and the seven-step quick matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, save_json, utc_now  # noqa: E402

EXPERIMENTS = [
    (1, "step01_baseline_512_ce.yaml"),
    (3, "step03_resolution_512.yaml"),
    (3, "step03_resolution_768.yaml"),
    (3, "step03_resolution_1024.yaml"),
    (4, "step04_weighted_ce.yaml"),
    (4, "step04_ce_dice.yaml"),
    (4, "step04_focal.yaml"),
    (5, "step05_rare_crop.yaml"),
    (6, "step06_domain_aug.yaml"),
    (7, "step07_segformer_b1.yaml"),
    (7, "step07_segformer_b2.yaml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "quick_7step")
    parser.add_argument("--only-step", type=int, action="append", choices=[1, 3, 4, 5, 6, 7])
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_completed(directory: Path) -> bool:
    required = [
        "config.yaml", "environment.json", "train_log.csv", "metrics.json", "per_class_iou.csv",
        "confusion_matrix.npy", "best_miou.pt", "last.pt", "completion.json",
    ]
    return all((directory / name).is_file() for name in required) and any(
        (directory / "visualizations" / "predictions").glob("*.png")
    )


def run(command: list[str], *, dry_run: bool) -> int:
    print("COMMAND:", subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    prechecks = [
        ("environment", [python, str(PROJECT_ROOT / "scripts" / "check_env.py"), "--output", str(output_dir / "environment_check.json")] + (["--require-cuda"] if args.device.startswith("cuda") else [])),
        ("create_or_preserve_split", [python, str(PROJECT_ROOT / "scripts" / "create_splits.py"), "--dataset-root", str(dataset_root)]),
        ("split", [python, str(PROJECT_ROOT / "scripts" / "check_split.py"), "--dataset-root", str(dataset_root)]),
        ("label_mapping", [python, str(PROJECT_ROOT / "scripts" / "test_label_mapping.py")]),
        ("transforms", [python, str(PROJECT_ROOT / "scripts" / "test_transforms.py")]),
        ("metrics", [python, str(PROJECT_ROOT / "scripts" / "test_metrics.py")]),
    ]
    precheck_results: list[dict[str, Any]] = []
    for name, command in prechecks:
        code = run(command, dry_run=args.dry_run)
        precheck_results.append({"name": name, "returncode": code, "passed": code == 0})
        if code != 0:
            if not args.dry_run:
                save_json(output_dir / "prechecks.json", {"generated_at_utc": utc_now(), "checks": precheck_results})
            print(f"fatal precheck failed: {name}", file=sys.stderr)
            return code
    if not args.dry_run:
        save_json(output_dir / "prechecks.json", {"generated_at_utc": utc_now(), "checks": precheck_results})

    selected_steps = set(args.only_step or [1, 3, 4, 5, 6, 7])
    selected = [(step, name) for step, name in EXPERIMENTS if step in selected_steps]
    if selected_steps & {4, 5, 6, 7} and not any(name == "step03_resolution_512.yaml" for _, name in selected):
        selected.insert(0, (3, "step03_resolution_512.yaml"))

    for step, config_name in selected:
        config_path = PROJECT_ROOT / "configs" / "quick" / config_name
        config = load_config(config_path)
        experiment_dir = output_dir / "experiments" / str(config["experiment_id"])
        if args.skip_completed and is_completed(experiment_dir):
            print(f"SKIP completed: {config['experiment_id']}")
            continue
        command = [
            python,
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config",
            str(config_path),
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(experiment_dir),
            "--device",
            args.device,
        ]
        code = run(command, dry_run=args.dry_run)
        if code != 0 and step == 1:
            print("Step 1 failed; stopping the remaining experiment matrix.", file=sys.stderr)
            break

    summary_command = [python, str(PROJECT_ROOT / "scripts" / "summarize_experiments.py"), "--output-dir", str(output_dir)]
    summary_code = run(summary_command, dry_run=args.dry_run)
    return summary_code


if __name__ == "__main__":
    raise SystemExit(main())
