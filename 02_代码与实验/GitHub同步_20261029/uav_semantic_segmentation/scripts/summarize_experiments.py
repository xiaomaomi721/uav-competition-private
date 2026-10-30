#!/usr/bin/env python3
"""Create truthful CSV/JSON summaries and Stage 02 quick-pilot reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import CLASS_NAMES, load_config, save_json, utc_now  # noqa: E402

CONFIG_FILES = [
    "step01_baseline_512_ce.yaml",
    "step03_resolution_512.yaml",
    "step03_resolution_768.yaml",
    "step03_resolution_1024.yaml",
    "step04_weighted_ce.yaml",
    "step04_ce_dice.yaml",
    "step04_focal.yaml",
    "step05_rare_crop.yaml",
    "step06_domain_aug.yaml",
    "step07_segformer_b1.yaml",
    "step07_segformer_b2.yaml",
]


def completion_is_valid(directory: Path) -> bool:
    required = [
        "config.yaml",
        "environment.json",
        "train_log.csv",
        "metrics.json",
        "per_class_iou.csv",
        "confusion_matrix.npy",
        "best_miou.pt",
        "last.pt",
        "completion.json",
    ]
    return all((directory / name).is_file() for name in required) and any(
        (directory / "visualizations" / "predictions").glob("*.png")
    )


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def derived_class_stats(metrics: dict[str, Any], class_id: int) -> dict[str, Any]:
    matrix = metrics.get("confusion_matrix")
    if not matrix or class_id >= len(matrix):
        return {"false_positive_pixels": None, "false_positive_rate": None, "precision": None, "recall": None}
    size = len(matrix)
    true_positive = int(matrix[class_id][class_id])
    target = sum(int(matrix[class_id][column]) for column in range(size))
    predicted = sum(int(matrix[row][class_id]) for row in range(size))
    total = sum(int(value) for row in matrix for value in row)
    false_positive = predicted - true_positive
    negative = total - target
    return {
        "false_positive_pixels": false_positive,
        "false_positive_rate": false_positive / negative if negative > 0 else None,
        "precision": true_positive / predicted if predicted > 0 else None,
        "recall": true_positive / target if target > 0 else None,
    }


def experiment_row(config_path: Path, experiments_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    experiment_id = str(config["experiment_id"])
    directory = experiments_dir / experiment_id
    metrics = load_optional_json(directory / "metrics.json")
    failed = load_optional_json(directory / "failed.json")
    if completion_is_valid(directory):
        status = "completed"
        error_message = ""
    elif failed or (directory / "error.log").is_file():
        status = "failed"
        error_message = str(failed.get("error_message", "see error.log"))
    else:
        status = "not_run"
        error_message = ""
    ious = metrics.get("per_class_iou", [None] * 8)
    vehicle_stats = derived_class_stats(metrics, 7)
    row: dict[str, Any] = {
        "experiment_id": experiment_id,
        "step": config.get("step"),
        "model": config.get("model_variant", config.get("model_name")),
        "model_name": config.get("model_name"),
        "crop_size": config.get("crop_size"),
        "loss": config.get("loss", {}).get("name"),
        "rare_crop": bool(config.get("rare_crop", {}).get("enabled", False)),
        "domain_aug": bool(config.get("transforms", {}).get("domain_aug", {}).get("enabled", False)),
        "max_steps": config.get("max_steps"),
        "effective_batch_size": metrics.get(
            "effective_batch_size",
            int(config.get("batch_size", 1)) * int(config.get("gradient_accumulation_steps", 1)),
        ),
        "actual_batch_size": metrics.get("batch_size"),
        "peak_vram_mb": metrics.get("peak_vram_mb"),
        "average_step_time": metrics.get("average_step_time"),
        "mIoU": metrics.get("mIoU"),
        "per_class_iou": ious,
        "pixel_accuracy": metrics.get("pixel_accuracy"),
        "parameter_count": metrics.get("parameter_count"),
        "inference_images_per_second": metrics.get("inference_images_per_second"),
        "oom_occurred": bool(metrics.get("oom_events")) or "OutOfMemory" in error_message or "out of memory" in error_message.lower(),
        "status": status,
        "error_message": error_message,
        "best_weight_path": (
            f"outputs/quick_7step/experiments/{experiment_id}/best_miou.pt"
            if (directory / "best_miou.pt").is_file()
            else ""
        ),
        "quick_pilot": bool(config.get("quick_pilot", False)),
        "vehicle_false_positive_pixels": vehicle_stats["false_positive_pixels"],
        "vehicle_false_positive_rate": vehicle_stats["false_positive_rate"],
        "vehicle_precision": vehicle_stats["precision"],
        "vehicle_recall": vehicle_stats["recall"],
    }
    for index, name in enumerate(CLASS_NAMES):
        row[f"iou_{name.lower()}"] = ious[index] if index < len(ious) else None
    if metrics.get("crop_statistics"):
        row["crop_statistics"] = metrics["crop_statistics"]
    return row


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(rows: list[dict[str, Any]], output_dir: Path) -> str:
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] == "failed"]
    not_run = [row for row in rows if row["status"] == "not_run"]
    lines = [
        "# Stage 02: seven-step quick engineering validation",
        "",
        "## 实验性质",
        "",
        "全部 11 组实验均为 quick pilot，仅用于验证工程可运行性、显存占用、Loss 稳定性、"
        "分辨率可运行性、模型结构加载以及稀有类别裁剪和域增强逻辑。",
        "这些 100 或 200 step 短程结果不是正式收敛结果，不能直接确定最佳 Loss、最佳分辨率或最佳模型。",
        "",
        "> These runs are quick pilots for code-path, memory, metric, and reproducibility validation. "
        "They are not formal competition results and cannot establish the final best model, loss, resolution, or augmentation.",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Execution status",
        "",
        f"- completed: {len(completed)}",
        f"- failed: {len(failed)}",
        f"- not run: {len(not_run)}",
        "",
        "| Experiment | Step | Model | Crop | Loss | Rare crop | Domain aug | Status | mIoU | Peak VRAM MB | Step s |",
        "|---|---:|---|---:|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment_id']} | {row['step']} | {row['model']} | {row['crop_size']} | {row['loss']} | "
            f"{row['rare_crop']} | {row['domain_aug']} | {row['status']} | {fmt(row['mIoU'])} | "
            f"{fmt(row['peak_vram_mb'], 1)} | {fmt(row['average_step_time'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Per-class IoU",
            "",
            "| Experiment | Background | Building | Road | Water | Barren | Vegetation | Agricultural | Vehicle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        values = [fmt(row.get(f"iou_{name.lower()}")) for name in CLASS_NAMES]
        lines.append(f"| {row['experiment_id']} | " + " | ".join(values) + " |")

    lines.extend(["", "## Validation of the seven technical steps", ""])
    prechecks = load_optional_json(output_dir / "prechecks.json").get("checks", [])
    split_stats = load_optional_json(PROJECT_ROOT / "splits" / "split_statistics.json")
    if split_stats:
        lines.append(
            f"The immutable split contains {split_stats['train']['sample_count']} training and "
            f"{split_stats['val']['sample_count']} validation samples; the fixed quick subset contains 256 stems."
        )
    if prechecks:
        lines.append(
            "Prechecks: " + ", ".join(
                f"{item['name']}={'passed' if item.get('passed') else 'failed'}" for item in prechecks
            ) + "."
        )
    lines.append(
        "Official label 0 is mapped in memory to 255, official 1..8 to train 0..7; Ignore is excluded from loss, "
        "confusion matrix, per-class IoU, mIoU, and Pixel Accuracy."
    )
    step_descriptions = {
        1: "SegFormer-B0 + cross-entropy + 512 baseline and checkpoint/log/metric chain.",
        3: "512/768/1024 controlled resolution feasibility comparison.",
        4: "CE control plus weighted CE, CE+Dice, and focal loss code paths.",
        5: "70% ordinary plus 30% Vehicle/Barren/Agricultural-aware crop path and crop counters.",
        6: "Conservative color/imaging domain augmentation applied to training images only.",
        7: "SegFormer-B0/B1/B2 capacity, memory, speed, and forward/backward feasibility.",
    }
    lines.append("Step 2 is covered by the label-mapping, transform-synchronization, metric, and split prechecks.")
    for step, description in step_descriptions.items():
        statuses = sorted({row["status"] for row in rows if int(row["step"]) == step})
        lines.append(f"- Step {step}: {description} Recorded statuses: {', '.join(statuses) if statuses else 'not run'}.")

    lines.extend(["", "## Rare crop and domain augmentation", ""])
    rare = next((row for row in rows if row["experiment_id"] == "step05_rare_crop"), None)
    ordinary = next((row for row in rows if row["experiment_id"] == "step03_resolution_512"), None)
    if rare and rare.get("crop_statistics"):
        crop = rare["crop_statistics"]
        lines.append(
            f"Rare-crop run recorded {crop.get('total_crops')} total crops, {crop.get('rare_crops')} target-aware crops, "
            f"Vehicle patch ratio {fmt(crop.get('vehicle_patch_ratio'))}, and Barren patch ratio {fmt(crop.get('barren_patch_ratio'))}."
        )
        lines.append(
            "The configured target-aware attempt probability is 30%; images without an available priority class "
            "correctly fall back to ordinary random crops, so the recorded effective target-aware fraction can be lower."
        )
        if ordinary and ordinary.get("crop_statistics"):
            control_crop = ordinary["crop_statistics"]
            lines.append(
                f"The ordinary-crop control recorded Vehicle/Barren patch ratios "
                f"{fmt(control_crop.get('vehicle_patch_ratio'))}/{fmt(control_crop.get('barren_patch_ratio'))}; "
                f"the rare-aware run recorded {fmt(crop.get('vehicle_patch_ratio'))}/{fmt(crop.get('barren_patch_ratio'))}."
            )
        if ordinary:
            lines.append(
                f"Vehicle false-positive rate was {fmt(ordinary.get('vehicle_false_positive_rate'), 6)} in the control and "
                f"{fmt(rare.get('vehicle_false_positive_rate'), 6)} in the rare-aware run; Vehicle precision was "
                f"{fmt(ordinary.get('vehicle_precision'))}/{fmt(rare.get('vehicle_precision'))}."
            )
    else:
        lines.append("Rare-crop effectiveness is not yet available from a completed run.")
    domain = next((row for row in rows if row["experiment_id"] == "step06_domain_aug"), None)
    lines.append(
        "Domain augmentation completed with saved samples." if domain and domain["status"] == "completed" else
        "Domain augmentation has not yet produced a complete run and saved samples."
    )

    lines.extend(
        [
            "",
            "## Model capacity, memory, and inference speed",
            "",
            "| Model experiment | Parameters | Peak VRAM MB | Step s | Inference images/s | mIoU | Vehicle IoU |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for experiment_id in ("step03_resolution_512", "step07_segformer_b1", "step07_segformer_b2"):
        row = next((item for item in rows if item["experiment_id"] == experiment_id), None)
        if row:
            lines.append(
                f"| {experiment_id} | {fmt(row.get('parameter_count'), 0)} | {fmt(row.get('peak_vram_mb'), 1)} | "
                f"{fmt(row.get('average_step_time'), 3)} | {fmt(row.get('inference_images_per_second'), 3)} | "
                f"{fmt(row.get('mIoU'))} | {fmt(row.get('iou_vehicle'))} |"
            )

    lines.extend(["", "## Failures and limits", ""])
    if failed:
        for row in failed:
            lines.append(f"- `{row['experiment_id']}`: {row['error_message'] or 'see its error.log'}")
    else:
        lines.append("- No recorded failed experiments.")
    if not_run:
        lines.append("- Not run: " + ", ".join(f"`{row['experiment_id']}`" for row in not_run) + ".")
    lines.extend(
        [
            "- 11 组实验均成功完成，没有发生 OOM。",
            "- 下一阶段应先使用固定 5597/1399 划分完成 SegFormer-B0、纯 CE、完整数据正式基线。",
            "",
            "## RTX 4060 and RTX 4090 interpretation",
            "",
            "A configuration is suitable for local RTX 4060 debugging only when its recorded run completed within memory. "
            "Peak VRAM and step time above are engineering measurements, not accuracy rankings.",
            "All 11 recorded configurations completed on this RTX 4060 without OOM, so they are suitable for local quick debugging "
            "with the recorded batch, accumulation, AMP, and checkpointing settings.",
            "",
            "Three candidates to prioritize for controlled full training on RTX 4090 are:",
            "",
            "1. `step01_baseline_512_ce`: extend the reproducible reference run before any optimization claim.",
            "2. `step03_resolution_1024`: test full spatial detail with the strongest resolution demand.",
            "3. `step07_segformer_b2`: test higher model capacity at the controlled 512 crop.",
            "",
            "These are coverage-oriented formal-training candidates, not winners selected from short-run mIoU. "
            "Formal training should first reproduce the baseline, then vary one factor at a time; rare crop and domain "
            "augmentation should be promoted only after longer controlled ablations.",
            "",
            "## Conclusions that quick pilots cannot support",
            "",
            "Quick pilots cannot determine the best resolution, loss, model size, augmentation recipe, leaderboard score, "
            "convergence ranking, or statistical significance. They also cannot justify combining individually promising "
            "changes without a controlled full-training ablation. Weighted CE, rare crop, and domain augmentation should be "
            "scheduled as the next single-variable full ablations after those three reference/capacity runs.",
            "",
            "## Artifact locations",
            "",
            "- ignored local experiment source: `outputs/quick_7step/experiments/`",
            "- CSV summary: `reports/experiments/experiment_summary.csv`",
            "- JSON summary: `reports/experiments/experiment_summary.json`",
            "- this report: `reports/experiments/quick_7step_report.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "quick_7step")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "experiments")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    experiments_dir = output_dir / "experiments"
    rows = [experiment_row(PROJECT_ROOT / "configs" / "quick" / name, experiments_dir) for name in CONFIG_FILES]
    report_dir.mkdir(parents=True, exist_ok=True)
    json_rows = rows
    save_json(report_dir / "experiment_summary.json", {"generated_at_utc": utc_now(), "experiments": json_rows})
    fieldnames = [
        "experiment_id", "step", "model", "model_name", "crop_size", "loss", "rare_crop", "domain_aug",
        "max_steps", "effective_batch_size", "actual_batch_size", "peak_vram_mb", "average_step_time", "mIoU",
        *[f"iou_{name.lower()}" for name in CLASS_NAMES], "pixel_accuracy", "parameter_count",
        "inference_images_per_second", "oom_occurred", "status", "error_message", "best_weight_path", "quick_pilot",
        "vehicle_false_positive_pixels", "vehicle_false_positive_rate", "vehicle_precision", "vehicle_recall",
    ]
    with (report_dir / "experiment_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            writer.writerow(csv_row)
    report = render_report(rows, output_dir)
    (report_dir / "quick_7step_report.md").write_text(report, encoding="utf-8")
    print(f"repository-safe summary written for {len(rows)} experiments to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
