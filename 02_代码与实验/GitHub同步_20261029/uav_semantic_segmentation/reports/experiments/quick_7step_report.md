# Stage 02: seven-step quick engineering validation

## 实验性质

全部 11 组实验均为 quick pilot，仅用于验证工程可运行性、显存占用、Loss 稳定性、分辨率可运行性、模型结构加载以及稀有类别裁剪和域增强逻辑。
这些 100 或 200 step 短程结果不是正式收敛结果，不能直接确定最佳 Loss、最佳分辨率或最佳模型。

> These runs are quick pilots for code-path, memory, metric, and reproducibility validation. They are not formal competition results and cannot establish the final best model, loss, resolution, or augmentation.

Generated: 2026-07-11T13:00:06.346858+00:00

## Execution status

- completed: 11
- failed: 0
- not run: 0

| Experiment | Step | Model | Crop | Loss | Rare crop | Domain aug | Status | mIoU | Peak VRAM MB | Step s |
|---|---:|---|---:|---|---|---|---|---:|---:|---:|
| step01_baseline_512_ce | 1 | SegFormer-B0 | 512 | cross_entropy | False | False | completed | 0.2551 | 684.0 | 0.918 |
| step03_resolution_512 | 3 | SegFormer-B0 | 512 | cross_entropy | False | False | completed | 0.1913 | 684.0 | 0.847 |
| step03_resolution_768 | 3 | SegFormer-B0 | 768 | cross_entropy | False | False | completed | 0.2275 | 687.4 | 1.526 |
| step03_resolution_1024 | 3 | SegFormer-B0 | 1024 | cross_entropy | False | False | completed | 0.2345 | 764.3 | 2.583 |
| step04_weighted_ce | 4 | SegFormer-B0 | 512 | weighted_cross_entropy | False | False | completed | 0.2474 | 684.0 | 0.849 |
| step04_ce_dice | 4 | SegFormer-B0 | 512 | ce_dice | False | False | completed | 0.1897 | 684.0 | 0.826 |
| step04_focal | 4 | SegFormer-B0 | 512 | focal | False | False | completed | 0.1975 | 684.0 | 0.858 |
| step05_rare_crop | 5 | SegFormer-B0 | 512 | cross_entropy | True | False | completed | 0.2562 | 684.0 | 1.006 |
| step06_domain_aug | 6 | SegFormer-B0 | 512 | cross_entropy | False | True | completed | 0.2563 | 684.0 | 0.952 |
| step07_segformer_b1 | 7 | SegFormer-B1 | 512 | cross_entropy | False | False | completed | 0.1978 | 807.7 | 0.844 |
| step07_segformer_b2 | 7 | SegFormer-B2 | 512 | cross_entropy | False | False | completed | 0.2472 | 2057.9 | 2.167 |

## Per-class IoU

| Experiment | Background | Building | Road | Water | Barren | Vegetation | Agricultural | Vehicle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step01_baseline_512_ce | 0.3360 | 0.4897 | 0.3694 | 0.1052 | 0.0000 | 0.6804 | 0.0604 | 0.0000 |
| step03_resolution_512 | 0.3037 | 0.2705 | 0.0772 | 0.1092 | 0.0063 | 0.6719 | 0.0914 | 0.0000 |
| step03_resolution_768 | 0.3495 | 0.3461 | 0.1979 | 0.2409 | 0.0000 | 0.6830 | 0.0001 | 0.0021 |
| step03_resolution_1024 | 0.3132 | 0.3993 | 0.2763 | 0.2459 | 0.0000 | 0.6394 | 0.0013 | 0.0002 |
| step04_weighted_ce | 0.2992 | 0.2785 | 0.2190 | 0.2903 | 0.0265 | 0.6711 | 0.1897 | 0.0047 |
| step04_ce_dice | 0.3051 | 0.2637 | 0.0846 | 0.1089 | 0.0085 | 0.6731 | 0.0740 | 0.0000 |
| step04_focal | 0.3024 | 0.2863 | 0.0879 | 0.1200 | 0.0075 | 0.6712 | 0.1045 | 0.0001 |
| step05_rare_crop | 0.3159 | 0.4071 | 0.2149 | 0.2789 | 0.0370 | 0.6572 | 0.1383 | 0.0001 |
| step06_domain_aug | 0.3200 | 0.3922 | 0.1570 | 0.3604 | 0.0041 | 0.6680 | 0.1490 | 0.0000 |
| step07_segformer_b1 | 0.3406 | 0.1467 | 0.3276 | 0.0917 | 0.0000 | 0.6759 | 0.0000 | 0.0000 |
| step07_segformer_b2 | 0.3342 | 0.3631 | 0.0428 | 0.5704 | 0.0000 | 0.6514 | 0.0158 | 0.0000 |

## Validation of the seven technical steps

The immutable split contains 5597 training and 1399 validation samples; the fixed quick subset contains 256 stems.
Prechecks: environment=passed, create_or_preserve_split=passed, split=passed, label_mapping=passed, transforms=passed, metrics=passed.
Official label 0 is mapped in memory to 255, official 1..8 to train 0..7; Ignore is excluded from loss, confusion matrix, per-class IoU, mIoU, and Pixel Accuracy.
Step 2 is covered by the label-mapping, transform-synchronization, metric, and split prechecks.
- Step 1: SegFormer-B0 + cross-entropy + 512 baseline and checkpoint/log/metric chain. Recorded statuses: completed.
- Step 3: 512/768/1024 controlled resolution feasibility comparison. Recorded statuses: completed.
- Step 4: CE control plus weighted CE, CE+Dice, and focal loss code paths. Recorded statuses: completed.
- Step 5: 70% ordinary plus 30% Vehicle/Barren/Agricultural-aware crop path and crop counters. Recorded statuses: completed.
- Step 6: Conservative color/imaging domain augmentation applied to training images only. Recorded statuses: completed.
- Step 7: SegFormer-B0/B1/B2 capacity, memory, speed, and forward/backward feasibility. Recorded statuses: completed.

## Rare crop and domain augmentation

Rare-crop run recorded 400 total crops, 60 target-aware crops, Vehicle patch ratio 0.1575, and Barren patch ratio 0.1175.
The configured target-aware attempt probability is 30%; images without an available priority class correctly fall back to ordinary random crops, so the recorded effective target-aware fraction can be lower.
The ordinary-crop control recorded Vehicle/Barren patch ratios 0.1550/0.0875; the rare-aware run recorded 0.1575/0.1175.
Vehicle false-positive rate was 0.000014 in the control and 0.000008 in the rare-aware run; Vehicle precision was 0.0313/0.1575.
Domain augmentation completed with saved samples.

## Model capacity, memory, and inference speed

| Model experiment | Parameters | Peak VRAM MB | Step s | Inference images/s | mIoU | Vehicle IoU |
|---|---:|---:|---:|---:|---:|---:|
| step03_resolution_512 | 3716200 | 684.0 | 0.847 | 13.594 | 0.1913 | 0.0000 |
| step07_segformer_b1 | 13679304 | 807.7 | 0.844 | 12.659 | 0.1978 | 0.0000 |
| step07_segformer_b2 | 27352776 | 2057.9 | 2.167 | 4.618 | 0.2472 | 0.0000 |

## Failures and limits

- No recorded failed experiments.
- 11 组实验均成功完成，没有发生 OOM。
- 下一阶段应先使用固定 5597/1399 划分完成 SegFormer-B0、纯 CE、完整数据正式基线。

## RTX 4060 and RTX 4090 interpretation

A configuration is suitable for local RTX 4060 debugging only when its recorded run completed within memory. Peak VRAM and step time above are engineering measurements, not accuracy rankings.
All 11 recorded configurations completed on this RTX 4060 without OOM, so they are suitable for local quick debugging with the recorded batch, accumulation, AMP, and checkpointing settings.

Three candidates to prioritize for controlled full training on RTX 4090 are:

1. `step01_baseline_512_ce`: extend the reproducible reference run before any optimization claim.
2. `step03_resolution_1024`: test full spatial detail with the strongest resolution demand.
3. `step07_segformer_b2`: test higher model capacity at the controlled 512 crop.

These are coverage-oriented formal-training candidates, not winners selected from short-run mIoU. Formal training should first reproduce the baseline, then vary one factor at a time; rare crop and domain augmentation should be promoted only after longer controlled ablations.

## Conclusions that quick pilots cannot support

Quick pilots cannot determine the best resolution, loss, model size, augmentation recipe, leaderboard score, convergence ranking, or statistical significance. They also cannot justify combining individually promising changes without a controlled full-training ablation. Weighted CE, rare crop, and domain augmentation should be scheduled as the next single-variable full ablations after those three reference/capacity runs.

## Artifact locations

- ignored local experiment source: `outputs/quick_7step/experiments/`
- CSV summary: `reports/experiments/experiment_summary.csv`
- JSON summary: `reports/experiments/experiment_summary.json`
- this report: `reports/experiments/quick_7step_report.md`
