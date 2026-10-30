# Stage 02：固定数据划分与七步 quick pilot

## 阶段目标

在不追求正式收敛的前提下，固定全队唯一的数据划分和标签/指标口径，打通 SegFormer 训练、验证、权重保存、推理和报告链路，并通过七步 quick pilot 验证不同分辨率、Loss、稀有类别裁剪、域增强和模型容量的工程可运行性。

本阶段所有训练均为 100 或 200 step 的短程 pilot，不代表正式比赛结论。

## 固定数据划分

| 项目 | 数值 |
|---|---:|
| train | 5597 |
| val | 1399 |
| 合计 | 6996 |
| seed | 42 |
| 策略 | stratified_presence_combination_largest_remainder |

固定文件：

- splits/train.txt
- splits/val.txt
- splits/split_statistics.json
- splits/split_report.md

划分已经验证无重复、无交集、无遗漏、无未知 stem，且每个 stem 都有官方图像和标签。所有成员必须直接复用这套文件，不允许自行重新随机划分。

## 标签映射

训练期间只在内存中动态映射，不改写官方 PNG：

| 官方 ID | 训练 ID |
|---:|---:|
| 0 Ignore | 255 |
| 1–8 | 0–7 |

预测提交时执行反向映射：训练 0–7 → 官方 1–8。普通模型没有 Ignore 输出通道，不使用低置信度规则产生官方 ID 0。

## 指标口径

- num_classes=8
- ignore_index=255
- Ignore 不进入 Loss
- Ignore 不进入混淆矩阵、逐类 IoU、mIoU 和 Pixel Accuracy
- mIoU 对 8 个有效训练类别计算
- 验证使用固定 val 列表且没有随机增强
- 标签几何变换使用 nearest，图像使用双线性插值
- 图像与标签的翻转、旋转和裁剪保持同步

主实现位于 src/label_mapping.py、src/losses.py、src/metrics.py、src/transforms.py 和 src/inference.py。

## 七步实验路线

1. SegFormer-B0、512 crop、纯 CE 基础链路。
2. 标签映射、增强同步、指标和固定划分预检查。
3. 512、768、1024 分辨率可运行性。
4. 纯 CE、Weighted CE、CE+Dice、Focal 的 Loss 路径。
5. Vehicle、Barren、Agricultural 感知的 Rare Crop 路径。
6. 仅训练图像使用的保守 Domain Augmentation。
7. SegFormer-B0、B1、B2 的容量、速度和显存可运行性。

## 实际完成的实验

真实来源为 reports/experiments/experiment_summary.csv 和 experiment_summary.json。

| 实验 | 模型 | Crop | Loss/变量 | Steps | mIoU |
|---|---|---:|---|---:|---:|
| step01_baseline_512_ce | B0 | 512 | CE baseline | 200 | 0.255148 |
| step03_resolution_512 | B0 | 512 | CE | 100 | 0.191296 |
| step03_resolution_768 | B0 | 768 | CE | 100 | 0.227450 |
| step03_resolution_1024 | B0 | 1024 | CE | 100 | 0.234465 |
| step04_weighted_ce | B0 | 512 | Weighted CE | 100 | 0.247365 |
| step04_ce_dice | B0 | 512 | CE+Dice | 100 | 0.189738 |
| step04_focal | B0 | 512 | Focal | 100 | 0.197491 |
| step05_rare_crop | B0 | 512 | Rare Crop | 100 | 0.256185 |
| step06_domain_aug | B0 | 512 | Domain Aug | 100 | 0.256315 |
| step07_segformer_b1 | B1 | 512 | CE | 100 | 0.197799 |
| step07_segformer_b2 | B2 | 512 | CE | 100 | 0.247218 |

11 组实验均成功完成，没有发生 OOM。完整逐类 IoU、显存、速度和参数量见 reports/experiments/quick_7step_report.md。

## 主要观察

- SegFormer-B0 的训练、验证、checkpoint 和推理链路可运行。
- 512、768、1024 crop 均能在已验证设置下运行。
- Weighted CE 在 quick 条件下改善了部分 Barren、Agricultural、Vehicle 指标，但总 mIoU 未超过最优 quick 结果。
- Rare Crop 和 Domain Augmentation 的 quick mIoU 略高于基础 pilot，但差异不足以形成正式优劣结论。
- B1/B2 的 100-step 结果不能用于判断正式模型容量优劣。
- Vehicle 在多数 quick 实验中仍接近零 IoU，是最主要薄弱类别。
- 11 组实验的显存和 step time 是工程测量，不是精度排名。

## 本阶段限制

- 只运行了 100 或 200 step。
- 使用受限训练样本和固定 quick 验证子集。
- 尚未正式收敛，也没有重复实验或统计显著性分析。
- quick 结果不能直接确定最佳 Loss、最佳分辨率、最佳模型或最终增强组合。
- 尚未完成完整 5597 训练样本上的正式基线。
- 尚无线上分数可用于判断泛化质量。

## 下一阶段

Stage 03：SegFormer-B0 完整数据纯 CE 正式基线。

Stage 03 必须首先复现单变量、可审查的 B0+CE 基线，使用固定 5597/1399 划分并完整记录逐类 IoU、训练曲线、最佳权重和推理配置；在此之前不组合 quick 中看似有利的改动。

## 阶段验收结论

Stage 02 通过验收：固定划分、标签映射、8 类指标、训练工程和 11 组 quick pilot 已完成并可复现；所有重型输出继续留在被 Git 忽略的 outputs/ 中，仓库仅保留代码、配置、固定划分和轻量报告。
