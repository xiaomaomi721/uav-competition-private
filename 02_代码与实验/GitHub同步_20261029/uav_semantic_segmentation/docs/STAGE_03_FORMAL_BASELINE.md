# Stage 03：SegFormer-B0 正式基线

## 1. 阶段目标

在固定数据划分、固定标签与指标口径下，训练完整数据纯 Cross Entropy 的 SegFormer-B0，建立后续所有优化实验的正式对照组。本文只归档已完成结果，不触发训练或推理。

## 2. 实验 ID

| 项目 | 值 |
|---|---|
| experiment_id | formal_b0_ce_512_seed42 |
| 阶段 | Stage 03 |
| 基线状态 | 已完成 |

## 3. 模型与预训练权重

- 模型：SegFormer-B0
- 预训练模型：`nvidia/segformer-b0-finetuned-ade-512-512`
- 类别数：8
- 正式 best checkpoint 预期 SHA256：`5846D7625FC92CCFDBB065BFC9F8910348F4B81C0010AE253AD598443DCBB4A1`
- 本轮本机搜索未重新找到与预期 SHA256 匹配的正式权重，因此权重哈希状态仍为未验证；该状态不改变已由混淆矩阵复算确认的本地指标。

## 4. 固定数据划分

| 集合 | 数量 |
|---|---:|
| train | 5597 |
| validation | 1399 |
| 合计 | 6996 |
| seed | 42 |

训练继续复用仓库中的 `splits/train.txt` 和 `splits/val.txt`，不重新随机划分。测试集不参与训练和本地验证。

## 5. 标签映射

官方 ID 0 为 Ignore，在训练内存中映射为 255；官方 ID 1–8 分别映射为训练 ID 0–7。提交时执行反向映射：训练 ID 0–7 → 官方 ID 1–8。官方 ID 8 对应 Vehicle，是合法提交类别。

## 6. 训练配置

| 配置 | 值 |
|---|---|
| crop size | 512×512 |
| Loss | Cross Entropy |
| optimizer | AdamW |
| 初始学习率 | 0.00006 |
| weight decay | 0.01 |
| scheduler | polynomial，power=1.0 |
| batch size | 2 |
| gradient accumulation | 2 |
| effective batch size | 4 |
| AMP | 开启 |
| max steps | 56000 |
| validation interval | 1400 |
| Rare Crop / Domain Aug | 均关闭 |

训练环境快照来自曾昭虹执行 Stage 03 正式训练的电脑；归档只保留版本与硬件信息，不保留其 Python 可执行文件、数据集或输出目录的机器绝对路径。

## 7. 训练完成状态

- 完成 step：56000 / 56000
- 验证次数：40
- 最佳 step：56000
- 最终 step：56000
- 最佳结果与最终结果一致
- completion、日志和指标证据均表明这是完整正式训练，不是 quick pilot

## 8. 本地验证指标

| 指标 | 高精度值 |
|---|---:|
| best mIoU | 0.7123106568503285 |
| final mIoU | 0.7123106568503285 |
| Pixel Accuracy | 0.8527743645537557 |
| peak VRAM | 685.2236328125 MB |
| average step time | 0.6871561887768143 s |
| final loss | 0.413554385304451 |
| minimum loss | 0.026120157446712255 |

以上主指标由 8×8 混淆矩阵独立重算，与 metrics.json 和 per_class_iou.csv 一致。

## 9. 八类 IoU

| 训练 ID | 官方 ID | 类别 | IoU |
|---:|---:|---|---:|
| 0 | 1 | Background | 0.6252879810111697 |
| 1 | 2 | Building | 0.8039648167000311 |
| 2 | 3 | Road | 0.7060204340972281 |
| 3 | 4 | Water | 0.8309477857329750 |
| 4 | 5 | Barren | 0.4835624404398186 |
| 5 | 6 | Vegetation | 0.8604343453099063 |
| 6 | 7 | Agricultural | 0.7191950718043510 |
| 7 | 8 | Vehicle | 0.6690723797071482 |

Barren 是当前最低 IoU 类别。Vehicle 整体 IoU 为 0.6690723797071482（约 0.669072），但部分小目标样本仍有漏分。

## 10. 训练稳定性

- CUDA OOM：无。
- NaN / Inf：无，loss 始终有限。
- step 1–56000 连续，40 次验证均位于 1400 的整数倍。
- 本地验证后期进入平台期，但最后一次验证仍创最佳结果，未见训练崩塌。
- polynomial 学习率在 step 56000 降至 0。

## 11. 可视化结论摘要

8 张验证三联图表明大类别轮廓与数值指标一致，预测没有整体坍缩。Building、Road、Water 和 Vegetation 主体通常较稳定；细边界、小面积类别和小目标仍容易遗漏。仓库只归档三张纯训练统计曲线，不归档原图/GT/预测三联图。

## 12. 主要错误模式

- Building 存在漏分为 Background 的情况，部分建筑轮廓欠分割。
- Agricultural、Barren 和 Vegetation 之间存在明显混淆，个别 Agricultural 区块会整块误为 Barren。
- Barren 全量 IoU 最低，是优先改进对象。
- Vehicle 总体指标可用，但部分小目标样本仍有漏分或轮廓不完整。
- 细 Road、边界区域和小面积 Water/Building 可能被主导类别吞并。

## 13. 当前基线的作用

Stage 03 是后续所有优化实验的正式对照组。Stage 05 的每项实验都应与本实验保持同一固定划分、指标实现和提交口径，并一次只改变一个核心变量。

## 14. 不建议直接续训的原因

原 polynomial 调度已在 56000 step 将学习率降至 0，直接恢复原 scheduler 状态机械续训没有实际学习率；同时后 10 次验证净增长已很小。若要延长训练，必须作为新实验重新设计学习率和调度，并记录新的 checkpoint 哈希，而不能覆盖本基线。

## 15. 后续优化方向

1. 围绕 Barren 做稀有类/困难样本 crop 的单变量实验。
2. 单独比较 Weighted CE、CE+Dice 或 Focal，不混合多个改动。
3. 针对 Agricultural/Barren/Vegetation 混淆评估保守域增强。
4. 针对 Building→Background 与 Vehicle 小目标评估更高分辨率或多尺度。
5. 以线上泛化差距为约束，优先检查分辨率、域偏移和提交链路证据。

## 16. 外部大型产物保存策略

正式权重、完整 56000 行训练日志、官方数据、8 张三联图、500 张测试预测和提交 ZIP 均只在团队授权的本地或备份介质保存，由 Git 忽略。仓库仅保留脱敏配置、轻量指标、混淆矩阵、训练曲线和证据状态；机器绝对路径只写入 `local_records/` 下的忽略文件。
