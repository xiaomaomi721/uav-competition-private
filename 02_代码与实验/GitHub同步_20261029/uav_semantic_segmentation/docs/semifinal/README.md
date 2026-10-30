# AIC2026 UAV 语义分割｜半决赛实验档案

> **快照时间：2026-10-29（北京时间）。** 此目录整理半决赛开启以来实际执行的实验、官方线上评分回报、Source 离线指标、结果包状态与未完成研究。今后新增成绩请按日期更新；不能把本快照视为实时排行榜。
>
> **证据约定：**“官方线上”系队伍/用户回报的平台分数，尚非本仓库连接赛事平台独立读取；“Source”是有 GT 的本地验证；“Target 预测变化”不是 mIoU。状态分为 **已评分 / 已完成未评分 / 仅离线诊断 / 训练中止 / 仅设计 / 已取消**。缺少数据写“未提供”，不填推算成绩。

## 快速导航

| 文件 | 内容 |
|---|---|
| [ONLINE_RESULTS_20261029.md](ONLINE_RESULTS_20261029.md) | 半决赛 **Target3594 官方分数**、Source 成绩、候选 ZIP 和 SHA/证据 |
| [EXPERIMENT_CATALOG_20261029.md](EXPERIMENT_CATALOG_20261029.md) | 半决赛以来全部主要已做/未做方案、机制、实验结论及限制 |
| [BIAS_CALIBRATION_20261029.md](BIAS_CALIBRATION_20261029.md) | 10月29日 H75 EM、官方出现率、队友荒地单轴扫描、四份候选的差异 |
| [RULES_AND_DELIVERY_20261029.md](RULES_AND_DELIVERY_20261029.md) | 官方赛题数据、合法性问答、评分协议、交付时间及材料清单 |
| [TIMELINE_20261029.md](TIMELINE_20261029.md) | 10月23–29日关键事件与9月旧赛段分数隔离 |
| [semifinal_experiment_registry_20261029.csv](semifinal_experiment_registry_20261029.csv) | 可筛选的实验台账：分数、协议与状态 |

## 截至 2026-10-29 的结果概览

- **当前已知半决赛最佳**：SegFormer-B3 **H75 + Bstar + D4-TTA8 = 70.998665%**，单checkpoint。H75 checkpoint SHA256：`b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a`。
- 其它已报告 Target3594：T3-FULL-ST **70.990138%**、H75 可靠性蒸馏 C1 **70.907678%**、CTRL@3594 **70.318575%**、H75 Zero **67.979564%**、Mask2Former E02 B1 **67.623700%**。
- **10月29日新候选均尚未官网提交**：H75_PRESENCE_V1、H75_BARREN_2P6、队友的 BARREN_3P4/3P8。各 ZIP 的存在和 QA **不等于官方成绩**。
- EM 相对先验校准未通过预定模拟测试，**未出包**；GPU2 M2F 冷启动长训止于5109有效更新；FDA/LoRA 只做外观可行性审计、**未正式训练A/B**；10月29日 R/W/U 仅为研究设计。
- **2026-11-01 20:00 北京时间**预测 ZIP 截止，**2026-11-02 20:00**作品材料截止，**每队每日最多2次**结果提交。成绩保留最高分；材料须与最高分对应。

## 赛题与评价口径

| 组成 | 数量 | 用途 |
|---|---:|---|
| Source train / val | 5,597 / 1,399 | 固定划分，共6,996有GT |
| Target | **3,594** | 旧test1 500 + 旧test2 1,300 + 新增1,794；不公开像素GT |
| 分割类 | **8** | Background, Building, Road, Water, Barren, Vegetation, Agricultural, Vehicle |

官方 GT 0=Ignore，1..8为八个有效类别；项目内部 `0..7`，正式预测 `argmax+1`。Source mIoU须以全图有效像素累计混淆矩阵后按八类IoU平均。**Source1399、Source VAL700、旧赛段test1/test2和Target3594绝不可直接混排。**

## 仓库范围与复现边界

本仓库原有 Stage01–04 / 9月代码、配置和脱敏指标继续保留。此目录是**10月半决赛文档和成绩归档**，并不表示所有新实验训练源码、checkpoint、3594张预测 PNG 或 ZIP 已上传；这些重型/官方数据仅在团队获授权的存储环境中保留。若要正式复现冠军，请按对应 checkpoint、原生logits、TTA、bias、插值、训练配置与Docker另行补齐，并以最高分对应模型为准。

资料基础：团队于10月29日提供的《半决赛全部已做方案中立交接》《官网规则与提交要求汇总》《Claude校准优先v1.4》《赛题规则确认问题汇总》，加上同日用户与执行方的最新校准/ZIP回报。原交接在 EM、PRESENCE、BARREN 的状态上早于最新报告，**本目录新状态以10月29日后续执行回报覆盖**。
