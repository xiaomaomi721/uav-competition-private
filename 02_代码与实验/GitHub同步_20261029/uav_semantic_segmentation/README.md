# 2026AIC 无人机低空航拍图像语义分割

> **最新进展（2026-10-29，北京时间）｜AIC2026 半决赛 Target3594：已知最高线上 mIoU = 70.998665%（H75 SegFormer-B3 + Bstar + D4-TTA8）。**
>
> 完整的10月半决赛技术档案已更新至 **[docs/semifinal/（点击查看）](docs/semifinal/README.md)**：
> - [线上成绩与Source/VAL对照、ZIP SHA](docs/semifinal/ONLINE_RESULTS_20261029.md)
> - [全部已做方案、实验结果、失败原因与未执行状态](docs/semifinal/EXPERIMENT_CATALOG_20261029.md)
> - [10月29日EM/官方出现率/bias单轴实验与四个候选ZIP](docs/semifinal/BIAS_CALIBRATION_20261029.md)
> - [比赛规则、交付要求及截止时间](docs/semifinal/RULES_AND_DELIVERY_20261029.md)
> - [半决赛逐日时间线](docs/semifinal/TIMELINE_20261029.md) · [可筛选CSV台账](docs/semifinal/semifinal_experiment_registry_20261029.csv)
>
> **重要状态**：H75_PRESENCE_V1、H75_BARREN_2P6和队友的BARREN_3P4/3P8已完成3594张ZIP/QA，但截至本次记录**均未线上评分**；勿误写成新纪录。早期 Stage 01–04、9月复赛与本次半决赛采用不同测试集合，成绩不可混排。仓库不上传官方图像/标签、模型权重或提交ZIP。

> **2026-09-30 桌面项目同步：** 后续实验的代码、配置、审计记录与文档已放在 [无人机/](无人机/README_从这里开始.md)。下文的 Stage 01–04 状态是此仓库早期基线的历史记录；最新实验请从桌面项目目录及其交接文档阅读。官方数据、模型权重、预测图与提交 ZIP 保留在本地，未上传。

## 历史 Stage 01–04 状态快照（非当前半决赛）

- Stage 01：数据集审计与项目初始化——已完成
- Stage 02：固定划分、标签映射与七步快速实验——已完成
- Stage 03：SegFormer-B0 完整数据纯 CE 正式基线——已完成；本地 mIoU：`0.7123106569`
- Stage 04：首次测试集推理与线上提交——已完成；线上 mIoU：`0.572820`
- Stage 05：RTX 4090 单变量优化——待开始

> Stage 03 是后续优化实验的正式对照组。Stage 04 线上成绩来自赛事平台返回结果，由用户人工记录，不是本地可复算指标。

## 项目简介

本项目面向“2026AIC·无人机低空航拍图像语义分割”赛题。任务是为 1024×1024 无人机航拍图像进行逐像素分类，最终生成与测试图片同名的单通道 PNG，主要评价指标为 mIoU。

仓库保留可复现的代码、配置、固定划分、测试和轻量报告。官方数据、模型权重、训练日志、测试预测和提交 ZIP 只保存在各成员本地并由 Git 忽略。

## 当前完成情况

- 完成 6996 张训练图像、6996 张标签和 500 张初赛测试图像的只读审计。
- 固定唯一的 5597/1399 train/val 划分，seed=42，无交集、重复、遗漏或未知 stem。
- 实现官方 0→训练 255、官方 1–8→训练 0–7 的内存标签映射。
- 实现 Dataset、同步增强、Loss、8 类 mIoU、训练、验证、checkpoint 和推理链路。
- 完成 SegFormer 七步 quick pilot，共 11 组短程实验，全部完成且没有 OOM。
- 完成第一次 500 张测试集探路推理，500 张预测全部成功。
- 提交 PNG 和扁平 ZIP 格式检查通过，ZIP 复检 testzip=None。
- 生成仓库安全的 quick 实验摘要和第一次提交记录。
- 完成 Stage 03 正式基线 56000/56000 step 训练，固定验证集 mIoU 为 0.7123106568503285。
- 完成 Stage 04 首次正式测试集提交，平台 mIoU 为 0.572820。
- 完成正式结果的独立复算、脱敏归档和统一实验台账；权重到 500 张预测的完整证据链仍为条件项。

尚未开始 Stage 05 单变量优化，也没有把 quick 结果直接当作最终最佳模型、Loss 或分辨率结论。

## 数据集核心信息

| 项目 | 审计结论 |
|---|---|
| 训练图像 | 6996 张 |
| 训练标签 | 6996 张 |
| 初赛测试图像 | 500 张 |
| 图像 | 1024×1024，RGB，uint8，PNG |
| 标签 | 1024×1024，单通道 L，uint8，PNG |
| 官方标签值 | 0–8 |
| 图像标签配对 | 完全对应 |
| 损坏、空文件或尺寸异常 | 未发现 |
| 主要风险 | 类别不平衡、小目标分割、跨场景泛化 |

Vehicle 仅占有效像素约 0.783%，Barren 约占 1.835%；最大和最小有效类别像素量约相差 36.745 倍。

## 类别与训练映射

| 官方 ID | 类别 | 训练 ID |
|---:|---|---:|
| 0 | Ignore | 255 |
| 1 | Background | 0 |
| 2 | Building | 1 |
| 3 | Road | 2 |
| 4 | Water | 3 |
| 5 | Barren | 4 |
| 6 | Vegetation | 5 |
| 7 | Agricultural | 6 |
| 8 | Vehicle | 7 |

训练使用 num_classes=8、ignore_index=255。映射只在数据加载时动态执行，不改写官方标签；预测提交时将训练 0–7 还原为官方 1–8。

## 项目结构

    uav_semantic_segmentation/
    ├── configs/
    │   ├── classes.yaml
    │   ├── split_config.yaml
    │   ├── quick/                  # 七步 quick 单变量配置
    │   └── full/                   # Stage 03 正式配置入口
    ├── data/
    │   └── README.md               # 官方数据本地放置规则
    ├── docs/
    │   ├── DATASET.md
    │   ├── ENVIRONMENT.md
    │   ├── FIRST_SUBMISSION.md
    │   ├── STAGE_01_DATA_AUDIT.md
    │   ├── STAGE_02_7STEP_QUICK.md
    │   ├── STAGE_03_BASELINE_PREPARATION.md
    │   ├── STAGE_03_FORMAL_BASELINE.md
    │   ├── STAGE_03_ACCEPTANCE.md
    │   ├── STAGE_04_FIRST_SUBMISSION.md
    │   └── TEAM_WORKFLOW.md
    ├── reports/
    │   ├── dataset_report.*
    │   ├── class_statistics.csv
    │   ├── abnormal_files.csv
    │   ├── experiments/            # 11 组 quick 的轻量摘要
    │   ├── stage03/                # 正式基线脱敏指标、验收摘要与曲线
    │   ├── stage04/                # 首次线上提交脱敏摘要
    │   └── visual_samples/         # 本地保留，Git 忽略
    ├── scripts/
    │   ├── inspect_dataset.py
    │   ├── create_splits.py / check_split.py
    │   ├── train.py / evaluate.py
    │   ├── run_7step_quick.py / summarize_experiments.py
    │   └── predict_submission.py / check_submission.py / package_submission.py
    ├── splits/                     # 固定 train/val 及统计，不得重建
    ├── src/                        # 数据、增强、Loss、指标、模型、训练与推理
    ├── tests/                      # 测试说明；可执行回归测试位于 scripts/test_*.py
    ├── experiments/                # 正式实验统一台账
    ├── local_records/              # 本机路径记录；除 README 外均忽略
    ├── outputs/                    # 本地重型产物，Git 整体忽略
    ├── requirements-audit.txt
    ├── requirements-segmentation.txt
    └── README.md

## 队友复现入口

1. 克隆或拉取仓库。
2. 通过官方授权渠道自行获取同一份数据集。
3. 将本机数据根目录记为 `<DATASET_ROOT>`，不要写进代码或提交记录。
4. 不要重新运行随机划分，不要生成个人版本的 train/val。
5. 直接使用仓库中的 splits/train.txt 和 splits/val.txt。
6. 按 docs/ENVIRONMENT.md 准备与显卡匹配的 PyTorch/CUDA，再安装轻量直接依赖。
7. 先运行环境检查、固定划分检查、三个专项回归测试和 Stage 03 配置静态检查。
8. 阅读 Stage 03 正式归档和实验台账；新实验从 Stage 05 开始并保持单变量对照。

建议检查命令：

    python scripts/check_env.py --output outputs/environment_check.json
    python scripts/check_split.py --dataset-root "<DATASET_ROOT>"
    python scripts/test_label_mapping.py
    python scripts/test_metrics.py
    python scripts/test_transforms.py
    python scripts/check_stage03_config.py

推荐执行顺序：

    check_env
    -> check_split
    -> test_label_mapping
    -> test_metrics
    -> test_transforms
    -> check_stage03_config
    -> 阅读 Stage 03 验收摘要
    -> 登记 Stage 05 单变量实验

Windows 环境默认建议 num_workers=0。PyTorch/CUDA 请单独按平台安装，不要让普通 requirements 命令覆盖已验证的 CUDA 构建。

## 只读数据审计

    python -m pip install -r requirements-audit.txt
    python scripts/inspect_dataset.py --dataset-root "<DATASET_ROOT>" --output-dir reports --sample-count 20 --seed 42

只有需要完整 SHA256 时才添加 --compute-hash。报告目录不得位于官方数据目录内部。

## 训练工程入口

quick 编排已经完成，命令仅用于复现或排查，不应在不了解成本时重复运行：

    python scripts/run_7step_quick.py --dataset-root "<DATASET_ROOT>" --device cuda --output-dir outputs/quick_7step --skip-completed

Stage 03 已使用 configs/full/stage03_b0_ce_512_seed42.yaml 完成正式训练。该配置是完整数据、固定 5597/1399 划分、SegFormer-B0、512 crop、普通 CE 的正式基线配置；正式结果见 [Stage 03 正式基线](docs/STAGE_03_FORMAL_BASELINE.md) 与 [Stage 03 独立验收](docs/STAGE_03_ACCEPTANCE.md)。不得把 quick 配置直接描述为正式最佳方案，也不建议沿用已降至零的原学习率调度机械续训。

## 团队成员与暂定职责

| 成员/资源 | 暂定职责 |
|---|---|
| 蒋鸿宇 | 队长；统筹协调、进度管理、GitHub 阶段审核、文档和任务对齐；使用商务本，不承担主要本地训练任务 |
| 曾昭虹 | 算法与训练实现；RTX 4060 游戏本，负责本地环境验证、冒烟训练和部分实验 |
| 谢诗博 | 技术路线和工程集成、算法与训练实现；RTX 4060 游戏本，负责 Codex 任务拆解、代码整合和结果分析 |
| 教师提供算力 | RTX 4090，用于完整训练、高分辨率实验和较大模型实验 |

## 数据与 Git 安全

- 官方图像、标签、压缩包和授权材料不进入 Git。
- 不提交 best_miou.pt、last.pt 或其他模型权重。
- 不提交 500 张测试预测、submission ZIP、缓存、日志或完整 outputs/。
- 不移动、覆盖、重编码或自动修复官方原始标签。
- reports/experiments/ 只包含从真实 outputs 提取的轻量脱敏摘要。
- 所有实验使用统一固定划分和指标实现，正式合并前由另一名成员检查。

## 阶段路线图

- Stage 01：数据集审计与项目初始化（已完成）
- Stage 02：固定数据划分与七步 quick pilot（已完成）
- 第一次初赛探路提交包与格式验证（已完成）
- Stage 03：SegFormer-B0 完整数据纯 CE 正式基线（已完成，本地 mIoU=0.7123106569）
- Stage 04：首次测试集推理与线上提交（已完成，线上 mIoU=0.572820）
- Stage 05：RTX 4090 单变量优化与消融实验（待开始）
- Stage 06：最佳单模型、Docker 和技术报告

## 文档导航

- docs/DATASET.md：数据格式、类别、长尾风险和训练映射
- docs/ENVIRONMENT.md：已验证环境和算力建议
- docs/STAGE_01_DATA_AUDIT.md：Stage 01 验收
- docs/STAGE_02_7STEP_QUICK.md：固定划分与七步 quick 验收
- docs/STAGE_03_BASELINE_PREPARATION.md：Stage 03 正式基线配置、执行顺序和验收标准
- [docs/STAGE_03_FORMAL_BASELINE.md](docs/STAGE_03_FORMAL_BASELINE.md)：Stage 03 正式基线结果与错误模式
- [docs/STAGE_03_ACCEPTANCE.md](docs/STAGE_03_ACCEPTANCE.md)：Stage 03 独立验收、证据门禁与权重状态
- [docs/STAGE_04_FIRST_SUBMISSION.md](docs/STAGE_04_FIRST_SUBMISSION.md)：Stage 04 首次线上提交记录
- [experiments/experiment_registry.csv](experiments/experiment_registry.csv)：正式实验统一台账
- docs/FIRST_SUBMISSION.md：第一次探路提交包与预测分布
- docs/TEAM_WORKFLOW.md：成员职责和 GitHub 协作规则
- reports/experiments/quick_7step_report.md：11 组实验详细轻量汇总
- reports/dataset_report.md：完整数据集审计报告
