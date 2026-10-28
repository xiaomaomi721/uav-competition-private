# ALL_SCHEMES_CORE_AUDIT — 本地训练core账本

范围：62条原始账本行＋6条缺号占位＋2条staged checkpoint，共70行。缺号是NOT_FOUND_IN_SCOPED_LOCAL_SOURCES，不是虚构实验。四条14000步短跑分别列出。

原CSV UNKNOWN保留；Phase2补Exp02/03/04/08/11精确线上分，UPerNet较新回填补Exp16=65.9078与Exp14=66.443。不同精度并列而非静默覆盖。22行没有无TTA标准绝对分（其中C7/C8只有TTA成绩）。

八维来源：许多其他core校准版复用Stage23不同阶段向量，并非该core专属八维搜索。存在本模型部分扫描文字记录者单列；Exp49/55b是否另外rescan仍UNKNOWN。

所有线上成绩为百分制，local为0–1。历史正负按原baseline记，不意味着优于冠军。无平台receipt重新核验，不从相对delta反推未知绝对分。

| id / 状态 | 方法/参数 | steps / local | 标准线上（原始精度并列） | TTA/校准关联成绩 | 原基线/涨跌 | 来源 |
|---|---|---|---|---|---|---|
| Exp01 | CE / rare_crop(disabled) | 56000 / 0.710122 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "Exp01=None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:2 |
| Exp02 | CE / - | 56000 / 0.743154 | 62.279098 phase2 model-linked online score;该表不单独说明TTA，不能凭表名calibration推断预测校准 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:3; phase2_calibration_table.csv:8 |
| Exp03 | CE / domain_aug(p0.5) | 56000 / 0.741288 | 61.807964 phase2 model-linked online score;该表不单独说明TTA，不能凭表名calibration推断预测校准 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:4; phase2_calibration_table.csv:9 |
| Exp04 | CE / rare_crop(p0.5) | 56000 / 0.749709 | 62.471861 phase2 model-linked online score;该表不单独说明TTA，不能凭表名calibration推断预测校准 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:5; phase2_calibration_table.csv:7 |
| Exp05 | CE / rare_crop | 56000 / 0.769940 | 64.283137 ; 64.283137 phase2 model-linked online score;该表不单独说明TTA，不能凭表名calibration推断预测校准 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:6; phase2_calibration_table.csv:6 |
| Exp06 | CE / rare_crop | 56000 / 0.787114 | 66.212776 ; 66.212776 phase2 model-linked online score;该表不单独说明TTA，不能凭表名calibration推断预测校准 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:7; phase2_calibration_table.csv:2 |
| Exp07 | CE / rare_crop | 56000 / 0.796894 | 65.484737 ; 65.484737 phase2 model-linked online score;该表不单独说明TTA，不能凭表名calibration推断预测校准 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:8; phase2_calibration_table.csv:5 |
| Exp08 | weighted_CE / weights clip.5-5.0 | 56000 / 0.778700 | 65.837945 phase2 model-linked online score;该表不单独说明TTA，不能凭表名calibration推断预测校准 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:9; phase2_calibration_table.csv:3 |
| Exp09 | CE / light_blur(p0.2,r.4-.8) | 56000 / 0.787219 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:10 |
| Exp10 | CE / ema(decay.999) | 56000 / 0.787047 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:11 |
| Exp11 | ce_lovasz / ce.8/lovasz.2 | 56000 / 0.789225 | 65.797297 phase2 model-linked online score;该表不单独说明TTA，不能凭表名calibration推断预测校准 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:12; phase2_calibration_table.csv:4 |
| Exp12 | CE / light_blur(p0.2,r.3-.8) | 56000 / 0.783942 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:13 |
| Exp13 | CE / light_blur(p0.1,r.3-.8) | 56000 / 0.782152 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:14 |
| Exp14 | CE / rare_crop(cwfa off) | 56000 / 0.804564 | 66.443 B2/1024 baseline引用；精确尾数未证实 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:15 |
| Exp15 | CE / rare_crop | 56000 / 0.798495 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:16 |
| Exp16 | CE / rare_crop+upernet头 | 56000 / 0.806580 | 65.9078 single-view;TTA=false;ensemble=false;无校准无后处理 | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false, "later_result": {"parent": "Exp14", "score": 66.443, "delta": -0.5352, "sources": [{"path": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\exp16_upernet_online_validation_20260909\\UPERNET_ONLINE_VALIDATION_REPORT.md", "line": 9}, {"path": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\exp16_upernet_online_validation_20260909\\UPERNET_ONLINE_VALIDATION_REPORT.md", "line": 10}]}} | raw_master_experiment_ledger.csv:17; UPERNET_ONLINE_VALIDATION_REPORT.md:5; UPERNET_ONLINE_VALIDATION_REPORT.md:8; UPERNET_ONLINE_VALIDATION_REPORT.md:27 |
| Exp17 | CE / rare+cwfa(p0.3,eps4,28001) | 56000 / 0.803976 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:18 |
| Exp18 | CE / rare_crop | 56000 / 0.810525 | 66.739544 ; 66.739544 无后处理 | 67.209674 TTA8无校准; 68.032596 TTA8 V1e; 68.039691 TTA8 V1f背景-0.020 | {"parent_as_recorded": "None", "delta_as_recorded": "0", "delta_computed": false} | raw_master_experiment_ledger.csv:19; 线上成绩记录表.md:13 |
| Exp19 | - / - | - / - | UNKNOWN | UNKNOWN | {"parent_as_recorded": "-", "delta_as_recorded": "-", "delta_computed": false} | raw_master_experiment_ledger.csv:20 |
| Exp20 | CE / copy_paste(smoke only) | - / UNKNOWN | UNKNOWN | UNKNOWN | {"parent_as_recorded": "-", "delta_as_recorded": "-", "delta_computed": false} | raw_master_experiment_ledger.csv:21 |
| Exp21a | CE / rare+cwfa(eps6.0) | 56000 / 0.803610 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:22 |
| Exp22 | CE / rare+pseudo | 56000 / 0.799759 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "None", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:23 |
| Exp23 | CE / rare+cwfa(p0.3,eps4,28001) | 56000 / 0.810986 | 67.371405 ; 67.371405 无后处理 | 68.686044 TTA8+V1e; 69.28129 自身8D扫描最终building+0.10 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "+0.632(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:24; 线上成绩记录表.md:43 |
| Exp24 | CE / rare+pseudo(500) | 56000 / 0.813576 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "Exp23?", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:25 |
| Exp25 | CE / rare+no_decay | 56000 / 0.810192 | 67.412 ; 67.412 标准版（该行明示/表头定义） | 69.2638 锁定8D迁移后本模型荒地扫描4.0 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "+0.673(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:26 |
| Exp26 | CE / rare_crop | 56000 / 0.817376 | 66.951058 ; 66.951058 无后处理 | 68.030036 TTA8+V1e迁移; 68.759823 本模型校准扫描车辆-0.50 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "+0.212(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:27; 线上成绩记录表.md:144 |
| Exp27 | CE / rare_crop | 56000 / 0.816936 | 66.405239 ; 66.405239 无后处理 | 67.200599 TTA8+V1e迁移; 67.679726 本模型校准扫描荒地4.5 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:28; 线上成绩记录表.md:273 |
| Exp28 | CE / rare+cwfa+no_decay | 56000 / 0.812250 | 66.195521 ; 66.196 标准版（该行明示/表头定义） | 67.691 三合一校准版，精确推理参数UNKNOWN | {"parent_as_recorded": "Exp18", "delta_as_recorded": "-0.543(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:29 |
| Exp29 | CE / rare+cwfa | 56000 / 0.815372 | 66.001109 ; 66.001109 无后处理 | 67.459865 TTA8+校准 | {"parent_as_recorded": "Exp26", "delta_as_recorded": "-0.950(相对Exp26)", "delta_computed": false} | raw_master_experiment_ledger.csv:30; 线上成绩记录表.md:178 |
| Exp30 | CE / rare+no_decay | 56000 / 0.816488 | 66.584083 ; 66.584083 无后处理 | 7.926311 TTA8+错误复用B3 bias，保留灾难分 | {"parent_as_recorded": "Exp26", "delta_as_recorded": "-0.367(相对Exp26)", "delta_computed": false} | raw_master_experiment_ledger.csv:31; 线上成绩记录表.md:186; 线上成绩记录表.md:191; 线上成绩记录表.md:192; 线上成绩记录表.md:193; 线上成绩记录表.md:194; 线上成绩记录表.md:195 |
| Exp31 | CE+distill / distill(λ1.0,start1,dinov2-large) | 56000 / 0.812413 | 66.991 ; 66.991 标准版（该行明示/表头定义） | UNKNOWN | {"parent_as_recorded": "Exp18", "delta_as_recorded": "+0.252(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:32 |
| Exp32 | CE+lora / lora(r8,α16,drop.1) | 56000 / 0.726849 | 63.68103 ; 63.68103 无后处理 | 63.159382 TTA8+校准 | {"parent_as_recorded": "Exp27", "delta_as_recorded": "-2.724(相对Exp27)", "delta_computed": false} | raw_master_experiment_ledger.csv:33; 线上成绩记录表.md:286 |
| Exp33 | CE+copy_paste / copy(p0.5,ids[4,6,7],max3) | 56000 / 0.815971 | 66.427993 ; 66.427993 无后处理 | 68.775101 TTA8+B3六维校准 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "-0.312(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:34; 线上成绩记录表.md:297 |
| Exp34 | CE+mixstyle / mixstyle(p0.5,α0.1) | 56000 / 0.809144 | 67.231 ; 67.231 标准版（该行明示/表头定义） | UNKNOWN | {"parent_as_recorded": "Exp18", "delta_as_recorded": "+0.492(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:35 |
| Exp35 | CE / dp0.2, lr2e-5 | 56000 / 0.794389 | 66.832587 ; 66.832587 lr 2e-5 + dp 0.2，无后处理 | 67.972872 TTA8无校准 | {"parent_as_recorded": "Exp26", "delta_as_recorded": "-0.118(相对Exp26)", "delta_computed": false} | raw_master_experiment_ledger.csv:36; 线上成绩记录表.md:201 |
| Exp36-λ0.1 | CE+distill / λ0.1, 14000步 | 14000 / 0.772570 | 66.225255 ; 66.225255 14000步标准版 | 66.736261 14000步校准版；完整参数UNKNOWN | {"parent_as_recorded": "Exp18", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:37 |
| Exp36-λ0.3 | CE+distill / λ0.3, 14000步 | 14000 / 0.773206 | 66.043135 ; 66.043135 14000步标准版 | 66.922685 14000步校准版；完整参数UNKNOWN | {"parent_as_recorded": "Exp18", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:38 |
| Exp37-α0.05 | CE+mixstyle / α0.05, 14000步 | 14000 / 0.762026 | 66.243589 ; 66.243589 14000步标准版 | 66.634921 14000步校准版；完整参数UNKNOWN | {"parent_as_recorded": "Exp18", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:39 |
| Exp37-α0.2 | CE+mixstyle / α0.2, 14000步 | 14000 / 0.764494 | 65.841038 ; 65.841038 14000步标准版 | 66.522947 14000步校准版；完整参数UNKNOWN | {"parent_as_recorded": "Exp18", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:40 |
| Exp38_UNRESOLVED [NOT_FOUND] | UNKNOWN | UNKNOWN / UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | raw_master_experiment_ledger.csv:1-63 complete scan; 线上成绩记录表.md:1-572 complete scan |
| Exp39 | CE+distill+mixstyle / λ0.1+α0.05, 56000步 | 56000 / 0.811321(vh@51800) | 66.915682 ; 66.915682 蒸馏λ0.1+MixStyleα0.05 无后处理 | 68.725136 TTA8+B3六维校准 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "+0.177(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:41; 线上成绩记录表.md:315 |
| Exp40 | CE / decode_head=attention_fusion | 56000 / 0.805751 | 65.978999 ; 65.978999 MCCA 式 CCA+AFF 解码头替代 concat+1×1 | 67.40267 TTA8+B3六维校准 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "-0.761(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:42; 线上成绩记录表.md:332 |
| Exp41 | CE+edge_BCE / edge(r5,w0.2) | 56000 / 0.811840 | 67.053 ; 67.053 单视图无校准（模型本体） | 67.480463 TTA8无校准; 68.79692 TTA8+锁定8D迁移; 68.812 叙述记录专属校准背景-0.10/荒地3.0；仅3位小数 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "+0.313(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:43; 线上成绩记录表.md:349; 线上成绩记录表.md:355 |
| Exp42 | weighted_CE / clip.5-2.5,weights=[.5,.5196,.7483,.8871,1.6717,.5,.8316,2.5] | 56000 / 0.810649 | 66.6902 ; 66.6902 B3 + 训练集频率逆平方根权重，clip[0.5,2.5] | 68.303 TTA8+锁定校准 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "-0.049(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:44; 线上成绩记录表.md:392 |
| Exp43 | focal / γ1.0 | 56000 / 0.810164 | 66.397 ; 66.397 B3 + Focal Loss | 67.9802 TTA8+锁定校准 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "-0.343(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:45; 线上成绩记录表.md:451 |
| Exp44 | ce_dice / ce.7/dice.3 | 56000 / 0.813167 | 66.5635 ; 66.5635 B3 + CE+Dice 联合 | 68.4128 TTA8+锁定校准 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "-0.176(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:46; 线上成绩记录表.md:407 |
| Exp45 | CE+mixup / mixup(α.1,p.5,start1) | 56000 / 0.811281 | 67.0098 ; 67.0098 B3 + 温和 MixUp | 69.1399 TTA8+锁定校准 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "+0.270(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:47; 线上成绩记录表.md:452 |
| Exp46 | CE+edge+cwfa / edge(r5,w.2)+cwfa | 56000 / 0.812938 | 66.2893 ; 66.2893 B3 + 边界分支BCE + CWFA | 68.1674 TTA8+锁定校准 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "-0.450(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:48; 线上成绩记录表.md:437 |
| Exp47 | CE+semeda / semeda(λ.5,fs256,noise.1) | 56000 / 0.809569 | 67.796 ; 67.796 标准版（该行明示/表头定义） | 69.0776 TTA8+锁定校准 | {"parent_as_recorded": "Exp18", "delta_as_recorded": "+1.056(相对Exp18)", "delta_computed": false} | raw_master_experiment_ledger.csv:49 |
| stage48 | CE+MaxEnt / λ0.1 | 56000 / 0.811948 | 67.1308  | UNKNOWN | {"parent_as_recorded": "Exp23", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:50 |
| stage49 | CE+MaxEnt / λ1.0 | 56000 / 0.809339 | 67.7711  | UNKNOWN | {"parent_as_recorded": "Exp23", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:51 |
| stage50 | CE+MaxEnt / λ5.0 | 56000 / 0.809813 | 65.3961  | UNKNOWN | {"parent_as_recorded": "Exp23", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:52 |
| exp51 | CE / 全数据6796,67960步 | 67960 / 0.818171 | 66.61  | UNKNOWN | {"parent_as_recorded": "Exp23(5597)", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:53 |
| exp52 | CE+MaxEntλ1 / 6796全数据 | 67960 / 0.815931 | 66.2421  | UNKNOWN | {"parent_as_recorded": "exp51", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:54 |
| Exp53_UNRESOLVED [NOT_FOUND] | UNKNOWN | UNKNOWN / UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | raw_master_experiment_ledger.csv:1-63 complete scan; 线上成绩记录表.md:1-572 complete scan |
| exp54 | CE(锐化resume exp49) / 10000步,cwfa start5000 | 10000 / 0.803651 | 66.8485  | UNKNOWN | {"parent_as_recorded": "exp49", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:55 |
| Exp55_UNRESOLVED [NOT_FOUND] | UNKNOWN | UNKNOWN / UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | raw_master_experiment_ledger.csv:1-63 complete scan; 线上成绩记录表.md:1-572 complete scan |
| stage55b | CE / 5597真+500伪标签 | 56000 / 0.809790 | 68.048  | UNKNOWN | {"parent_as_recorded": "Exp23", "delta_as_recorded": "+0.677(相对Exp23标准67.371)", "delta_computed": false} | raw_master_experiment_ledger.csv:56 |
| exp57 | CE+RegionContrast / modeB w0.1 | 56000 / 0.810375 | 67.5182  | UNKNOWN | {"parent_as_recorded": "Exp23", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:57 |
| exp58 | CE+RegionContrast / modeC pairs[[6,0],[1,0]] w0.1 | 56000 / 0.811458 | 67.6376  | UNKNOWN | {"parent_as_recorded": "Exp23", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:58 |
| Exp59_UNRESOLVED [NOT_FOUND] | UNKNOWN | UNKNOWN / UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | raw_master_experiment_ledger.csv:1-63 complete scan; 线上成绩记录表.md:1-572 complete scan |
| Exp60_UNRESOLVED [NOT_FOUND] | UNKNOWN | UNKNOWN / UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | raw_master_experiment_ledger.csv:1-63 complete scan; 线上成绩记录表.md:1-572 complete scan |
| exp61 | CE+MaxEntλ1 / 5597真+500伪 | 56000 / 0.805786 | 67.78  | UNKNOWN | {"parent_as_recorded": "stage55b", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:59 |
| stage62 | CE(续训) / resume stage55b last.pt→70000步 | 70000 / 0.810394 | UNKNOWN | UNKNOWN | {"parent_as_recorded": "stage55b", "delta_as_recorded": "+0.05", "delta_computed": false} | raw_master_experiment_ledger.csv:60 |
| stage63 | CE / wd0.05 | 56000 / 0.812205 | 67.52  | UNKNOWN | {"parent_as_recorded": "Exp23", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:61 |
| stage64 | CE / wd0.005 | 56000 / 0.810903 | 67.61  | UNKNOWN | {"parent_as_recorded": "Exp23", "delta_as_recorded": "UNKNOWN", "delta_computed": false} | raw_master_experiment_ledger.csv:62 |
| stage65 | CE / P1第二轮3教师,84000步 | 84000 / None(训练中) | UNKNOWN | UNKNOWN | {"parent_as_recorded": "stage62等", "delta_as_recorded": "-", "delta_computed": false} | raw_master_experiment_ledger.csv:63 |
| Exp66_UNRESOLVED [NOT_FOUND] | UNKNOWN | UNKNOWN / UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | raw_master_experiment_ledger.csv:1-63 complete scan; 线上成绩记录表.md:1-572 complete scan |
| LATE_CWFA_STAGED_C7 | 成熟Exp47 SEMEDA best初始化→Late-CWFA staged fine-tune | 8000计划总步数；本行checkpoint step7000 / UNKNOWN | UNKNOWN | 67.9246 TTA8 D4 only; 69.0699 TTA8 D4+FIXED8D | {"references": {"source": {"path": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "line": 564}, "verbatim": "**对照锚点**：Exp47 raw(无TTA) 0.67796｜Exp47 历史 TTA8+locked 0.690776｜Champion Exp23+TTA8+final8D **0.6928129**｜Exp18 B3 TTA8 0.67209674。"}, "warning": "Exp47raw不能当TTA8-only严格对照；未重新做因果计算"} | 线上成绩记录表.md:555; 线上成绩记录表.md:559; 线上成绩记录表.md:560; 线上成绩记录表.md:571 |
| LATE_CWFA_STAGED_C8 | 成熟Exp47 SEMEDA best初始化→Late-CWFA staged fine-tune | 8000计划总步数；本行checkpoint step8000 / UNKNOWN | UNKNOWN | 67.8534 TTA8 D4 only; 69.121 TTA8 D4+FIXED8D | {"references": {"source": {"path": "C:\\Users\\Lenovo\\OneDrive\\Documents\\桌面\\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\\文档归档\\04_实验决策与成绩\\线上成绩记录表.md", "line": 564}, "verbatim": "**对照锚点**：Exp47 raw(无TTA) 0.67796｜Exp47 历史 TTA8+locked 0.690776｜Champion Exp23+TTA8+final8D **0.6928129**｜Exp18 B3 TTA8 0.67209674。"}, "warning": "Exp47raw不能当TTA8-only严格对照；未重新做因果计算"} | 线上成绩记录表.md:555; 线上成绩记录表.md:561; 线上成绩记录表.md:562; 线上成绩记录表.md:571 |

## 未知标准分 / 无分原因
- Exp01：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- Exp09：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- Exp10：只记录相对跌幅，基线和绝对分不闭合；禁止反推
- Exp12：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- Exp13：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- Exp15：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- Exp17：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- Exp19：服务器find无*exp19*；历史账本标为无独立线上提交，不代表当前全局状态
- Exp20：仅smoke无正式run；历史账本标为无独立线上提交，不代表当前全局状态
- Exp21a：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- Exp22：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- Exp24：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- Exp38_UNRESOLVED：指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩
- Exp53_UNRESOLVED：指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩
- Exp55_UNRESOLVED：指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩
- Exp59_UNRESOLVED：指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩
- Exp60_UNRESOLVED：指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩
- stage62：指定账本未提供绝对线上分；未用相对涨跌反推绝对分
- stage65：历史账本2026-09-09记训练中、尚未提交；不是2026-09-14实时状态
- Exp66_UNRESOLVED：指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩
- LATE_CWFA_STAGED_C7：指定账本没有staged无TTA分或local分；不能从TTA结果推断
- LATE_CWFA_STAGED_C8：指定账本没有staged无TTA分或local分；不能从TTA结果推断

## 校准来源与扫描核验
- Exp18：BASELINE_OWN_PRIOR_AND_CLASS_CALIBRATION；model_specific_scan_verified=YES_RECORDED_IN_LOCAL_LEDGER_NOT_RUNTIME_VERIFIED: 自身prior/分类校准历史；不是Stage23迁移
- Exp23：STAGE23_OWN_8D_SCAN；model_specific_scan_verified=YES_RECORDED_IN_LOCAL_LEDGER_NOT_RUNTIME_VERIFIED: 自身逐维/组合扫描，最终8D
- Exp25：LOCKED_STAGE23_VECTOR_TRANSFER_THEN_OWN_BARREN_SCAN；model_specific_scan_verified=YES_RECORDED_IN_LOCAL_LEDGER_NOT_RUNTIME_VERIFIED: 确认本模型荒地扫描，不等于全部8维重扫
- Exp26：V1E_TRANSFER_THEN_OWN_CLASS_SCANS；model_specific_scan_verified=YES_RECORDED_IN_LOCAL_LEDGER_NOT_RUNTIME_VERIFIED: 本模型多类扫描记录
- Exp27：V1E_TRANSFER_THEN_OWN_BG_BARREN_SCAN；model_specific_scan_verified=YES_RECORDED_IN_LOCAL_LEDGER_NOT_RUNTIME_VERIFIED: 本模型背景/荒地扫描记录
- Exp28：UNKNOWN；model_specific_scan_verified=UNKNOWN
- Exp29：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp30：STAGE23_B3_CWFA_VECTOR_WRONGLY_TRANSFERRED_TO_B4_NODECAY；成绩表191-195明示；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp32：UNKNOWN；model_specific_scan_verified=UNKNOWN
- Exp33：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp35：UNKNOWN；model_specific_scan_verified=UNKNOWN
- Exp36-λ0.1：UNKNOWN；model_specific_scan_verified=UNKNOWN
- Exp36-λ0.3：UNKNOWN；model_specific_scan_verified=UNKNOWN
- Exp37-α0.05：UNKNOWN；model_specific_scan_verified=UNKNOWN
- Exp37-α0.2：UNKNOWN；model_specific_scan_verified=UNKNOWN
- Exp39：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp40：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp41：LOCKED_STAGE23_VECTOR_TRANSFER_PLUS_RECORDED_OWN_BG_BARREN_SCAN；model_specific_scan_verified=YES_RECORDED_IN_LOCAL_LEDGER_NOT_RUNTIME_VERIFIED: 351迁移版；362叙述专属背景/荒地扫描，不能混同
- Exp42：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp43：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp44：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp45：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp46：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- Exp47：STAGE23_VECTOR_TRANSFER_OR_B3_LOCKED_HISTORY;具体阶段vector需推理/校准分账核对；model_specific_scan_verified=UNKNOWN;引用迁移版不证明从未做本模型扫描
- LATE_CWFA_STAGED_C7：STAGE23_FINAL_CHAMPION_VECTOR_TRANSFER_FROZEN；model_specific_scan_verified=NO_FOR_THIS_FROZEN_VARIANT_明确未改任何维；不外推全部历史
- LATE_CWFA_STAGED_C8：STAGE23_FINAL_CHAMPION_VECTOR_TRANSFER_FROZEN；model_specific_scan_verified=NO_FOR_THIS_FROZEN_VARIANT_明确未改任何维；不外推全部历史

## 关键历史冲突
- 成绩表头更新2026-08-30、锁定69.278739，与后文2026-09-01的69.281290及2026-09-10增补并存
- 待提交/在跑附录547-549是历史残留，不能用来推翻正文已记线上分
- Exp11：relative_only_evidence仍保留此前MD相对跌幅条目，但phase2现在补齐独立绝对分65.797297。
- Exp14：原CSV UNKNOWN保留；后来Exp16报告引用66.443，明确不得补成66.443162。
- Exp16：原CSV绝对分UNKNOWN保留；较新UPerNet报告顶部5-10用户提交后回填优先。该报告34-45仍留待提交模板，不是后续成绩撤销。
- Exp20：仅run_dir名称含smoke40；max_steps字段缺失，trainsteps保持原值而非自动填40。
- Exp22：成绩表写伪标签(Exp22/B3)=-0.136/-0.114，未显式将B3绑定Exp24；不强行分配精确delta或反推绝对分。
- Exp24：成绩表写伪标签(Exp22/B3)=-0.136/-0.114，未显式将B3绑定Exp24；不强行分配精确delta或反推绝对分。
- Exp25：方法称no_decay但历史行wd仍0.01；保留原字段，不能据此推断实际参数分组。
- Exp36-λ0.1：14000步短跑，只能同预算横比；parent Exp18不能证明与56000步baseline公平同口径。
- Exp36-λ0.3：14000步短跑，只能同预算横比；parent Exp18不能证明与56000步baseline公平同口径。
- Exp37-α0.05：14000步短跑，只能同预算横比；parent Exp18不能证明与56000步baseline公平同口径。
- Exp37-α0.2：14000步短跑，只能同预算横比；parent Exp18不能证明与56000步baseline公平同口径。
- Exp39：local来源写validation_history@51800；source称final-eval崩，不得将其等同最终metrics成功。
- Exp41：成绩表355明确推翻早期+0.741归因，trueNoTTA+0.313；标准包命名不能代替推理口径。
- Exp42：历史source记录attempt1 BrokenPipe，当前行指后续resume/v2；失败尝试保留为事件，不捏造独立线上分。
- Exp46：历史source记录attempt1 BrokenPipe，当前行指后续resume/v2；失败尝试保留为事件，不捏造独立线上分。
- exp51：6796全数据与原5597/1399协议不同，不应当作同协议单变量。
- exp52：6796全数据与原5597/1399协议不同，不应当作同协议单变量。
- stage55b：id=stage55b但stage/run_dir=stage56/exp56；别名同一条，不重复当两个独立模型。
- LATE_CWFA_STAGED_C7：staged8000步不是完整SEMEDA+CWFA co-training，不能自动绑定Exp66。
- LATE_CWFA_STAGED_C7：570行把0.691210与0.6928129差写成-0.00016，与原分不一致；正确算术差-0.0016029（0-1尺度），不将原笔误当精确delta。
- LATE_CWFA_STAGED_C8：staged8000步不是完整SEMEDA+CWFA co-training，不能自动绑定Exp66。
- LATE_CWFA_STAGED_C8：570行把0.691210与0.6928129差写成-0.00016，与原分不一致；正确算术差-0.0016029（0-1尺度），不将原笔误当精确delta。

## 原始来源绝对路径
- C:\Users\Lenovo\OneDrive\Documents\桌面\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\文档归档\04_实验决策与成绩\historical_audit_20260909\raw_master_experiment_ledger.csv（SHA256 20d2064f8834d3fd2ecc8f05d1a19cf0e160c8fef705f553599f7b107c1ce609）
- C:\Users\Lenovo\OneDrive\Documents\桌面\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\文档归档\04_实验决策与成绩\线上成绩记录表.md（SHA256 157524b931444a1019eeb9c9d2a5f6f7a5bb9a6fd6a3a5d99ae5f63bcd0c562e）
- C:\Users\Lenovo\OneDrive\Documents\桌面\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\文档归档\04_实验决策与成绩\exp16_upernet_online_validation_20260909\UPERNET_ONLINE_VALIDATION_REPORT.md（SHA256 2f24f0e4ff40dceef1c3e9c4dea6250d671b57e004ec47b0d9317a5b1dd436ef）
- C:\Users\Lenovo\OneDrive\Documents\桌面\Phase1_Group_Discovery_蒋鸿宇执行包_2026-08-23\历史归档\_phase2_return\phase2_calibration_table.csv（SHA256 d7e3f10035c9d675f8abe3874972603704af45e051481e49faead8bbf6ac06e0）
