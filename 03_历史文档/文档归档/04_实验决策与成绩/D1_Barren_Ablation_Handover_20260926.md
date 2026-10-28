# D1 去除目标域荒地正伪标签：实验交接总结（含线上结果补记）

日期：2026-09-26  
状态：两组训练、本地评估、test2 推理、候选打包及一次线上评分均已完成。  
证据边界：线上分数由用户在会话中回报；当前没有平台提交 ID、提交时间或截图，证据级别为 `USER_REPORTED`。精确配置、脚本及日志仍以服务器实验目录为准。

## 1. 最终结论

在原始 D1 上，仅取消目标域被教师标为 Barren（荒地）的有效正伪标签像素的 hard CE 监督，保留 source 荒地真值监督及其他配置。匹配 source 对照中：

- 合并集单模型 mIoU 提升 **0.2541pp**；
- Barren IoU 提升 **1.8659pp**；
- 沿用旧 parent/new logits alpha=0.5 融合方式后，source 合并 mIoU 提升 **0.1417pp**。

但是，最终生成的双 checkpoint logits 融合 ZIP 在线上 test2 仅得：

```text
70.506962
```

它相对：

- 当前合规冠军 A05_WEIGHT（70.775011）：**−0.268049pp**；
- D1_2k 单 checkpoint（70.736134）：**−0.229172pp**；
- 旧非合规 D1_alpha050 teacher（70.763755）：**−0.256793pp**；
- theta278（70.586921）：**−0.079959pp**。

因此本路线表现为：

> **source holdout 上明显正收益，但 test2 线上总分显著下降。**

同时，scored ZIP 是 parent/new 两个 checkpoint 的 raw logits alpha=0.5 融合，按现行单-checkpoint规则不合规。该分数只作为消融实验结果保存，**不参与合规冠军排序，也不能记成单个 D1_NO_BARREN_POS checkpoint 的线上分数**。

## 2. 为什么做这个实验

本轮从“按类别查看 IoU，针对薄弱类别改进，同时保护其他类别”出发，先审计评估协议，再分析逐类混淆。

- 正确屏蔽 ignore=255 后，原先 Vehicle IoU 约29.3%的结论被推翻：冠军 Vehicle IoU 实际约81.73%。旧脚本把 ignore 错当成 Vehicle，因此不能据此加强车辆损失。
- 冠军 Barren IoU 约51.83%，Precision约55.42%、Recall约88.88%，FP约21.72M、FN约3.38M，是明显的误报主导问题。
- Background 主要是漏检，部分被误判为 Barren；Agricultural 在原 D1 中有收益，应列为保护类别。
- D1 原本已包含弱教师伪标签、强光度增强和随机 D4 学生监督，因此不再优先重复加入泛化的“强弱增强一致性”。

待检验假设：

> 目标域 Barren 正伪标签监督可能推动荒地预测扩张；去掉这部分监督，可能降低 FP，改善荒地 IoU。

这是一项基于诊断的定向损失消融，不是新增网络模块。

## 3. 基础方案与唯一改动

### 3.1 模型身份

- `theta278_export`：原 D1 的父模型。
- `f_s2p100_s3p125.pt`：另一正式基准；审计发现它与上述父模型有402个张量不同，不能混用。
- 原 D1：本轮要复现的训练方案。
- `D1_alpha050`：旧的**非合规 logits 融合 teacher 行为**，不是参数插值模型。
- `CONTROL`：按原 D1 配置重新训练的匹配对照。
- `D1_NO_BARREN_POS`：本轮唯一改动的实验组单 checkpoint。
- `D1_NO_BARREN_POS_alpha050_test2.zip`：parent/new 双 checkpoint raw-logits融合包；其线上分不等于单个实验组 checkpoint 的分数。

### 3.2 对照设计

| 项目 | CONTROL | D1_NO_BARREN_POS |
|---|---|---|
| 初始化、优化器、可训练层 | 原 D1 配置 | 相同 |
| 训练长度 | 2000 loop steps | 2000 loop steps |
| source 真值监督，包括 Barren | 保留 | 保留 |
| target 非 Barren 有效伪标签 hard CE | 保留 | 保留 |
| target 教师标为 Barren 的有效正伪标签 hard CE | 保留 | 排除 |
| hard CE 归一化分母 | 原 `mh.sum()` | 仍为原 `mh.sum()` |
| source/target 采样、D4视图 | 固定匹配序列 | 与 CONTROL 一致 |
| 增强、其他损失和评估协议 | 原方案 | 不变 |

实现含义：只让原本有效且 `y278 == Barren` 的像素不贡献 target hard CE 分子；不能把这些像素改成 Background，也不能用删除后的有效像素数重新计算分母。保留原分母避免额外放大其他七类像素监督权重；同时总 target hard CE 强度有所降低，这是干预的一部分。

实验组累计移除 Barren 正伪标签像素 **869,978**。这是训练过程累计计数，不代表互不重复的独立像素。

### 3.3 最终 scored ZIP 的融合方式与合规状态

令 `z_parent` 为 theta278_export raw logits，`z_new` 为新实验模型 raw logits：

```text
z_fused = z_parent + 0.5 × (z_new - z_parent)
        = 0.5 × z_parent + 0.5 × z_new

D4 TTA8 / FP32 view mean
→ 融合 raw logits
→ 加一次 Bstar
→ bilinear 上采样，align_corners=False
→ argmax并按提交协议编码
```

不改 alpha，不重新调 bias，不重复叠加 Bstar。

该流程在推理时使用两个 checkpoint 的 logits，按现行规则属于：

```text
NON_COMPLIANT_INFERENCE_TIME_MULTI_CHECKPOINT_LOGIT_FUSION
```

## 4. 执行与可比性

- 两组均完成2000/2000个训练迭代，实际 optimizer 更新均为 **1999次**，AMP skip均为 `[359]`。不能写成“2000次成功参数更新”。
- 两组 source/target 图像和D4视图逐步一致。
- `matched_run_verification.json` 与 `loss_contract_test.json` 核验了预定损失差异。
- CONTROL 对旧D1：合并mIoU差异约 **−0.0002pp**，最大逐类IoU绝对差异约 **0.0008pp**，复现一致性良好。
- 本次使用物理 GPU0，UUID `GPU-bc304d79-c807-968a-eaad-d2c2ecc5a984`，程序内部映射 `cuda:0`。

GPU0的使用属于本次已经发生的历史执行事实；后续本项目仍应遵守当前 GPU2-only约束，不得据此延伸授权。

## 5. 本地结果

所有Δ均为百分点（pp）。合并集指标来自合并混淆矩阵，不是两个集合mIoU简单平均。

### 5.1 单模型：实验组对匹配 CONTROL

| 集合 | CONTROL mIoU | 实验组 mIoU | ΔmIoU | Barren IoU Δ | Barren Precision Δ | Barren Recall Δ |
|---|---:|---:|---:|---:|---:|---:|
| H300 | 72.6913% | 72.9495% | +0.2582 | +1.4313 | +2.7136 | −2.4919 |
| R1099 | 81.1173% | 81.3972% | +0.2799 | +2.1481 | +2.5617 | −0.3966 |
| 合并 | 79.1026% | 79.3567% | +0.2541 | +1.8659 | +2.6812 | −1.2848 |

合并集 Barren FP减少 **2,584,457**，TP减少 **390,285**、FN相应增加；其中 Background→Barren错误减少 **1,994,807**。

因此不是荒地所有方面都变好，而是减少大量误报，同时增加部分漏检，最终source IoU净改善。

### 5.2 单模型：全部类别的合并集变化

| 类别 | CONTROL IoU | 实验组 IoU | ΔIoU（pp） |
|---|---:|---:|---:|
| Background | 71.7436% | 72.0008% | +0.2572 |
| Building | 87.9864% | 87.9748% | −0.0116 |
| Road | 83.9533% | 83.9169% | −0.0364 |
| Water | 89.5664% | 89.5722% | +0.0058 |
| Barren | 51.2272% | 53.0930% | +1.8659 |
| Vegetation | 89.6242% | 89.6073% | −0.0169 |
| Agricultural | 76.9191% | 76.9303% | +0.0112 |
| Vehicle | 81.8008% | 81.7586% | −0.0422 |

### 5.3 融合候选：相对旧非合规 D1_alpha050 teacher

| 集合 | 旧 teacher mIoU | 新融合 mIoU | ΔmIoU | Barren IoU Δ | Background IoU Δ | Agricultural IoU Δ |
|---|---:|---:|---:|---:|---:|---:|
| H300 | 72.9435% | 73.0619% | +0.1184 | +0.6223 | +0.3232 | −0.0120 |
| R1099 | 81.1409% | 81.3203% | +0.1794 | +1.3008 | +0.1221 | +0.0881 |
| 合并 | 79.2035% | 79.3452% | +0.1417 | +1.0090 | +0.1567 | +0.0284 |

source合并 Barren IoU：**51.8293% → 52.8383%**。两个source子集方向一致，但没有迁移为test2线上总分提升。

## 6. 线上结果与解释

### 6.1 结构化结果

| 项目 | 记录 |
|---|---|
| 文件 | `D1_NO_BARREN_POS_alpha050_test2.zip` |
| 文件 SHA256 | `c494d46a8db85e25f244c2a177deab00810da9c44af8ebcc55a1573a86e03d20` |
| 用户回报 test2 mIoU | **70.506962** |
| 证据级别 | `USER_REPORTED` |
| 平台提交 ID | `null`（待补一手回执） |
| 平台提交时间 | `null`（待补一手回执） |
| 平台截图 | `null`（待补） |
| 推理构成 | parent/new 双 checkpoint raw-logits alpha=0.5 |
| 合规状态 | 非合规 |
| 冠军状态 | 否；仅实验记录 |

### 6.2 已有证据支持什么

1. 在匹配source训练条件下，取消target Barren正伪标签hard CE显著降低荒地误报并提高source Barren IoU。
2. 干预符合预期机制：Background→Barren错误减少。
3. CONTROL复现良好，source对照中的干预差异可归因性较强。
4. 但是这种source收益**没有迁移到test2线上总分**。

### 6.3 不能声称什么

1. 不能证明被移除的Barren伪标签都是错误标签。
2. 不能证明test2下降完全来自Barren recall；test2没有GT。
3. 不能把source holdout逐类改善当作线上方向选择器。
4. 不能把70.506962归因于单个D1_NO_BARREN_POS checkpoint，因为scored包是双checkpoint融合。
5. 不能把该包列入合规冠军候选。

### 6.4 与提交前门槛的关系

原自动建议是不消耗线上机会，因为预设“单模型 Barren recall下降不超过1pp”未通过；合并下降 **1.2848pp**，H300下降 **2.4919pp**。此次后来进行的提交属于人工例外验证，不能反向写成自动gate曾通过。

线上负结果进一步说明：

> source holdout上按类定向改善，即使H300和R1099一致为正，也不能可靠预测test2。

## 7. 故障与恢复记录

- test2阶段曾因缓存元数据校验退出：脚本只接受字面量`raw/no Bstar`，实际缓存使用`no Bstar`，另有`stored`字段表明raw logits。
- 修复仅把连写文本检查拆为：`protocol`包含`no Bstar`、`stored`包含`raw`；仍检查D4 TTA8、FP32 view mean、checkpoint SHA和1300个缓存文件。
- 未重新训练、未重新跑source评估。
- 恢复仅补跑缺失的test2推理及候选流程。
- 原日志与失败目录保留，修复diff为`RECOVERY_DIFF.patch`。
- 完成时间：2026-09-26 18:27:16（北京时间）。

## 8. 产物与服务器路径

实验根：

```text
/home/jhy/rare_diag_20260915/d1_barren_positive_ablation_gpu0_20260926
```

| 文件 | 用途 |
|---|---|
| `RUN_SUMMARY.md` | 原始结果及自动判定 |
| `status.json` | 完成状态 |
| `evaluation/per_class_metrics.csv` | 逐类指标 |
| `evaluation/confusion_matrices.json` | 混淆矩阵 |
| `matched_run_verification.json` | 匹配训练核验 |
| `loss_contract_test.json` | 损失修改核验 |
| `test2/test2_behavior.json` | test2无标签行为 |
| `test2/D1_NO_BARREN_POS_alpha050_test2.zip` | 已评分的非合规融合实验包 |
| `d1_barren_positive_ablation_audit.zip` | 审核包，不用于比赛提交 |
| `ARTIFACT_HASHES.json` | 文件与哈希 |
| `RECOVERY_DIFF.patch` | 恢复阶段最小修复 |
| `diagnostic_correction/` | 诊断补正材料 |

checkpoint SHA256：

```text
CONTROL:
53550ab12509dddb79daf7b8600c428a68762b950e149d8aca6cfd912caa214c

D1_NO_BARREN_POS:
9f55284339fe7a11de69922866ecaea553f0125a696bb9cdec2702a7b17ece5e
```

ZIP SHA256：

```text
c494d46a8db85e25f244c2a177deab00810da9c44af8ebcc55a1573a86e03d20
```

本地归档：

```text
提交包/批次_20260926_A05_D1Barren/D1_NO_BARREN_POS_alpha050_test2.zip
```

## 9. 后续决定

1. **停止继续加强Barren抑制。** 线上结果已经否定“source Barren改善可直接迁移为test2提分”的当前实现。
2. 不围绕70.506962继续调alpha、Bstar或阈值，因为它本身还是非合规双checkpoint融合。
3. 保留本次为关键反例：source H300/R1099及合并集均正，不代表test2线上正。
4. 当前test2合规冠军保持 A05_WEIGHT `70.775011`。
5. 若未来重新研究Barren，必须先提出新的target-domain证据，不能只依据source confusion；且最终推理必须单checkpoint。
