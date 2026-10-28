# H75_GLOBAL_LOCAL_V1：H75 冠军基础上的整图＋局部放大双视图监督

> **交接版本**：V1.0 · 2026-10-24  
> **赛题**：2026 低空图像语义分割（半决赛）  
> **状态**：方案已确定，**尚未执行本轮新训练，也没有新的本地/线上成绩**。  
> **核心原则**：保留原始冠军、只改源域第二视图、严格匹配对照、先过证据门槛后训练。

---

## 0. 给执行队友的最简指令

**本轮只验证一个假设：在现有 H75/ACDD 框架上，把同一张源域图像的一份额外监督视图，从“重复整图”改为“均匀随机 640×640 局部裁剪并放大到 1024×1024”，能否带来可复现且能迁移到线上测试集的增益。**

- **起点必须是原始 H75 checkpoint 的独立副本**，不能用 Arm B、ACDD 原始单轨 checkpoint、CE010_REBUILT_CONTROL 或任意“近似 H75”代替。
- 不改网络、教师、伪标签、target KL、M/G 门控、Anchor、类别采样、冻结层、优化器基本协议及原冠军的推理配置。
- 核心配对实验仅两组：`DUPLICATE`（整图＋同一整图）和 `TREAT`（整图＋640 局部放大）；`CONTINUE` 仅作普通续训收益参考；原始 `H75` 永远作为冻结冠军参照。
- **只在 `TREAT` 明确胜过 `DUPLICATE` 且最终候选线上超过 71.329400% 时，才有资格讨论替换冠军。**

**先阅读现有审计文件，P0 条件不满足则直接 NO-GO，不启动训练。**

---

## 1. 既有成绩与冠军溯源（严禁混淆）

| 对象 | 定位 | 已知线上 mIoU | 本轮处理 |
|---|---|---:|---|
| **H75** | 当前线上冠军；多个同源轨迹参数加权形成的一份单独权重 | **71.329400%** | **永久冻结、不覆盖；本轮唯一合法初始化与最终比较基准** |
| ACDD_V1_ANCHOR step1000 | H75 冠军谱系的早期关键节点 | 71.158317% | 只用于说明溯源，不作为本轮初始化 |
| Arm B | 历史冠军节点 | 71.045315% | 不作为本轮初始化或主要 CONTROL |
| 新实验 CONTINUE / DUPLICATE / TREAT | 从相同 H75 原件分出的独立训练分支 | **暂无** | 本轮待实施 |

### H75 的已知构造与身份

已归档资料中的 H75 参数构造为：

\[
\theta_{H75}=0.75\theta_C+\frac{\theta_{R05}+\theta_{R06}+\theta_{R07}}{12}
\]

仅对既定 `stage2/stage3/decode_head` 可训练参数做加权；冻结参数、buffers、metadata 按 Champion 处理。**H75 已产生，不能因为本轮要续训而再次进行平均或重建。**

需要在开始前实测并记录 H75 原件 SHA256：

```text
b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a
```

> 若本机找到的文件校验值不同：**立即停止**；先做只读溯源，不得将其他 checkpoint 重命名为 H75，也不得改写该 SHA 记录。

正式推理协议以 H75 已经核验的归档为准：**单一模型结构、单一 checkpoint、D4 TTA8、FP32 平均、Bstar 执行一次、输出 1024 分辨率**。这里的 D4 TTA8 是同一模型的多增强推理，不是多模型预测融合。所有新候选必须按与 H75 **完全相同**的配置评估和打包。

---

## 2. 为什么选这条路线，而不是重复旧方案

只读审计已经发现 ACDD/H75 谱系包含：

1. `RareClassCropper` 的调用逻辑，但归档设置中 `1024` 输入配 `1024` crop 时实际退化为整图，**不能当作有效的局部视角监督**。
2. 源域已有一定几何/颜色增强；归档中 `domain_aug` 关闭，不能把它描述成“已验证有效的局部多尺度”。
3. target 已有整图 `D4 + strong_photo`、donor gate、non-gate Anchor、固定教师软蒸馏机制。
4. 部分层冻结与 H75 参数平均早已存在，不属于本轮新机制。

因此，**泛泛地“增加随机裁剪、稀有类采样、上下文增强、目标域伪标签、D4、HRDA”并不是本轮值得再做的方案**：有些已做过，有些会一次引入多个变量，无法判断真实收益。

本轮唯一需要验证的新增机制，是：

> **同一个 source 样本、相同真实标签来源、相同更新步骤下，将第二个有监督输入由整图改为均匀随机局部窗口，带来额外的对象尺度视角。**

`640 → 1024` 对被裁剪物体提供约 `1.6×` 的线性放大、约 `2.56×` 的像素面积放大，但**不会增加图像原本不存在的信息**，并且会丢掉窗口外上下文；收益是否存在必须靠对照验证。它是受限的“source 双视图尺度监督”，**不宣称为完整 HRDA 或新型教师蒸馏**。

---

## 3. P0：训练前的强制只读证据门槛

### P0.1 确认没有等价实现

复查归档配置与**实际生效代码路径**，证明现有 H75/ACDD 并不存在与本方案等价的：

- 同一 source 图像的一次整图 GT 监督；
- 加上第二次 **640 局部放大后的独立 GT CE**；
- 两项在同一次 optimizer update 中按固定权重合成。

**配置出现某个函数名不等于实际执行。**要给出分支调用、实际参数和一次可检查的样例。若发现已存在等价方案且实际启用，本方案立即 NO-GO。

### P0.2 确认局部尺度确有诊断价值

只用有标签 source 数据及合法诊断信息，输出：

- 小目标/窄目标（尤其 `Vehicle` 等）在原始 `1024` 视图里的像素尺寸、占比、边界特征；
- 相关类别与骨干/预测 stride 的实际分辨率瓶颈；
- 采样到的若干 `640` 窗口与原图的 GT 对照，确认裁剪后标签精确对应；
- 可能因丢失上下文受损的类（例如 Building、Road、森林、农田等）的样例。

这里是**判断方案是否有理由实验**，不是拿 source1399 的小幅涨分证明线上肯定提升。若无可解释的目标尺度瓶颈，或者现有证据显示局部上下文损失主导，则 NO-GO。

### P0.3 锁定不变量与实验身份

生成只读审计表：原始 checkpoint SHA、源/目标数据清单与 hash、实际教师和 cache、M/G 逻辑、loss 公式、源域系数、Anchor 系数、冻结参数列表、优化器/调度器、AMP、batch、推理配置和提交脚本。

尤其检查历史上曾出现的 **loss 系数代码与配置文字不一致** 问题；以实际运行图为准。若发现不一致，只报告并停止，不得借本轮实验暗中修复（否则不再是单变量实验）。

**P0 交付**：`P0_READONLY_AUDIT.md`、`P0_SCALE_DIAG.md`、`P0_GO_NO_GO.json`。仅 `GO` 可进入实施。

---

## 4. 实验分组：H75、CONTINUE、DUPLICATE、TREAT

| 分组 | 起始权重 | 单训练循环中的 source 监督 | 保持不变的内容 | 目的 |
|---|---|---|---|---|
| **H75_REF** | 原始 H75 | 无新训练 | 原始推理及线上成绩 | 最终绝对门槛，不参与重训 |
| **CONTINUE** | H75 独立副本 | 既有整图 source GT 监督 | ACDD 全部既有训练机制 | 测算“仅继续训练”的贡献 |
| **DUPLICATE** | H75 独立副本 | 整图视图 A ＋ 相同整图视图 B | 和 TREAT 完全同预算 | **最重要的算力/双 CE 对照** |
| **TREAT** | H75 独立副本 | 整图视图 A ＋ `640→1024` 局部视图 B | 唯一变更是第二视图的空间尺度 | 检验局部放大的净增益 |

**实验解释的优先级**：

1. **主要因果比较：`TREAT − DUPLICATE`。**如果差异不为正，不能说局部视图有效。
2. **工程价值比较：`TREAT − CONTINUE`。**如果效果仅与普通续训相当，增加算力并无充分理由。
3. **最终冠军比较：`TREAT − H75_REF`。**即使前两项提高，没过原 H75 线上 71.329400% 也不能替换冠军。

`CONTINUE` 和 `DUPLICATE` 的作用不同，不得以某个重建的“CONTROL”代替 H75 原件；也不得只跑 TREAT、不跑 DUPLICATE 就声称证明了方法有效。

---

## 5. 唯一方法变化：source 双视图

### 5.1 数据流

沿用原 ACDD 已授权数据和划分：`source=5597` 张有标签训练图像，`target=1300` 张无标签图像；**不改变样本列表、标签含义、教师推理方式或目标域训练逻辑**。

对每个 source 图像 `I` 与真实分割标签 `Y`：

**视图 A（全局）**：

- 延用原来实际生效的 source 整图预处理路径，保留 `1024×1024` 的整体上下文；
- 继续以与该视图严格对齐的 source 真值计算原有监督 CE。

**视图 B（局部）**：

- 从同一张 `I` 与同一张 `Y` 取一对**同坐标**的 `640×640` 窗口；
- 位置均匀采样且合法；对于原始 `1024×1024` 像素坐标，可用 `x0,y0 ∈ {0,…,384}`，取 `[x0:x0+640, y0:y0+640]`；
- 将 RGB 裁剪结果通过 **bilinear** resize 为 `1024×1024`；将分割标签通过 **nearest-neighbor** resize 到 `1024×1024`，不得插入非整数类别；
- 该标签仍是 source **真实 GT**，不是教师伪标签；对 ignore 区域继续按原代码跳过；
- **均匀随机**选窗口，不使用类别偏置、RareClassCropper 的类别筛选、混淆热区或 target 伪标签选框；
- 确保裁剪所处阶段与原 source 图像/标签几何增强的坐标严格一致，不得使 GT 错位；裁剪 RNG 使用独立、可复现的随机流。

视图 B **只给学生提供第二份 source 真值监督**；教师缓存、target 的 D4/strong_photo、pB/pD/M/G、非门控 Anchor 都不因视图 B 改变。

### 5.2 三组 source 监督公式

设 `CE(·)` 表示原项目的 source GT 像素交叉熵，**每个视图单独按其有效 GT 像素数取 mean**（不能先把两视图的有效像素混在一起重新归一化）。固定：

\[
\alpha=0.25,\quad 1-\alpha=0.75
\]

- `CONTINUE`：`L_source = CE(whole)`。
- `DUPLICATE`：`L_source = 0.75·CE(whole_A) + 0.25·CE(whole_B)`，其中 `whole_B` 的**输入视图与 whole_A 相同**，但作为独立前向完成，以匹配 TREAT 的两次 source 前向预算。
- `TREAT`：`L_source = 0.75·CE(whole) + 0.25·CE(local_640_to_1024)`。

保留 ACDD 既有的 source 总系数，原则上的总损失为：

\[
\boxed{L_{total}=0.1L_{source}+L_{target}^{\mathrm{existing}}+0.15L_{anchor}^{\mathrm{existing}}}
\]

这里的 `L_target` 指现有 gate/KL/teacher 路径聚合后的结果，不允许为本实验重构。由此 TREAT 的两项 source CE 在总损失中的权重分别为 **`0.075` 与 `0.025`**，而不是在现有 `0.1` 之外再附加一个新 `0.25 CE`。

**注意**：最终实现必须先逐行对照当前归档的实际 loss 计算。如果历史训练版本中系数有实现偏差，不能擅自“修好”后宣称只改一个变量；应先出具差异报告，再决定 GO/NO-GO。

### 5.3 明确禁止的伴随改动

不得同时新增或改变：

- 类别权重、focal/dice 辅助损失、边界损失、对比损失、跨视图 consistency loss；
- donor 替换、双教师策略、teacher 更新、target 局部伪标签、目标域选框；
- 模型结构、解码头、额外高分辨率分支、BN 重校准；
- source/target 数据比例、rare-class sampling、训练样本次序策略；
- Anchor 系数、target KL 系数、阈值、M/G gate、ramp、冻结参数范围；
- 原始 Bstar 校准、D4/TTA 参数或任何新的推理后处理。

---

## 6. 训练预算、等价性和复现要求

### 6.1 本轮固定预算

本轮执行约定为**固定 1000 次训练循环（attempted iterations）**；保存 `250 / 500 / 1000` 的节点快照，并同时记录每个节点的**真实成功 optimizer update 数**。此 `1000` 是本轮的实验预算，**不是声称 H75 原始官方训练周期等于 1000**。

`CONTINUE / DUPLICATE / TREAT` 从相同 H75 权重、相同起始训练状态约定开始；沿用已核验的 AdamW、LR 调度、分母 `10000`、原 `ramp500`、batch=1、AMP、冻结范围与 train/eval 模式。**不可为了短实验把 scheduler 分母从 10000 改成 1000，或把 ramp500 删除。**

如果 H75 原件不含有效 optimizer/scaler/scheduler state，三组必须按**同一份明确记录的新优化器初始化规则**建立，不得从 Arm B 或其他模型借来 optimizer state。若任何一组发生 AMP step skip，必须报告 attempted/effective update 的差异；未经更新步数可比性核验，不能比较为等预算实验。

### 6.2 控制变量与随机种子

- 首轮使用同一个预先固定并公开记录的 seed，在三组之间复用同一 H75 初始化、source 样本顺序、target 样本顺序、训练终点和 checkpoint 时点。
- TREAT 的窗口位置使用独立 RNG；除视图 B 的裁剪/resize 之外，不应引入额外随机变化。
- `DUPLICATE` 中两个 whole 输入应当在模型输入层面实际相同；若模型内部 dropout 等使两次 forward 不完全数值一致，应记录其影响，不改模型来掩盖。
- 若第一组实验确实通过初步门槛，再使用**第二组预先冻结的种子**成对复验 `DUPLICATE` 与 `TREAT`；若效应方向反转或不稳定，不做参数搜索，先停止。

### 6.3 前向、反向和更新的对应关系

`DUPLICATE` 与 `TREAT` 每个训练循环均须：

1. 取 **同一 source ID** 与同一 target batch；
2. 运行 source 视图 A 与 B 的两次监督前向；
3. target 教师与 KL、Anchor 走原始路径；
4. 按上式合成单个总 loss；
5. 完成常规反传，保持原有梯度累积/裁剪语义（若存在）；
6. **最多一次** `optimizer.step()` / `scaler.step()` 以及既有的调度更新。

不能把“两个 source 视图”实现成“多做一次梯度更新”，也不能因显存不足只给 TREAT 用特殊梯度裁剪或冻结策略。显存不够时先报告阻塞问题；如果采取通用内存优化，两组必须一致且证明等价。

---

## 7. 落地实施顺序与验收条件

### 阶段 A：只读审计与 GO/NO-GO

- 阅读现有 `H75_EXISTING_MECHANISMS_AUDIT.md` 与 `NEXT_STEP_DELTA_PROPOSAL.md`；核对原始 H75、配置、生效代码、教师、数据、推理协议。
- 完成上述 P0.1～P0.3 的证据文件；证明局部 GT 与整图 GT 坐标匹配，现有机制不包含等价双视图。
- 先写明本轮唯一代码 diff、固定实验参数和训练预算，并冻结版本、seed、起始 SHA。
- **只有证据完整且 GO 时才允许执行下一阶段。**

参考已有审计位置（队友按实际权限核实，不要据此猜测服务器路径）：

```text
D:/西财/大三上/--B算法挑战赛/半决赛/20261024/H75_readonly_audit_20261024/
  H75_EXISTING_MECHANISMS_AUDIT.md
  NEXT_STEP_DELTA_PROPOSAL.md
```

### 阶段 B：代码最小差异与 smoke test

- 从 H75 正式实现复制出**新的独立实验目录**；原代码、原始 checkpoint、既有缓存保持只读。
- 将 source 第二视图模式封装为 `duplicate_whole` / `local640_resize1024` 开关；CONTINUE 使用原单视图路径。
- 构建单元测试：
  - RGB/GT 窗口坐标一致；RGB 双线性、mask 最近邻；mask 值仅合法类别/ignore；
  - DUPLICATE 两个 source 输入完全相同；
  - `alpha=0.25` 下有效 GT mean 的归一化正确；
  - 源 loss 总系数始终 `.1`，Anchor `.15`，其余 target 路径不变；
  - 不额外触发 optimizer.step，梯度在两组都可正常反传；
  - 两组的模型参数冻结列表和初始 SHA 相同。
- 用很短的 smoke test 检查无 NaN、OOM、数据错位、teacher cache miss、AMP scaler 异常；smoke 结果**不能当作正式成绩**。

### 阶段 C：配对实验

- 固定主 seed，运行 `DUPLICATE` 与 `TREAT` 的 `1000` 循环，保存 `250/500/1000`。如缺乏可比的同 H75 起点普通续训，额外运行 `CONTINUE`。
- 每组输出配置和代码哈希、checkpoint SHA、数据顺序摘要、target 命中与 gate 统计、loss 曲线、GradScaler skip、成功更新数、显存峰值、运行时间。
- **不要在观察中途结果后修改 alpha、crop size、学习率、训练时长或停止节点；本轮不扫超参。**

### 阶段 D：本地验证与复验

固定同一 source1399（`H300+R1099`）及 H75 原有评估协议，逐 checkpoint 对照：

- 总体 mIoU（百分比），全部 8 类 IoU；
- `TREAT-DUPLICATE`、`TREAT-CONTINUE`、`TREAT-H75` 的 **pp 差值**；
- Vehicle、荒地（Barren）、农田（Agricultural）等既往薄弱类；
- Building、Road、森林（Vegetation）等上下文敏感类的退化风险；
- 根据实际预测检查局部小目标是否改善，同时排查碎片化与边界错误。

若首轮证据支持可复现收益，**使用第二 seed 完成匹配复验**；未通过不继续扩展到不同 crop/alpha/新 teacher。

**注意**：source1399 只是本地代理指标。**不能因为本地领先，就宣布超过 71.329400% 的线上冠军**。

### 阶段 E：线上提交门槛

仅在 `TREAT` 已通过本地配对验证、第二 seed 方向一致、关键类别没有无法接受的下降且候选确有价值时，使用**完全相同的 H75 单模型 D4-TTA8 + FP32 mean + Bstar once** 生成 test2 提交文件。先核验：

- test2 应有的 `1300/1300` 文件完整；
- 提交为**一个模型、一份 checkpoint**，无双模型预测融合；
- 内部 `0~7` 到提交 `1~8` 的标签映射正确；ignore 处理遵守原提交协议；
- PNG 单通道、分辨率 `1024×1024`，ZIP 根目录放置规定 PNG；
- 输出、checkpoint、config、推理脚本与 ZIP 全部记录哈希；
- **先打包和审阅，不经队长确认不得自动上传比赛平台**。

如果官方仍未明确允许未经继续训练的权重平均提交，需将这项历史合规不确定性单独报告；**不能把“最终只有单模型”当作已经获得官方批准**，也不能用本轮续训自动推断其合规。

---

## 8. 停止条件（任何一条满足即可终止当前路线）

| 阶段 | 停止触发条件 | 处理 |
|---|---|---|
| P0 | 无法证明 H75 起点、SHA/数据/教师/推理身份；或者原方案已有等价的双 CE 局部视图 | NO-GO，不训练 |
| P0 | 未发现可信的小目标尺度瓶颈，或局部上下文损失过强 | NO-GO，不训练 |
| 工程 | GT 错位、标签插值错误、两组优化步数或系数不一致、未知配置变更 | 停止，归因修复，不把结果计入正式实验 |
| 本地 | **TREAT 不优于 DUPLICATE** | 不调 alpha/crop、不加新模块，停止本路线 |
| 复验 | 不同 seed 的 `TREAT-DUPLICATE` 方向不一致 | 停止，不宣称稳定方法收益 |
| 分类别 | 关键上下文类明显恶化，整体收益不能合理抵消 | 停止或标记失败，不直接提交 |
| 泛化 | 只在 source1399 涨分，线上未提升 | 不替换 H75，停止扩展 |
| 最终 | 候选线上 **≤71.329400%** | H75 继续为冠军；保留日志用于复盘 |

> 本方案不是“必须完成所有阶段”的长实验。**优先证伪，越早出现 NO-GO 越早止损**。不要为了形成正结果而事后调参。

---

## 9. 队友须交付的文件和结果

在自己有权限的实验目录创建一个新目录，例如：

```text
H75_GLOBAL_LOCAL_V1_20261024/
├── P0_READONLY_AUDIT.md
├── P0_SCALE_DIAG.md
├── P0_GO_NO_GO.json
├── ORIGINAL_H75_IDENTITY.json
├── CONFIG_DIFF.md
├── configs/
│   ├── continue.*
│   ├── duplicate.*
│   └── treat.*
├── tests/
│   └── source_view_alignment_report.md
├── runs/
│   ├── continue_seed_<seed>/
│   ├── duplicate_seed_<seed>/
│   └── treat_seed_<seed>/
├── evaluation/
│   ├── source1399_per_class.csv
│   ├── matched_delta.csv
│   └── seed_consistency.md
├── submit_candidate/          # 仅通过门槛后创建
├── REPORT.md
└── RUN_SUMMARY.json
```

以上是**建议组织结构而不是已存在的服务器路径**；由队友在经过授权的工作目录内落地。任何涉及其他用户的路径、模型、缓存、环境、配置或 checkpoint 均不允许读取、复制或引用。

### `REPORT.md` 必须回答

1. 是否通过 P0？证据链接在哪里？
2. H75 原件 SHA 与冠军成绩是否确认？有无改动其原文件？
3. 相比原版 ACDD/H75，**唯一代码差异**是哪几行、运行时是否生效？
4. `DUPLICATE` 与 `TREAT` 的数据顺序、seed、更新步数和 loss 权重是否真正匹配？
5. `250/500/1000` 各节点本地 source1399 总体及 8 类 IoU 各是多少？
6. `TREAT-DUPLICATE`、`TREAT-CONTINUE`、`TREAT-H75` 分别是多少 pp？
7. 第二 seed 的方向是否一致？是否有上下文类退化？
8. 有无任何 AMP skip、OOM、缓存错误、GT 错位或未授权变化？
9. 是否符合继续打包和线上提交的门槛？最终是否**超过 H75 线上 71.329400%**？
10. 结论必须写为 `PASS / FAIL / BLOCKED`；若失败，说明理由，**不请求擅自展开新一轮参数搜索**。

### `RUN_SUMMARY.json` 建议最小字段

```json
{
  "experiment": "H75_GLOBAL_LOCAL_V1",
  "status": "NOT_STARTED",
  "baseline": "H75",
  "baseline_sha256_expected": "b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a",
  "baseline_online_miou_percent": 71.329400,
  "p0_decision": null,
  "crop_size": 640,
  "local_resize_size": 1024,
  "local_sampling": "uniform_valid_window",
  "source_alpha_local": 0.25,
  "source_total_coeff": 0.1,
  "anchor_coeff": 0.15,
  "planned_attempted_iterations": 1000,
  "checkpoint_attempted_iters": [250, 500, 1000],
  "groups": ["CONTINUE", "DUPLICATE", "TREAT"],
  "primary_comparison": "TREAT_MINUS_DUPLICATE",
  "candidate_online_miou_percent": null,
  "replaces_h75": false
}
```

本 JSON 中的 `null`/`NOT_STARTED` 表示**尚未执行或未取得结果**，队友不能在事前填虚构成绩。实际运行还应追加两个或多个 seed 的完整日志、checkpoint 哈希、成功更新数、服务器授权 GPU、代码版本和产物校验。

---

## 10. 不允许队友擅自扩大范围的事项

- 不回头做 B1-V2、P2C 开关扫描或 alpha/LR/clip 搜索；B1 的历史差异极小且已止损。
- 不因 H75 当前来自参数平均就再平均多个 TREAT checkpoint，亦不进行多模型预测融合。
- 不用 `Mask2Former + Swin-L` 教师升级来替代本次学生局部尺度检验；那是另一条独立研究线，不能混入本轮实验。
- 不以 A05、Arm B、CE010 重训 CONTROL 的结果替代 H75 冠军比较。
- 不用没有 GT 的 test2 预测自证 mIoU；本地与线上指标分别报告。
- 不访问未获授权的其他用户文件或路径；不覆盖原始模型或历史结果。

---

## 11. 最终决策规则

**有证据 GO → 匹配实验 → 二次 seed → 本地检查 → 候选审核 → 有条件线上提交 → 与 H75 比较。**

本轮的成功定义同时包括：

1. `TREAT-DUPLICATE` 在配对实验中为正，且复验方向稳定；
2. 收益没有以不可接受的关键上下文类别下降为代价；
3. 推理协议与 H75 完全一致、模型单 checkpoint 合规问题已确认；
4. **最终线上 mIoU 严格高于 71.329400%**，才允许提出新冠军替换建议。

如果实验机制有效但线上未超过冠军，可以保留作为研究结论，**不能修改“当前冠军 H75”的正式记录**。

**交接总结**：本轮不是泛泛“多尺度增强”，而是一次从原始 H75 开始的、受严格对照约束的 **source global/local 真标签双视图尺度实验**。只动第二份 source 监督视图，其他核心机制与推理冻结；用 `DUPLICATE` 排除双前向收益，用 `CONTINUE` 排除普通续训收益，用原始 H75 的 `71.329400%` 决定是否真正有提交价值。
