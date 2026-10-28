# `uav_semantic_segmentation` 远程仓库完整审计与工作流接入记录

- 审计日期：2026-10-09
- 仓库：`https://github.com/2856643972xsb/uav_semantic_segmentation.git`
- 远程分支与冻结 commit：
  - `main`：`f5d8a39fb58ac5c2abc187a3cf01e0c6b5630661`，2026-09-30 19:54:53 +0800；
  - `archive/20261001-1004-experiments`：`b97b26d2cd56caa92e8393f0c50b494ea6dd3658`，2026-10-09 11:40:00 +0800；
  - `stage05/4090-optimization`：`55af61bc5016b7845e7f2b3fd4f66ce91dcb697e`，2026-07-17 17:42:58 +0800；
  - `stage06/h75-final-handoff`：`76c829045c797c16c2a69f4ab6533682051e6858`，2026-10-07 21:01:04 +0800。
- `main`总跟踪文件：737；根目录早期基线88，内嵌`无人机/`项目649（业务文件647+快照元文件2）。
- `archive`相对`main`新增460个归档文件、68,281行，覆盖2026-10-01至10-04的24个方法目录、40行实际运行/预检记录。
- `stage06`与`stage05`树相比只新增/修改5个交接文件；它给出了H75与HPRD-Lite的后续线上记录，但没有把全部训练代码和重型产物纳入该分支。

> 本记录替代先前只审计根目录 Stage01–04、以及后来只审计`main`的两个不完整版本。仓库不能只读默认分支：`main`提供截至2026-09-30的完整历史快照，`archive`补齐10月1–4日实验执行证据，`stage06`才冻结当前H75冠军和10月5日后的关键结果。

---

## 1. 仓库层次不能混淆

### 1.1 根目录：早期 Stage01–04 基线

内容是：

- SegFormer-B0；
- source 6996、固定5597/1399；
- test1 500张；
- 本地 mIoU `0.7123106568503285`；
- 早期线上 `0.572820`。

根目录适合参考数据只读、标签映射、同步增强、固定划分、指标和工程验收，不能代表后续成绩。

### 1.2 `无人机/`：后续完整研究快照

`无人机/GITHUB_SNAPSHOT.md` 明确说明该目录是2026-09-30从桌面项目同步的后续研究归档。它包括：

- test1时期的大量训练方法与线上分；
- M2F、KD、SEMEDA、MaxEnt、RegionContrast、伪标签、ATDA等历史；
- test2半决赛的theta278、GTRUST、D1、A05、MIC、盲类召回、反转配比、迭代教师和硬伪标签；
- 方案手册、执行代码、SHA和提交包清单。

快照明确排除了：

- 289个ZIP；
- 9个DOCX；
- 4个PDF；
- 2个PPTX；
- 3个PT；
- 40张PNG；
- 官方数据和本机应用设置。

因此Git仓库能审计代码和轻量证据，但不能凭Git单独复算所有checkpoint→ZIP链。

### 1.3 与当前本地项目的结构对比

将远程 `无人机/` 的649个跟踪路径与当前桌面项目逐路径比较，当前本地仅缺：

- `GITHUB_SNAPSHOT.md`
- `SNAPSHOT_MANIFEST.json`

也就是说，远程内嵌项目不是另一套神秘代码；它基本就是2026-09-30时的桌面项目快照。它的价值在于把大量历史方法和成绩集中成可搜索的Git证据。

---

## 2. 远程仓库能证明的最高成绩边界

### 2.1 快照内最后一个明确、已绑定的test2冠军

`无人机/提交包/清单与记录/USER_REPORTED_SCORES_20260928.json` 的最高明确记录是：

```text
REVERSE_ARMB = 71.045315
checkpoint SHA256 = f186e06ddd531724f647757587e5afec1415fb672410534de9658be60e3c58b9
ZIP SHA256        = 9b426a5bda7fcc8b6b96af9a7a6c2a8e86467dbb4d8bb61441717538f29f1885
```

对应方法：A05初始化和teacher、盲类召回soft-KL、source coefficient 0.1、target coefficient 1.0、500-step ramp、2000 loops。

### 2.2 `71.3`不在`main`，但已在`stage06`精确闭合到H75

对`main@f5d8a39`的737个跟踪文件检索时，确实没有平台成绩`71.3xx`；当时命中的`71.39% keep ratio`和`71.339% flip`都不是线上mIoU。这个结论只适用于默认分支截至2026-09-30的快照，不能扩展到整个远程仓库。

完整枚举远程refs后，`stage06/h75-final-handoff@76c8290`给出了精确记录：

```text
当前冠军：H75
online mIoU = 0.713294 = 71.329400%
相对ACDD_V1_ANCHOR 71.158317：+0.171083pp
checkpoint SHA256 = b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a
ZIP SHA256        = 7163035014854573920ca319a5f282c1575fc4b9a62d872d209b9ad2decd9b71
package_t.py SHA  = e52bb7831a98f82108ae353a01fedbc1423f13d32405bf86a210068b00eba0b0
```

H75不是推理期ensemble。它把四条同配方ACDD轨迹聚合成一个普通B3 checkpoint：

```text
C   = ACDD_V1_ANCHOR step1000
H75 = (9*C + R05 + R06 + R07) / 12
    = 0.75*C + (R05+R06+R07)/12
```

只平均475个可训练参数张量，其他参数和buffer复制C；最终只加载H75一个checkpoint做B3 D4 TTA8、native 256 FP32 logits mean、Bstar一次、bilinear 1024、argmax+1。该权重平均方式符合项目已获确认的“训练后合并为单权重”边界。

身份边界也必须如实保留：`stage06`绑定了最终checkpoint、ZIP和推理代码SHA，并声明1300/1300预测像素与PNG字节黑盒重放一致；`archive`中能找到相同SHA的`package_t.py`源码。但`stage06`没有收录C/R05/R06/R07四个成员的完整SHA表、参数平均脚本/receipt、平台回执截图或R06/R07独立run report。因此H75可定为**当前冠军且最终工件身份已绑定**，但按本项目五分法仍更接近`SCORED_IDENTITY_PARTIAL`，不能写成整个生成链已经完全`SCORED_BOUND`。

### 2.3 分支时间边界

必须同时保留三层事实：

1. `main@f5d8a39`快照内最高是Arm B `71.045315`；
2. `archive@b97b26d`记录10月1–4日实验，历史线上最高写到ACDD `71.158317`；
3. `stage06@76c8290`冻结当前H75 `71.329400`和HPRD-Lite `71.177602`。

禁止再用默认分支未更新来否定其他远程分支中的后续事实，也禁止把后续结果倒填成旧commit当时已经包含的内容。

---

## 3. test2成绩演进：仓库中的明确证据

| 方法/候选 | test2线上分 | 相对关键基线 | 身份/结论 |
|---|---:|---:|---|
| theta261+D4 | 69.639339 | — | 旧基线 |
| theta278+D4 | 70.569757 | +0.930418 vs theta261 | 单checkpoint |
| GTRUST_T | 70.080740 | −0.506181 vs theta278 | 本地R1099 +0.626pp但线上反转 |
| C2 SWA | 70.488526 | −0.098395 vs theta278 | 轨迹平均无新target信息 |
| D1_2k | 70.736134 | +0.149213 vs theta278 | hard pseudo agreement，单checkpoint |
| D1_alpha050 logits | 70.763755 | +0.193998 vs theta278 | 推理期双checkpoint融合，非合规历史参考 |
| A05_WEIGHT | 70.775011 | +0.038877 vs D1 | 合规单merged checkpoint |
| D1_NO_BARREN_POS alpha050 | 70.506962 | −0.268049 vs A05 | 非合规双checkpoint；source改善未迁移 |
| BlindRecall CONTROL | 70.272000 | −0.503011 vs A05 | source-only continuation税 |
| BlindRecall TREAT | 70.867100 | +0.092089 vs A05；+0.5951 vs control | 盲类召回soft-KL正因果 |
| Reverse Arm A | 70.957249 | +0.182238 vs A05；−0.088066 vs Arm B | source=0，纯target略过拟合 |
| **Reverse Arm B** | **71.045315** | **+0.270304 vs A05** | 快照内冠军；source=0.1/target=1.0 |
| Round2 merge | 71.016535 | −0.028780 vs Arm B | 加test1并延长到5000步无增量 |
| Hard pseudo | 69.932145 | −1.113170 vs Arm B | 硬标签固化teacher错误 |

这些结果给出的可靠因果链：

```text
盲类soft-KL TREAT − CONTROL = +0.5951pp
source continuation CONTROL − A05 = −0.503011pp
轻source锚的Arm B − A05 = +0.270304pp
纯target Arm A − Arm B = −0.088066pp
加target数量/步数 = −0.028780pp
改硬标签 = −1.113170pp
```

因此仓库已经证明：

- target-domain软监督是真杠杆；
- source约束不能太强，也不能完全去掉；
- hard pseudo不是soft-KL的升级；
- 单纯更多target图和更多步数不能继续放大收益。

---

## 4. test1历史方法与成绩总览

以下是仓库中已经实际做过、不能再包装成“新方法”的代表项。分数为各自记录下的test1线上标准分或明确校准分，不能与当前test2直接横比。

### 4.1 正收益方法

| 方法 | 代表标准分 | 相对B3基线66.739544 | 结论 |
|---|---:|---:|---|
| CWFA Exp23 | 67.371405 | +0.631861 | B3有效，搬到B4反而−0.95 |
| no-decay Exp25 | 67.412 | +0.672456 | B3有效，B4无效 |
| MixStyle α0.1 | 67.231 | +0.491456 | 弱域随机化正收益 |
| DINOv2 distillation | 66.991 | +0.251456 | 小正收益，不是大杠杆 |
| Edge BCE Exp41 | 67.053 | +0.313456 | 结构监督正，但小于CWFA |
| MixUp Exp45 | 67.0098 | +0.270256 | 温和混合正收益 |
| SEMEDA Exp47 | 67.796 | +1.056456 | 训练期大正收益，主要是粗结构而非细边界 |
| Stage55b pseudo500 | 68.048 | +1.308456 | test域信息大收益，但规则合规性曾阻塞 |
| RegionContrast exp57 | 67.5182 | +0.778656 | modeB正收益 |
| RegionContrast exp58 | 67.6376 | +0.898056 | modeC正收益 |
| MaxEnt λ1 stage49 | 67.7711 | +1.031556 | λ1有效，λ5崩；后续仍受校准/泛化限制 |

### 4.2 失败或关闭方法

| 方法 | 结果 | 结论 |
|---|---:|---|
| B4+CWFA | 66.001109 | CWFA不跨backbone迁移 |
| B4+no-decay | 66.584083 | no-decay不跨backbone迁移 |
| B5+LoRA | 63.681030 | −2.72pp，灾难 |
| CopyPaste | 66.427993 | 本地高、线上负；强空间扰动失配 |
| attention fusion head | 65.978999 | 改head破坏预训练对齐 |
| weighted CE | 66.6902 | 微负 |
| Focal | 66.397 | 负 |
| CE+Dice | 66.5635 | 负 |
| Edge BCE + CWFA | 66.2893 | 两个单独正收益组合后互毁 |
| distill+MixStyle | 66.915682 | 微正但低于单方法，无协同 |
| Full6996 | 六个matched包全负 | 更多source数据路线关闭 |
| Stage65三teacher pseudo round2 | 本地最高、线上低 | 本地→线上反转，伪标签螺旋失败 |
| 多尺度TTA | M2F与旧本体均全负 | 非原生尺度有害 |
| 空间硬规则清车 | 60.034600 | 图级先验误上推到像素邻接，灾难 |

### 4.3 不能直接重开的教训

- “方法在B3正”不代表在B4/B5正；
- “单方法正”不代表组合正；
- “本地mIoU更高”不代表线上更高；
- 校准向量不能跨模型硬迁移；
- More data、more capacity、more steps都已出现明确负例；
- 训练期异构信息和target signal比普通source拟合更有价值。

---

## 5. 仓库里确实存在、此前未充分纳入当前讨论的方案

## 5.1 RareExposure v2：已训练完成、仓库内无线上分

后期审计发现：

```text
run: stage23_rareexposure_v2_seed42_56000
completed: 56000/56000
best checkpoint SHA:
dbe1063c72131303b9759b8948a072b07ff093968ecd20c63d11b40e27115851
local val mIoU: 0.810348
```

唯一变量：不用在1024图上实际无空间作用的RareCrop，改成全图类别感知曝光采样：

- 每epoch 4197 uniform；
- 700 Barren；
- 700 Agricultural；
- 无放回子流；
- 每图每epoch最多3次。

仓库没有找到其明确线上成绩。它是“训练已完成但未在快照中闭环评分”的真实资产，不得写成失败，也不得从local 0.810348推断会上涨。

## 5.2 TAST-E / AC-FDA：设计完整，但GPU正式实验没有执行

目标：只更新stage0/1，使用source content和test2 donor的Fourier低频幅度混合，匹配SOURCE_DONOR control与TEST2_DONOR treatment。

状态：

- 首版CPU eligibility实现有致命问题：HWC FFT轴错、所谓MMD不是MMD、source split混入pseudo、bootstrap不互斥；
- 原结果被撤销为`INVALID_DIAGNOSTIC_IMPLEMENTATION`；
- corrected CPU方案已写，但没有证据表明完成并获GPU授权；
- 正式GPU TAST-E没有运行，没有线上成绩。

这不是“已证失败的方法”，而是“原诊断无效后关闭优先级、未形成正式因果结果”。若未来重开，必须从corrected operator-specific gate开始，不能采用旧39.3% reduction。

## 5.3 A05 + MIC block masking：训练方法存在，正式评分未闭环

MIC是仓库中真正不同于普通teacher consistency的机制：teacher看完整target，student看70%/64px block masking后的target，要求恢复完整语义。

仓库同时存在：

- 完整CONTROL/TREAT冻结手册；
- 已运行的MIC TREAT step8000；
- 旧MIC包因1299/1300文件错名和第8个D4 inverse错误得到无效分`6.976285`；
- 修复推理包SHA `c9b67be903090722834c8d4e4f3f9d82e886559320202180fe919e0bfc667e41`；
- 修复包全链工程PASS，但快照中没有平台分。

结论：`6.976285`不是MIC方法分；MIC正确包在快照截止时仍属未评分候选。它不能与后续ACDD混为一谈。

## 5.4 ROUND2_ARMB_TEACHER：包已存在，快照无分数

配方：

- student init=A05；
- teacher=Arm B 71.045315；
- soft KL + blind recall mask；
- source=0.1，target=1.0；
- 2000步。

身份：

```text
checkpoint SHA = acb3a6fe00c4eb554d25479d4f4568b6df9d84aa3e04c589ee0970aebafb80ff
ZIP SHA        = 8329d2d07be2020fb76462684ec49e93fa3a5cf8ab4c4cd4e00cc8171118efe0
```

仓库没有绑定线上分。它回答“更强teacher能否继续迭代自训练”，是快照里最接近后续ACDD问题、但尚未闭环的一项。

## 5.5 Stage66：SEMEDA + delayed CWFA，训练完成但无明确线上回填

状态：

- SEMEDA λ0.5；
- delayed CWFA start28001；
- best约51800/52000；
- 本地记录约0.81252；
- 多个checkpoint/包存在；
- FP16 cache精度门失败只是cache工程失败，不等于模型算法失败；
- 快照中没有可绑定的线上成绩。

它值得记为“资产存在、未评分”，但历史上Edge+CWFA互毁已经对组合路线给出较强负面先验。

## 5.6 B5 blind-recall代码：只是草稿，不能当实验

`scripts_20260930/b5_blindrecall.py`存在，但内部有明显身份/文档不一致：

- docstring写“A05 student + Arm B teacher”；
- 实际student路径是`runs/b5_train/checkpoint_step56000.pt`；
- 模型源是SegFormer-B5；
-常量MAX_STEPS=5000；
-没有匹配执行报告或线上分。

因此它只能记为未完成草稿，禁止声称B5 blind recall已经训练或评分。

## 5.7 Surgical SEMEDA / donor pool：研究提案，未形成后续正式闭环

仓库提出过：

- 从theta209训练stage2-only SEMEDA短程方向；
- 以当前冠军叠加增量；
- 开采no-decay、MixStyle、DINO distill、RegionContrast、RareExposure等已训练donor；
- 用layer-specific donor gate判断有价值参数子空间。

其中历史SEMEDA stage2参数注入确有test1 `+0.1483pp`，但该“新一轮surgical训练+新donor池”在快照中没有完整后续结果，不能当作已执行。

---

## 6. 其他方案的状态判定

| 方案 | 状态 | 成绩/判定 |
|---|---|---|
| ATDA | 已做、test1有效 | winner 0.706154，相对theta261 +0.33pp；不是test2 71.x |
| GTRUST Geometry | 已做、test2失败 | 70.080740；本地正、线上负 |
| D1 hard target adaptation | 已做、test2正 | 70.736134 |
| ETKD v1/v2 | 已做工程pilot、gate失败 | 未形成可提交正结果 |
| A05 weight projection | 已做、test2正 | 70.775011 |
| UW1 uncertainty weighting | 500-step pilot | H300恶化约0.49pp，无线上 |
| BlindRecall pair | 已做、强因果正 | TREAT−CONTROL +0.5951pp |
| Reverse ratio | 已做、快照冠军 | Arm B 71.045315 |
| test1+test2延长训练 | 已做、负 | 71.016535 |
| hard pseudo | 已做、强负 | 69.932145 |
| TAST-E | corrected方案存在，formal未做 | 无线上 |
| MIC corrected package | 训练已有，正确包未评分 | 无有效线上分 |
| RareExposure v2 | formal完成，仓库未绑定线上 | 无线上 |
| ROUND2_ARMB_TEACHER | checkpoint+ZIP已有，仓库未绑定线上 | 无线上 |
| Stage66 | formal资产存在，仓库未绑定线上 | 无线上 |
| B5 blind recall | 仅不一致草稿 | 不算已做 |
| CT Pair / T@2000 | 已做，本地正但线上负 | T=70.790474，低于Arm B 0.254841pp |
| ACDD E850-A20 | 已做并提交 | 71.115171，低于ACDD旧冠军0.043146pp |
| ACDD S3_ONLY | 已做并提交 | 71.062800；source更高但线上更低 |
| ACDD GROUP_MASS | 训练/打包完成，未提交 | online N/A |
| X110 | checkpoint/ZIP/QA完成，未提交 | online N/A |
| R05 | READY，线上身份未确认 | 禁止绑定71.062772 |
| HPRD-Lite v1 | 已做并提交，身份部分绑定 | 71.177602；相对ACDD +0.019285pp |
| H75 | 已做并提交，当前冠军 | 71.329400；最终checkpoint/ZIP/package SHA已给 |

---

## 7. 更新分支补齐的后续事实

`main`截止2026-09-30；后续结果必须从`archive`和`stage06`读取，不能继续写成“仓库没有”。

### 7.1 10月1–4日执行归档

`archive@b97b26d`新增460个文件，统计为24个方法目录、40行实际运行/预检：30 `COMPLETE`、1 `PARTIAL`、9 `STOPPED`。归档只读核验了28个checkpoint和61项历史代码绑定，自动安全检查为PASS；`COMPLETE`只表示执行链完成，不等于线上提高。

关键新增事实：

- RAS-KD CONTROL/TREAT本地几乎相同，线上RAS TREAT最终为`71.042244`，未胜Arm B；
- BLACK_CORE_DROP线上`71.029651`，未胜Arm B；
- TCA V2和CE020虽然source指标上升，线上分别`70.892150`、`70.977045`，继续证明本地排序会反转；
- CT T在source1399相对C增加约`+0.377259pp`，但对应T@2000线上只有`70.790474`；donor互补存在不等于直接蒸馏会上涨；
- ACDD在CT T上加入non-Gate Arm B anchor 0.15后，step1000线上`71.158317`；
- ACDD endpoint/anchor local sweep选出E850-A20，后续`stage06`绑定线上`71.115171`，仍低于原ACDD；
- S3_ONLY本地达到`77.687231%`，后续线上`71.062800`，是又一个本地更高、线上更低的反例；
- GROUP_MASS、R05和X110有候选资产但未形成可信线上分，不能列为已验证收益；
- SHBR原门控与SUPPORT9均按协议STOP；forced-train也因target支持和行为门失败，禁止称成功。

### 7.2 10月5日后的两条真正新增结果

`stage06`给出：

| 方案 | 线上mIoU | 相对ACDD | 相对H75 | 状态 |
|---|---:|---:|---:|---|
| **H75** | **71.329400** | **+0.171083pp** | 0 | 当前冠军；最终ckpt/ZIP/package SHA已给 |
| HPRD-Lite v1 | 71.177602 | +0.019285pp | −0.151798pp | 新监督机制小幅正收益；代码/工件SHA未在该分支给出 |
| ACDD_V1_ANCHOR | 71.158317 | 0 | −0.171083pp | 旧冠军 |
| E850-A20 | 71.115171 | −0.043146pp | −0.214229pp | 已提交；local筛选未转成冠军 |
| S3_ONLY | 71.062800 | −0.095517pp | −0.266600pp | 已提交；local/online反转 |

HPRD-Lite v1的结构不是旧prototype/contrastive的简单重包装：Champion初始化，Mask2Former真实query region，source-GT prototype资格，Stage3 rank-4 LoRA，训练后merge回普通单B3 checkpoint。训练参数110,592，约占0.2336%；R1099 region precision约99.30%，但test2仅125/1300图有region、coverage约1.59%，1000步中只有103步non-empty。结果说明**异构region+prototype+LoRA方向符号为正，但V1受support/coverage瓶颈限制**。

### 7.3 旧“同seed重跑分数”绑定必须修正

`stage06`明确写明：R05状态是`READY / online unconfirmed`，历史`0.71062772`不能严格绑定到R05。因此不能再把`71.062772`和`71.115171`都称为“同seed exact-recipe重跑”。当前可信绑定是：

- `71.115171` → E850-A20；
- `71.062800` → S3_ONLY；
- `R05` → 无已确认线上分；
- 用户此前报告的`71.062772`保留为历史未闭合身份，不强行改绑。

当前工作必须使用“main历史快照 + archive执行归档 + stage06冠军交接 + 用户平台回报”四层证据，并明确每层的时间和身份边界。

---

## 8. 接入当前工作流的新增规则

### 8.1 每次先做时间边界

引用外部/远程仓库时必须写：

```text
repo URL
commit SHA
snapshot date
latest dated record inside snapshot
```

不能把2026-09-30快照称为2026-10-09现状。

### 8.2 方案状态必须五分法

以后每个方案统一标为：

1. `PROPOSED_ONLY`：只有文档；
2. `CODE_DRAFT_ONLY`：有代码但身份/执行未闭环；
3. `TRAINED_UNSCORED`：checkpoint存在，未绑定平台分；
4. `SCORED_IDENTITY_PARTIAL`：用户报分但缺平台ID/截图等；
5. `SCORED_BOUND`：score、checkpoint SHA、ZIP SHA、生成链和平台回执全部绑定。

不得把“有脚本”写成“做过”，不得把“有ZIP”写成“有成绩”，不得把无效推理包写成方法失败。

### 8.3 当前新方案评审必须先查历史重叠

提出新方法前，至少检查是否与下列历史重复：

- Fourier/TAST-E；
- MIC masking；
- RareExposure；
- RegionContrast；
- SEMEDA/stage2 capability vector；
- no-decay/paramwise；
- DINO蒸馏；
- blind-recall soft-KL；
- hard pseudo；
- iterative stronger teacher；
- weight projection/averaging；
- M2F semantic mixture。

只有指出“历史做到哪一层、缺的独立残量是什么”，才能称为创新。

### 8.4 当前冠军记录规则

当前应写：

```text
当前冠军：H75 = 71.329400
H75 checkpoint SHA = b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a
H75 ZIP SHA        = 7163035014854573920ca319a5f282c1575fc4b9a62d872d209b9ad2decd9b71
package_t.py SHA   = e52bb7831a98f82108ae353a01fedbc1423f13d32405bf86a210068b00eba0b0
旧冠军：ACDD_V1_ANCHOR step1000 = 71.158317
main快照冠军：REVERSE_ARMB = 71.045315
```

同时标注：H75的最终工件身份已给出，成员轨迹SHA、平均脚本和平台回执仍未在该分支完全归档。不能把这个缺口扩大成“冠军未知”，也不能把它缩小成“完整全链SCORED_BOUND”。

---

## 9. 对当前研发方向的影响

### 9.1 已经不新颖的部分

- 普通prototype/contrastive：RegionContrast、prototype诊断已做过；
- LoRA本身：B5 LoRA有灾难负例；如果再用LoRA，必须说明“冻结强B3、零初始化、低秩只作漂移限制”与旧B5能力阉割的结构差异；
- 更强teacher迭代：ROUND2_ARMB_TEACHER已经实现并出包，只是未绑定分数；
- MIC block masking：已有完整实现和正确包候选；
- Fourier style transfer：TAST-E已有设计与失败的诊断实现；
- rare exposure：已有训练完成checkpoint；
- SEMEDA stage2-only：已有参数注入正证据及surgical训练提案。

### 9.2 HPRD已从设计变成线上小正收益，下一步不能原样重跑

此前HPRD-LoRA只是一份独立设计；`stage06`现已证明HPRD-Lite v1线上`71.177602`，比ACDD `+0.019285pp`，但比H75低`0.151798pp`。这改变了状态判定：

- “M2F query-region + source-GT prototype + Stage3 LoRA + merge单checkpoint”已经做过，不能再包装成新方案；
- 它证明异构region监督方向不是纯理论，线上符号为正；
- 但V1 region support只有125/1300图、1.59% coverage，1000步仅103步有非空region，收益上限被曝光率限制；
- 后续若继续，独立残量必须是**提高Barren/Agricultural有效support，同时守住region precision和单checkpoint合规**，而不是重扫LoRA rank或直接放宽全部region。

HPRD-Lite在`stage06`只有机制、统计和线上分，没有checkpoint/ZIP/code SHA，当前分类为`SCORED_IDENTITY_PARTIAL`。补齐身份优先于复训。

### 9.3 H75之后的主线判断

H75证明同一ACDD配方的多条独立轨迹存在可平均掉的方差，参数空间聚合能比单次冠军再增`+0.171083pp`。但这不等于可以无限增加轨迹或扫描汤权重：

- 目前只有一个H75在线点；没有C/R05/R06/R07各自完整在线分和协方差，不能估计增加第5条轨迹的边际收益；
- H75的75% Champion权重是已成功配方，不应在无新证据时做密集比例扫描；
- X110虽有checkpoint/ZIP/QA，但未提交；它是沿B→C方向外推，不是H75的独立验证；
- 最有信息量的下一步，应优先补齐H75成员身份与HPRD工件，之后再比较“同盆地稳健聚合”与“提高异构region support”两类真正正交方向。

---

## 10. 三个更新分支的用途边界

| 分支 | 它真正补充了什么 | 不能据此声称什么 |
|---|---|---|
| `archive/20261001-1004-experiments@b97b26d` | 10月1–4日460个轻量归档文件；24方法/40记录；训练代码、配置、local metrics、checkpoint/code SHA审计 | `COMPLETE`不等于线上提高；多数候选无线上分；归档没有H75/HPRD正式工件 |
| `stage05/4090-optimization@55af61b` | 旧B0/test1-500 Wave1；768与Rare Crop本地实验；Exp04包SHA和工程链 | 这是2026-07旧阶段，不是当前test2主线；Exp04线上仍pending，不能与71.x比较 |
| `stage06/h75-final-handoff@76c8290` | H75 71.329400冠军；HPRD-Lite 71.177602；后续成绩时间线；H75最终SHA | 分支只新增/改5个文档/CSV文件；不含完整H75平均脚本、成员SHA、HPRD代码和平台回执 |

`stage05`中Exp04 Rare Crop本地`0.749708991082`、checkpoint SHA `57c216...`、500图ZIP SHA `878dda...`是早期工程资产；因线上仍`pending`且使用test1-500/B0旧合同，只作历史参考，禁止复活为当前test2候选。

---

## 11. 最终结论

1. 远程仓库必须按四个refs读取；只看`main`会漏掉10月实验和当前冠军。
2. 当前最高已经精确绑定为 **H75 = 71.329400**，不是“约71.3待确认”，也不是ACDD 71.158317。
3. H75的成功来自同配方多轨迹参数平均成一个checkpoint，相对ACDD增`+0.171083pp`；最终checkpoint、ZIP和推理代码SHA已给，但成员轨迹与平均生成链仍需补齐。
4. HPRD-Lite v1线上`71.177602`，证明异构query-region/prototype/LoRA方向小幅正收益；瓶颈是support不足，不是precision不足。它已经做过，后续不能原样重跑或改名包装。
5. `archive`补齐了ACDD、CT、M2F、SHBR、RCS等执行证据，也再次显示source本地排序会在线上反转；任何新候选仍必须把工程正确、本地机制和线上收益分开。
6. R05没有可信线上绑定；`71.115171`属于E850-A20，S3_ONLY记录为`71.062800`，此前`71.062772`保持身份未闭合，不能继续合称同seed复现。
7. 当前最便宜且最高优先级的工作不是立刻再开训练，而是补齐H75成员checkpoint/平均脚本/receipt与HPRD checkpoint/ZIP/code SHA；这些身份补齐后，才决定是做有限的H75稳健聚合验证，还是做提高Barren/Agricultural support的HPRD v2。
