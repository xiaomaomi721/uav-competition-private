# AIC2026 UAV 半决赛｜Claude Code 融合研究与执行方案 v1.1

> **制定时间**：2026-10-28（北京时间）  
> **适用对象**：Claude Code / 负责服务器执行的队员 / 负责证据审核的队长  
> **状态**：**可作为实施任务书；未表示服务器已有新执行结果，也不等于授权自动平台提交。**  
> **来源**：《AIC2026_UAV_半决赛全对话与分支汇总_新对话交接_20261028(1).md》（10月28日上午的证据台账）、Muse《PLAN_FINAL_20261028.md》（讨论版）、已有官方半决赛通知及附件；另结合EM、BBSE、SAM3、DINOv3研究资料设计验证方法。  
> **统一策略**：**P0冠军复现与材料保障并行不让路；P1原生缓存/数值回放+EM/BBSE核心实验；P2现成M2F单模型探针；P3 SAM3独立监督审计；P4 DINOv3-SAT限时探针。高级Source/Target分域基建后置且可选。**  
> **最关键的执行纪律**：不要把【文档所述】当【主机核实】，不要把【Source1399】当【Target3594真值】，不要把【候选预测差异】当【纠错率】，不要把【通过本地Gate】当【已线上提分】。
> **单一执行事实源**：自本版起，**v1.1是唯一可用于安排/验收任务的执行任务书**；原v1.0保留历史，Muse `PLAN_FINAL_20261028.md`及其先前版本仅作为**决策背景/来源记录**，不得并行派单或用其时间表覆盖本版Gate。官方最新规则及经核实原始证据仍具有更高优先级。任务书本身**不是任何阶段的执行授权**。
>
> **v1.1变更（2026-10-28；仅范围和合规流程）**：① §5.3域分组/特征距离/Source重加权降为P1 **可选后置**，只有原生缓存与金丝雀、EM/BBSE源域模拟、M2F探针、SAM3审计四项核心交付均完成且P0安全时才可申请启动；② Stage 0增加**队长今日发送SAM3点名许可问、将发送证据和正式答复归档`00_audit/`**的人工作业；③ 调整调度、Stage0输出、Claude首条指令避免双轨和自动授权。**未修改**任何已知线上成绩、H75的SHA、原生推理协议或算法评审阈值。

---

## 0. Claude Code 必须先读：命令与授权边界

1. **先做只读Stage 0审计**：读取现存代码、checkpoint、正式ZIP、数据、脚本SHA、GPU与磁盘资源、模型许可和官方问答原始凭证，交付 `00_PREFLIGHT_AUDIT.md`、`ASSET_MANIFEST.json`、`OPEN_ISSUES.md`。**另登记SAM3专项问答人工任务**（§2.4）；Claude Code只能准备问句、记录证据及核对答复，**不能代替队长发送或冒充官方许可**。有缺项报告，不猜测路径、不伪造完成状态。
2. **可自主执行的低风险工作**：新建独立实验目录；读取/校验既有资产；写脚本、测试、文档；基于已有小样本缓存进行CPU分析；在不触及其他任务的资源许可下开展短小smoke（GPU是否获用户授权以实际会话指令为准）。
3. **需队长/授权负责人放行的动作**：长时GPU训练、大规模外部模型下载或推理、覆盖/移动原冠军资产、切换正式冠军、公开/人工发布比赛结果ZIP、上传报名平台、提交网盘链接。**禁止Claude Code自行登录平台提交**。若用户将本文件交给Claude Code并明确授权某阶段，则按授权范围执行该阶段；否则停在审计/实现/报告。
4. **不可触碰**：不人工标注半决赛3594张再用于训练/校准；不使用外部非许可训练数据或闭源商业模型API；不把多模型在线推理融合伪装成单模型；不以官方排行榜反复调参构造“准真值”；不修改H75正式原件；不抢占他人GPU。
5. **策略依赖**：测试集RGB的训练期使用、基础模型产生伪标签/固定bias拟合、公开权重许可证与预训练数据合规必须核验**原始**官方十问答复及模型使用条款。Muse写“已确认合规”是报告，尚未见原文之前不能替代证明。**队长须于10月28日向官方发出点名SAM3的书面确认问并归档，状态初始`AWAITING_OFFICIAL_REPLY`**。SAM3点名用途若无书面明确许可，只运行与允许范围相符的离线审计，不提交相关血统成果。
6. **时间**：预测ZIP截止 **2026-11-01 20:00 CST**；作品和网盘截止 **2026-11-02 20:00 CST**；每日最多2次官方预测提交；半决赛取最高分。**内部在11-01 15:00冻结拟提交冠军**，不是官方截止。若因新模型冲击影响P0，可直接停止新线。

### 0.1 当前有效线上台账（不可跨赛段混排）

| 已知正式方案 | 半决赛3594线上 mIoU | 证据性质 | 决策 |
|---|---:|---|---|
| `H75 + Bstar + D4-TTA8` | **70.998665%** | 用户回报官方成绩 | 当前冠军/保险 |
| `H75_T3_FULL_ST_5597_3594` | 70.990138% | 用户回报官方成绩 | 未超过 |
| `CTRL@3594 + Bstar` | 70.318575% | 用户回报官方成绩 | 未超过，勿重复提交相同包 |
| `H75 + Zero` | 67.979564% | 用户回报官方成绩 | 强烈说明源/目标偏好相反 |
| `CTRL@5400` / `soup_4mem` | **未知** | 只有Source本地分 | 不得写为已评分 |
| M2F E02 / SAM3 B_sam / DINOv3-SAT | **未知** | 计划/源域实验 | 不得写为已评分 |

历史`71.329400%`是**不同赛段的H75复赛数据口径**；历史`70.569757%`也是复赛test2，**不能与3594成绩比较名次**。

### 0.2 固定数据、标签和推理协议

- Source **6996**张有标注：固定训练**5597**、验证**1399**（H300+R1099），严禁让1399混入模型训练或新伪标签蒸馏。Source原始GT `0` 为ignore，`1..8`有效；模型训练标签 `0..7`，ignore `255`。
- Target半决赛**3594**张无GT（test1=500、test2=1300、test3=1794）。**不能把3594再与1800相加**。3594目标仅有RGB，任何目标逐类IoU均未知。
- 八类顺序：`Background, Building, Road, Water, Barren, Vegetation, Agricultural, Vehicle`。
- 正式H75：RGB1024 → `/255` + ImageNet normalize → **单checkpoint** SegFormer-B3 → D4几何8视图（各视图先旋转、可选水平镜像）→ inverse → **FP32原生logits求均值** → 原生logits加固定8维bias → 双线性上采样1024（`align_corners=False`）→ `argmax + 1` → PNG模式`L`、uint8、1024²、像素1..8 → 与输入同名，ZIP根目录平铺3594张。
- `Bstar = [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]`。**不得误用**更早版本`Water=0.7, Agricultural=3.3215`。
- H75最重要的稳定信息：
  - 权重SHA256：`b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a`
  - 已评分3594 ZIP SHA256：`c012bae1237160a0c0d5fbccd168f58facf62c03bf60ee2cc82877cb21454d30`
  - 正式打包脚本`package_semifinal_bias.py` SHA：`206cc80d6bc2a88012f6dd06b02829ca72a8ecef245ed06d6f160ce63d7723a1`
  - `bias_bstar.json` SHA：`5848eb3cbcfa521b8ec464f839d966dc2c78b10d827f50253c705701d860cbe0`
  - `semifinal_3594.csv` SHA：`76463ff99a495afe82085bb59530032f13f9efbff2cc1936f09178285e6cdd19`
- Source1399 **同协议D4+Bstar** H75=77.5311488%；D4+Zero=80.4417713%；CTRL@3594 D4+Bstar=79.1242%；T3-FULL-ST D4+Bstar=77.6267%。不同评估协议不能混算。

---

## 1. 成功标准、优先级和总体架构

### 1.1 核心研究问题

- **RQ-A（获得新信息）**：在官方3594张无标签目标域中，SAM3/DINOv3-SAT等独立视觉表征是否能在**有GT的Source分歧区域**实证纠正H75错误，并形成可迁移的可信监督？
- **RQ-B（目标域选模）**：在没有Target GT时，能否建立比单一Source1399 mIoU更有辨识力的**风险筛选体系**，识别已知的H75+Zero、CTRL下降与T3几乎持平，同时避免为4次线上结果拟合伪评分器？
- **RQ-C（先验/决策校正）**：Saerens-EM、BBSE在“类别先验偏移”假设不完全成立的航拍场景下，能否提供**经过源域模拟验证**的候选全局8维bias？能否保持H75线上已验证的Bstar优势？
- **RQ-D（成果交付）**：在任何改进失败时，是否仍能在截止前**用干净环境复现当前H75已评分预测**并提交配套作品材料？

**成功分级**：
- **P0必须成功**：H75模型、协议、全3594预测/QA、Docker或官方要求的复现环境、提交材料具备可审计对应关系。
- **研究成功**：可重复的评估报告、可靠性审计、校准模拟报告与判据；即使没有新线上冠军也属有价值结果。
- **成绩成功**：只有官方同一3594排行榜**严格大于70.998665%**、且相应模型/代码/许可/材料可复现，才称“超越H75”。

### 1.2 核心执行流水线

```text
                  ┌─ P0 现有冠军H75 → Docker洁净环境复现 → 材料冻结(始终并行)
                  │
只读资产核对 ──────┼─ P1 H75 Source/Target统一raw scores缓存 ─┬─ Source GT基础诊断
                  │                                      ├─ Target无GT风险指示器
                  │                                      └─ EM / BBSE模拟与对照
                  │
                  ├─ P2 M2F E02 单模型 → 工程/速度/源域Gate → 候选ZIP
                  ├─ P3 SAM3 Source审计 → 分歧纠错检验 → (许可)B_sam/训练监督
                  └─ P4 DINOv3 SAT权重与许可证 → 2000步A/B探针 → 选做后续
                                                          │
       各线共用：协议锁定、同口径诊断、合规Gate、QA、manifest、人工决定线上提交
```

### 1.3 本方案修正Muse原计划的地方

| Muse原建议 | 融合版处理及理由 |
|---|---|
| 让EM对齐Bstar的6维符号/余弦≥0.7作为认证 | **取消硬认证**：bias全维加常数不改变softmax/argmax；必须先去除gauge，再比较预测图、面积和Source模拟效果。Bstar是线上参考不是“真先验”。 |
| Source混淆矩阵+目标后验统称EM | **拆分经典EM和BBSE**：前者需校准后验，后者利用混淆矩阵；两者假设/求解不同。 |
| SAM3 `Precision≥0.70, Recall≥0.20`可供目标监督 | 仅作**screening**；真正准入要验证与H75分歧区的净纠错、置信下界、覆盖及对象与语义映射。 |
| `512×512 fp16 logits`直接存盘 | **先查实测原生logits维度**；按原生分辨率缓存原始score，并经过与正式H75预测金丝雀重建的数值验证；不默认512。 |
| EM认证后所有新模型通用 | **禁止自动跨模型迁移**：各模型概率校准/输出含义不同，M2F需独立校验。 |
| `Source1399`绝对阈值与预测差异0.5%~12%直接筛选 | Source仅用于灾变否决，预测差异是诊断**不是必过区间**；新增目标域风险审计。 |
| EM失败则新模型Zero bias，视为低估 | Zero**不是理论下界**，只是固定推理对照；不得把其线上成绩称为低估。 |
| P3/P4固定时间和速度 | 所有耗时均需以实际GPU吞吐测量；没有核验就不能写“已在30分钟完成”。 |

---

## 2. Stage 0：远程资产、法律/规则、GPU只读核验（最先执行）

### 2.1 主机/路径——只作为查找线索

```text
uavmain（user2）:
  /data/uav/reproduction/H75_713294_REPRO/
  /data/uav/runs/ASTRA_H75_X110_20261005/ckpt/H75.pt
teacher-4090:
  /home/jhy/aic2026_h76_20261026/
  /home/jhy/aic2026_ctrl3594_teacher_20261027/
teacher-4090-2（实际主机/别名和GPU占用必须核实）:
  /data/uav/runs/H75_T3_FULL_ST_5597_3594_20261027_user2/
  /data/uav/runs/H75_ORIGINAL_TEST3_NINE_RUNS_20261026_qFb3An/
```

**Claude执行**：读取SSH别名/现有配置；在授权主机上只读运行`hostname; pwd; date; nvidia-smi; df -h; find ... -maxdepth ...`及`sha256sum`；核对目标输入、CSV ID清单、Source train/val清单、H75 checkpoint、正式H75 ZIP、正式脚本、Python/CUDA/PyTorch；检查目录是否指向同一存储。不得对全磁盘做无边界find；不得改动现有模型、旧运行目录或占用其他人正在使用的GPU。

### 2.2 资产锁定表

| 资产 | 最低验收 |
|---|---|
| H75权重 | 文件可读、`sha256sum`精确命中既有值；结构严格加载成功 |
| H75已评分ZIP | SHA命中，3594 PNG均合格；与正式提交记录/截图对应 |
| 正式推理代码 | 关键脚本SHA命中；差异或包装器bug单独创建patch，旧金标准不覆盖 |
| Target3594 | 唯一文件名3594；RGB可读；1024²；manifest无缺失/重复，文件SHA清单 |
| Source5597/1399 | 原始训练与验证清单互斥；标签映射和ignore处理一致 |
| 资源 | 可用GPU和显存、进程归属、磁盘容量与剩余时间实测 |
| 官方许可 | 原始十问答复、规则与外部模型许可证/数据来源归档；不明确则标`PENDING` |
| M2F E02 / PMT权重 | 仅在找到实物、来源、SHA、训练配置后标`FOUND_VERIFIED` |

**Gate S0**：若H75权重/正式ZIP/映射/CSV不符，所有改进线暂停；先修P0。若只有外部权重许可未清楚，仍可先做合法的H75/Source分析，不运行未获授权的目标域伪标签训练。**SAM3专项确认问已发不等于获准；未答/模糊答复仍标`PENDING_APPROVAL`，不得越过§8许可门。**

### 2.3 工作区，严禁覆盖原模型

在已授权且容量充分的共享/独立根目录创建（实际绝对路径由Claude记录，不硬编码）：

```text
SEMIFINAL_FUSION_20261028/
  00_audit/                    # 资产清单、规则批准记录、环境快照、SAM3专项问答证据
  01_h75_reproduce/            # H75容器和canary
  02_cache/                    # H75 raw score/metadata、source/target 分开
  03_target_eval/              # Source基本分层/Target无GT诊断；高级域分组默认不启动
  04_prior_shift/              # EM/BBSE校准模拟及偏置候选
  05_m2f/                      # M2F E02只读资产引用+输出
  06_sam3/                     # audit、伪标签候选、license
  07_dino_sat/                 # probe
  08_packages/                 # 候选预测ZIP（非正式提交）
  09_reports/                  # 综合决策和时间进展
  manifests/                   # JSON/CSV/SHA256SUMS
  code/                        # 共享脚本、配置、测试
  docs/                        # README、Docker、技术方案骨架
```

设立两个硬约束：`input_assets`只读；所有新权重、logits、mask、预测ZIP必须写入本工作区对应子目录。不同候选分配唯一`experiment_id`，禁止`best.pt`/`final.zip`重复覆盖。

### 2.4 当日人工事项：SAM3点名合规确认（Stage 0必须登记，不由Claude代发）

**责任人：队长；目标日期：2026-10-28北京时间。** 这是一项**人工发问及证据归档**任务，不占GPU和官方预测提交次数。Claude Code只负责生成文本模板、检查归档完整性和记录状态；**发问不代表得到许可**。

建议由队长在官方赛题QQ群或组委会明确认可的书面渠道，发送以下确认问（可按实际官方渠道格式调整，不能引导性截断）：

> 我们参加AIC2026无人机低空航拍图像语义分割半决赛。拟使用公开发布、可获得权重且符合其许可证的 **Meta SAM3**（请确认该具体基础模型）在官方提供的Source有标注图像上检验提示分割可靠性，并在获准情况下对官方test1/test2/test3合并的3594张**无人工标注RGB**生成训练期伪标签/教师mask，可能用于训练期蒸馏，也可能用于拟合一个固定的8类全局logit bias；最终提交仅为一个分割模型（允许的同模型TTA）输出，不在推理时集成SAM3。请问：①允许使用该SAM3公开预训练权重作为训练期辅助教师吗？②允许其对上述官方无标签测试图生成mask并用于学生训练吗？③允许利用上述自动生成的mask拟合固定bias吗？④对于公开权重的许可证/预训练数据来源，有无额外约束或需提交的证明？如各项权限不同，请分别明确允许或不允许。我们不进行人工标注测试图再训练。

`00_audit/`中需创建/保存：

- `SAM3_OFFICIAL_CONFIRMATION_REQUEST_20261028.md`：原始问句全文、发出渠道、队长已发/未发状态、北京时间发送时间（未发填null）、消息或截图原件位置、消息ID（如渠道提供）；**禁止Claude填写虚构的发送记录**。
- `SAM3_OFFICIAL_REPLY_EVIDENCE/`：官方原始回复截图/导出/链接及消息上下文，保留日期、发言身份、完整问答，禁止仅截取“可以”两个字作为许可；若未收到回复，目录可为空但状态为`AWAITING_OFFICIAL_REPLY`。
- `SAM3_LICENSE_APPROVAL_STATUS.json`：分别记录`public_pretrain_weight`、`target_mask_generation`、`train_time_distillation`、`target_mask_bias_fitting`四项的`APPROVED / DENIED / PENDING / AMBIGUOUS`，附准确证据路径与人工判定人。缺少明确且适用的许可则不能进入相应Target训练/拟合/正式提交阶段。

**Stage 0验收**：至少完成问句和责任人/状态登记；若队长尚未发问，写`ACTION_REQUIRED_BY_CAPTAIN`列入`OPEN_ISSUES.md`，**Stage 0可报告完成资产审计，但SAM3许可不能标PASS**。收到答复后追加原件并人工确认，不得静默改写历史证据。所有外部权重的独立许可证和比赛禁外部数据/预训练重叠约束仍须分别核查。

---

## 3. P0：H75冠军保险——可复现，不只是“跑过”

**优先级：最高，任何时刻可抢占改进实验资源。目标最晚10-30形成干净环境E2E证据，11-01完成最高分绑定。**

### P0.1 固定冠军依赖和环境

1. 用正式脚本/权重SHA确认`H75 + Bstar + D4`的源文件版本、环境；导出Python依赖锁定、GPU/CUDA/PyTorch版本、输入文件身份及完整命令。若有历史 `qa_p0b_zip.sh` 路径推断问题，**另建修正版wrapper**，不得改写原始文件后冒充同SHA。
2. 制作Dockerfile、requirements或锁定环境清单、`README_REPRODUCE.md`、`run_inference.sh`、`qa_submission.py`统一入口。容器输入目录用环境变量/CLI显式参数，不写死`uavmain`特定层级；支持只读数据挂载和独立输出路径。
3. 在真正干净环境里做 `strict=True`权重加载 + 40张canary → QA → 3594全量推理（具备资源且用户许可时）。原生输入顺序、通道、Bstar、TTA、上采样和类别映射严格对齐。
4. 对比**预测内容**而非只对比ZIP容器字节：逐图逐像素比较已评分3594 PNG；记录相同图数、总差异像素、最大差异图；在不同PyTorch/CUDA环境下允许**明确量化并解释的浮点边界差**，但不允许将`PASS_WITH_NUMERIC_TOLERANCE`写成`BIT_EXACT`。
5. 如果无法重建逐像素相同，优先使正式Docker环境贴合原线上环境，再查模型/normalize/TTA/align/bias/图像加载；保留正式最高分ZIP不动。**最终材料必须如实陈述实际复现误差和规避策略**。

### P0.2 P0验收状态

- `PASS_EXACT_INFERENCE`：全3594 PNG逐像素一致，且权重、输入、代码身份已核对。
- `PASS_WITH_NUMERIC_TOLERANCE`：非逐像素一致但差异稀少、可定位浮点边界，且在同一校验协议下记录误差；**这不是自动等同于正式复现PASS，需评估评委实际复现偏差风险**。
- `BLOCKED`：关键资产缺失、文件哈希不符、预测明显偏差或洁净环境不能运行。P0优先修复，禁止仅以旧服务器曾跑通推断可交付。

### P0.3 材料清单（随时滚动维护）

- **最高分对应**：已评分ZIP、唯一checkpoint、训练/参数平均来历、推理脚本和固定bias、配置、SHA256SUMS、技术方案PDF源文档及最终PDF、Dockerfile/环境锁、可复现README、QA报告、完整数据使用说明及许可证、正式排名截图/提交记录。
- 根目录命名必须满足：`参赛团队编号-团队名称-半决赛-队长姓名/`，包含技术方案、模型、代码与数据、赛题结果文档。技术方案至少动机、算法结构、流程、伪代码、训练/推理细节、规则符合性及证据。
- 若新冠军替换H75：**先证明新候选可复现且产物绑定SHA，才申请切换**；否则保留H75作为已知保险。不要把失败实验编进正式最佳模型技术链。

---

## 4. P1-A：统一raw score缓存与数值正确性

### 4.1 缓存目的与定义

对H75分别建立：`Source1399`缓存（用于有GT评估/EM校准模拟）、`Target3594`缓存（无GT诊断/拟合候选bias）。同一权重、同一D4、同一输入normalize、同一native score协议。**raw score指D4 inverse后FP32均值、尚未加Bstar、尚未上采样/argmax的8类logits**。

不要误把：`argmax mask`、`softmax后的概率`、`加过Bstar的logits`、`512 resize后的插值张量`当作原始logits。需要另存时字段名清楚，例如`score_stage=raw_native_fp32_d4_mean`。

### 4.2 缓存输出与格式

- 对每个输入ID生成一份分数缓存，建议独立容器`.npy`或分片`.npz`/zarr；每条记录`image_id、input_sha、model_sha、script_sha、shape、dtype、native_resolution、tta_views、timestamp、bias_applied=false、upsampled=false、sha256`。
- `FP32`作为数值校验的canary基准；全量磁盘不足可用`FP16`存盘并在读取时还原FP32，但FP16量化必须单独做数值影响测试，**不可默认与原版像素完全一致**。
- **按实际原生尺寸分配容量**。若native=256，八类FP16原生logits仅约`3594×8×256×256×2≈3.77GB`（十进制）；若人为扩大512则约15.1GB，增加空间且改变数值链。实际尺寸以原版forward打印为准，不推断为256或512。
- 附加输出：所有3594张相对H75+Bstar的预测类别面积、置信度/熵分布，按旧test1/2/新test3分别统计（**仅诊断，不允许未经规则确认做分域bias提交**）。

### 4.3 canary门槛

1. 冻结一份来自完整manifest的40张确定性样本ID（分test1/2/3，覆盖不同文件名；不能临时选表现好的样本）。
2. 使用缓存重建**H75+Bstar**并与**已评分的同图PNG**逐像素比对；先比较FP32缓存，再比较FP16缓存。
3. 40张共41,943,040像素。历史CTRL迁移B1曾出现**21个不同像素**（比率约`5.0e-7`），这是历史环境漂移证据，不是允许无限偏差。若同环境FP32预期应争取0差异；跨环境数值误差超过`1e-6`像素比例、或出现连续区域/成片分歧，则`BLOCKED_REVIEW`。即使低于门槛也要报告实际数值，不能宣称exact。
4. 对`FP16`量化额外记录`max_abs logits error`（相对FP32可测样本）、预测像素分歧数与位置、分歧处top1-top2 margin；若改变会影响拟合结论，校准使用FP32或更高精度分片。
5. **只有在正式脚本链、40张重建QA通过时**，才能复用缓存做任何EM、B_sam、候选偏置拟合。

### 4.4 实现建议

`code/cache_scores.py --model --ids --input-root --out --precision fp32|fp16 --protocol h75_d4_native --device`（脚本路径、参数自行实装，README提供确切命令）。脚本读取已核验官方ID，不从目录自然遍历推断数据顺序；推理前后记录GPU显存、吞吐及失败ID；断点续跑以ID+原始sha去重，禁止混合不同checkpoint缓存。

**交付**：`CACHE_PROTOCOL.json`、`CANARY_40_IDS.txt`、`CANARY_REPLAY_REPORT.json`、`SCORE_CACHE_MANIFEST.json`、可重复运行脚本。

---

## 5. P1-B：轻量核心评价 + 可选后置的Source/Target分域研究

> **目标不是“无标签算出真实mIoU”，而是产生有证据支撑的风险筛选结论。**任何单一无标签指标都不能保证目标排名正确。
>
> **v1.1范围门（P1 OPTIONAL / DEFERRED）**：在以下四项核心交付**全部真实完成**之前，**不得开工§5.3的域分组、embedding特征距离、Source→Target相似度、Source重加权或搭建其通用平台**：① H75原生缓存+40张金丝雀校验；② EM与BBSE的Source fit/select/holdout互斥模拟；③ M2F E02的资产核验+40图探针/可行性结论；④ SAM3在许可范围内的Source两级审计或明确`BLOCKED_BY_LICENSE`且已记录不可推进的证据。P0冠军链必须保持绿色，且经队长**另行授权**才能解锁可选研究。四项中有一项被阻塞，不得通过声称“已规划”来绕过；可将该项标记`BLOCKED`并上报队长决定是否放弃整个可选项。**此处不影响§5.1基础有GT检查和§5.2低成本面积/分歧统计。**

### 5.1 有标签Source1399：统一评估口径

- 对**相同样本、相同推理协议、相同bias条件**计算预测与GT混淆矩阵`8×8`，真实类别为行、预测类别为列，忽略`255`，类别映射必须先统一。
- mIoU：对每类 `IoU_c=TP_c/(TP_c+FP_c+FN_c)`，取明确有效的八类宏平均；保存per-class IoU、Precision、Recall、总体像素混淆矩阵、每图/每组统计。子集合并用**混淆矩阵先池化再算IoU**；绝不靠两个宏mIoU标量相减反推子集。
- **核心必做**：保留历史H300(300)和R1099(1099)已有分层，输出Source全量及这两组的GT度量；不新增特征提取基建。
- **可选后置**（仅过本节范围门后）：设计先定义特征、后看GT表现的视觉风格/尺度/独立embedding分组。**不要拿Target人工观察结果反向打Source场景GT标签**。
- 所谓605无检测重叠Source子集可以单独报告，但标记`HEURISTIC_ONLY`，不等于目标域GT，不作为唯一优选依据。

### 5.2 Target3594：仅无标签诊断

候选每次必须与H75在**同一3594张、同一几何/输出标签口径**比较，并分别给出test1(500)、test2(1300)、test3(1794)及全体：

1. 每类预测面积占比、与H75的面积差，类别塌缩/突增警报；
2. 全像素和**分类边界分布**的top1-top2 score margin、entropy（校准bias会影响这两项，不能与不同bias跨模型直接等价比较）；
3. D4翻转/旋转一致性（只有单视图logits确实保存时才能计算），邻域/轻微成像扰动下预测稳定性；
4. H75 vs候选预测分歧矩阵，按类别、边界与高低置信区域分解；**预测变化率是诊断，不是纠错率**；
5. **可选后置（不在核心必做范围）**：低维图像/特征分布差异、Source—Target相似性、Source组别权重/近邻覆盖图；仅在§5.3经解锁后执行，且**距离近不等于条件标签分布相同**；
6. 对已有官方结果H75+Bstar、H75+Zero、CTRL@3594+Bstar、T3-FULL-ST+Bstar记录相同的风险向量，不在有意调参后宣布能“准确预测”线上分数。

### 5.3 Source分域验证实现（**P1可选项，默认不启动**）

**解锁要求**：§4的原生缓存/金丝雀PASS、§6的EM与BBSE Source模拟有明确报告、§7的M2F E02已有资产/40图探针结论、§8的SAM3 Source审计有完整结论（或明确许可阻塞并经队长单独决策），四件核心任务均有报告与状态，且P0没有红灯。队长明确批准后才能启动；**禁止作为给上述四项搭通用基础设施的理由提前开工**。时间不够直接跳过§5.3，不影响其它Gate。

- 使用**Source图片本身**与Target无标签RGB计算无监督视觉描述符，固定算法/种子/分组规则，输出`domain_bins.json`、`source_to_target_similarity.csv`；Source验证GT只在分组确定以后用于评估。
- 给出每个分组样本数、8类像素覆盖及mIoU，样本过少或类别缺失的组写`INSUFFICIENT`，不把空类算0抬升或拉低。
- 若本可选项解锁，每个候选再增加视觉分组对照表；未解锁时只需`Source全1399 / H300 / R1099`和`Target三部分基础风险向量`，不可将缺失分域表格写成核心Gate未通过。
- 图像特征如果使用H75内部表征，需报告“表征和被评估模型非独立”的潜在偏好；如使用新外部预训练表征，先做许可检查。

### 5.4 如何利用历史在线标签而不造假

- 预先固定候选风险信号和阈值，先做负控：**H75+Zero的Source高分不应被当作支持目标提分**、CTRL不能因Source+1.59被自动放行；T3-FULL-ST目标近乎持平应谨慎解释。
- 四个已知线上方案中两者共享同一H75模型仅换bias，它们不是四个独立模型样本。**不得以这四个点训练回归器拟合“目标mIoU”或通过线上多次提交调超参**。
- 最终`TARGET_RISK_REPORT.md`按 `GREEN/AMBER/RED/UNKNOWN`标注**风险**，不得输出无证据的预测mIoU数字。示例：类面积极端塌缩→RED；指标不确定/冲突→UNKNOWN，默认需人工审查。

### 5.5 报告的最低验收

- `SOURCE1399_METRICS_<experiment>.json`：每类confusion、TP/FP/FN、IoU和已有H300/R1099分组；额外视觉特征分组仅在§5.3获批后附加；
- `TARGET3594_DIAGNOSTICS_<experiment>.json`：无标签基础统计、test1/test2/test3变化、分歧、输入清单；高级Source/Target视觉分组非必做；
- `TARGET_RISK_REPORT.md`：可检验观察、支持/反驳证据、尚未知事项，不给“准确率预测”。

---

## 6. P1-C：EM/BBSE 全局先验校正的**可证伪实验**

### 6.1 为什么做，以及关键假设

已知H75+Bstar在线上比H75+Zero高 **3.019101 pp**，这是“固定类间决策阈值显著影响目标域成绩”的强证据，**不是“Bstar等于目标真实类先验”**的证据。经典Label Shift假设`P_T(x|y)≈P_S(x|y)`，但本赛题视角、季节、尺度和成像均可能变化，不能假设自动满足。

**明确区分**：
- **Saerens-EM**：用校准后`P_S(y|x)`与`P_S(y)`迭代估`P_T(y)`；不需要把混淆矩阵直接塞进EM核心迭代。
- **BBSE**：用Source真实GT上的预测混淆`C_{ij}=P_S(ŷ=i | y=j)`，与Target无GT预测边际`q_T(ŷ=i)`求`q_T≈C p_T`，在概率单纯形上求非负稳健估计；条件矩阵病态时失效。两方法都要独立实现、比较，并允许判定“不适用”。

### 6.2 原始logits、后验校准、数据泄漏控制

1. 统一`raw_native`与`raw_native+Bstar`两个基准，两个分支分别推导，禁止把Bstar logits称为未校准源域后验。
2. 对EM所需的`P_S(y|x)`，仅在有GT的Source**校准拟合集**拟合温度或其他明确的、轻量后验校准模型；Source验证子集严格分离，并报告NLL/ECE与分割mIoU。温度改变概率置信度但正温度整体缩放不会改变argmax；**不要把NLL下降等同于mIoU改善**。
3. Source1399生成稳定、按H300/R1099比例分层的**三份互斥图片列表**，例如`fit=699`、`select=350`、`holdout=350`；分配时固定seed/清单SHA，不按像素随机拆同一张图避免泄漏。所有分组数总和必须1399。
4. **先拟合、再选择、最后holdout一次报告**。若做多种模拟目标先验，所有模拟配方及随机种子要预注册，不能看holdout再调阈值。
5. Source类先验按有效GT像素统计并做小概率平滑；报告pixel-frequency加权带来的大场景偏差，使用图片为单位的bootstrap估不确定性，不能当像素i.i.d.计算虚假超小置信区间。

### 6.3 数学实现（8类，bias gauge固定）

记`p_s(c)`为Source真实有效像素类别先验，`r_i(c)=P_S(y=c|x_i)`为Source上校准的预测后验。迭代：

```text
初始化 p_t^(0) = p_s
E步: q_i^(k)(c) = [ r_i(c) * p_t^(k)(c)/p_s(c) ]
                  / sum_j [r_i(j) * p_t^(k)(j)/p_s(j)]
M步: p_t^(k+1)(c) = mean_i q_i^(k)(c)
终止: L1(p_t^(k+1)-p_t^(k)) < tol 或达到 max_iter
b_EM(c) = log(p_t(c)+eps) - log(p_s(c)+eps)
```

实现必须使用`logsumexp`数值稳定形式、eps/clip与预先限制的`max_iter`、收敛迭代日志、输入像素采样预算（分层覆盖图片，不只取容易的中心像素）、图片级bootstrap区间。`EM`本身预测的是分布先验，不保证最优mIoU。

BBSE：

```text
C[i,j] = P_S(pred=i | gt=j)  # 每列归一，使用Source有GT估计
q[i]   = P_T(pred=i)         # Target无GT预测类别面积
p_t    = argmin_{p>=0, sum(p)=1} ||C p - q||_2^2 + λ||p-p_s||_2^2
b_BBSE = log(p_t+eps) - log(p_s+eps)
```

**bias的可辨识性**：对于任何常数`a`，`argmax(z+b)=argmax(z+b+a·1)`。所有新bias统一保存`b_centered = b-mean(b)`，与`Bstar_centered`比较；优先看**最终像素决策、每类面积、Source模拟收益**，符号一致/余弦相似仅作辅助描述。还需验证若温度`T`不同，实际logit空间bias应与所用温度/score标度一致。

### 6.4 Source上的预注册可证伪模拟

- 构造**仅在Source持有GT的像素/图像上进行的label shift仿真**：从独立的已固定模拟池构造多组目标类比例（例如Barren高/低、Agricultural高/低、Vehicle高/低、均匀对照），用类别条件采样并记录像素所在图片；评估留出图像、避免同图泄漏。要明确“按像素重采样”是**测试Label Shift算法极限的合成条件**，不代表真实航拍存在同样偏移。
- `fit/select/holdout`三份必须图像互斥；如果模拟样本量不足则减少情景数，不从holdout反复试配置。
- 对每个模拟情景记录`p_t_GT`（这里只是**Source有GT构造的模拟真值**）、`p_t_EM`、`p_t_BBSE`、先验估计L1/JS误差、heldout分割mIoU变化、每类IoU及bootstrap 95%区间。
- 可计算oracle bias作为**仅用于Source仿真上限的参照**，绝不可将真实半决赛Target GT、官方线上分数或H75+Bstar当作oracle拟合标签。Muse“拿到oracle增益≥70%”可作为**探索性诊断**，但oracle增益接近0、为负或不稳定时该比值无意义，**不能单独作为PASS标准**。
- **PASS（仅技术可用）建议条件**：预先冻结的多个模拟情景中，先验估计与分割收益相对Zero/固定bias至少不反复大幅退化，估计结果收敛、稳定且能给出区间；若有模拟失败/BBSE病态/多情景相互矛盾，标`UNRELIABLE`。这仍不等于Target可用。
- **目标域分析**：与Bstar比较中心化bias、目标类别面积、margin变化和H75预测分歧，给出`BENEFIT_UNKNOWN_ON_TARGET`；没有真值不可证明EM正确。

### 6.5 是否生成官方提交候选

- H75+EM或H75+B_BBSE只有在**Source模拟可信+Target无塌缩+数值重建QA+规则合规+可清晰解释比Bstar何处不同**时进入`PROPOSE_FOR_ONLINE`；并记录对当前H75+Bstar的分歧率、分歧类别和预测样例。新bias不应无限网格搜官方线上分数。
- 任何模型的8维bias都是**模型/score协议绑定**的：新M2F不可照搬B3的Bstar或EM值而不校验score含义；EM或BBSE失败时，`Zero`只是一个未知性能的对照，**不是保守下界**。可优先保留多种未提交诊断版本，由负责人选择最可解释一个。
- 建议简单的候选bias不超过一次受控全局调整（如相对当前bias做收缩`b_new=(1-α)b_ref+αb_method`，α提前在Source模拟里定）；若新方法和Bstar强冲突，默认`NO-GO`，除非有额外Source外证据与队长特批。

**交付**：`EM_BBSE_METHODS.md`、`SOURCE_SHIFT_SIMULATION_REPORT.md`、`TARGET_PRIOR_DIAGNOSTICS.json`、`CALIBRATION_VERDICT.json`和候选bias JSON（含模型、协议、来源、许可、SHA）。

---

## 7. P2：Mask2Former+Swin-L E02 已有单模型探针

### 7.1 目的与事实

已知E02 B1 Source1399约**79.5654%**，C0约**79.5325%**（来自交接报告）；并无已确认的**当前半决赛3594官方线上分**。目标是验证**新架构在目标域是否优于B3 H75**，不是拿Source高分宣称成功。

### 7.2 执行步骤

1. **只读审计**：找到E02 B1/C0真实权重、代码、config、基础预训练与许可，逐项记录SHA、实际评估尺寸（历史SIZE=768）、标签变换、类别数与加载日志。定位不到或结构strict加载失败→`BLOCKED_ASSET`，不得下载其他文件冒名替换。
2. **统一可比较输出**：确认M2F `semantic_inference`/mask-classification输出转成8类**每像素语义分数**的具体计算规则；禁止把M2F的object query logits、softmax概率和SegFormer raw logits当作天然同标度的值。尤其全局bias拟合须明确是在`log probability`、语义logits还是其他score空间执行。
3. **40图canary速度与格式**：测实际推理尺寸和batch、单视角吞吐、显存；再估算完整3594张的实际总耗时，优先使用**该架构自身原评估协议**做首个可解释基线，而不是强行复制H75输入尺寸/TTA。若采用D4(8)、D4(4)或单视角(1)，分别记录，**不同TTA档位不可直接与H75 Source D4绝对数简单归因**。
4. **Source1399 G1灾变检查**：至少检查class mapping、全图valid、per-class IoU、Vehicle和Barren；Muse的`新架构<65 / 任一类IoU<10 / Barren<30`可作为**预注册安全警报**，不是目标精度排名器。若未达到之前报告的B1性能，需要解释为何评估协议不同而非认定模型失效。
5. **Target3594打包**：根据已验证的架构输出做1024灰度PNG与3594根目录ZIP；候选可生成`M2F_RAW`和另一个**经过自身EM/BBSE验证的校准版本**，但**一次线上提交只选其中一个**。候选输出附每图SHA和`qa_submission.py`结果。
6. **上线决策**：由队长根据P1风险审计、可复现性、剩余时间、每日额度批准人工提交。上线上分后，按以下预注册处置记录：`<70.0`不再耗资源；`70.0..70.998665`留备但不是冠军；`>70.998665`登记新冠军候选、启动P0新链复现预检。不能用`Source79.5654`预测具体线上分。

**时间预算是可测约束，不是“预计2.5小时”承诺**。测完40图由Claude报告`ms/image`、速度范围和完整估算，若不符合截止或影响P0，降低TTA/缩小实验或终止。

**交付**：`M2F_ASSET_AUDIT.md`、`M2F_SCORE_PROTOCOL.md`、`M2F_40_CANARY.json`、`M2F_SOURCE1399.json`、`M2F_TARGET3594_RISK.md`、候选ZIP+QA+SHA清单、`M2F_ONLINE_STATUS.json`（正式上传前状态为`NOT_SUBMITTED`）。

---

## 8. P3：SAM3新语义监督——先证明“能纠错”，再动Target

### 8.1 原则与合法性门槛

- SAM3是**外部预训练模型产生的语义/实例候选mask**，不是半决赛GT。可作为训练期教师候选，但**正式最终提交只能用获准的单个分割模型**；禁止线上直接融合H75+SAM3分割结果。
- 在获得官方原始问答核验以及适用SAM3权重许可/预训练数据相关限制验证前，**不启动Target全量mask拟合或用其训练正式提交模型**；Source离线方法探针也需遵守该公开权重的使用条款。
- 禁止人工修正Target mask后回训，禁止人工建立Target真值。可以人工目检用于**异常审查**，但不得将其编码为训练标签或优化目标。

### 8.2 Source1399：两级可靠性审计

**第一级：基础能力/成本筛选**
- 先用固定的Source分层样本（建议按H300/R1099抽取50~100张，并记录seed与ID，数量可由吞吐调整）测试Prompt和加载流程。对每一候选英文/中文Prompt留档，例如Vehicle/Building/Water等映射需要独立校验；**对象实例mask不能天然等价于八类地表覆盖语义mask**，Road/Barren/Agricultural尤需警惕。
- 每类计算mask-to-class的TP/FP/FN/Precision/Recall/IoU及被提示图像覆盖率；不能只在“目标存在且生成mask成功”图片上算Precision而忽略失败实例和负例图。
- Muse筛选阈值`Precision≥0.70且Recall≥0.20`作为**探索性候选线**，过门仅表示值得继续，不意味着可用于伪GT训练。

**第二级：相对H75的独立纠错能力**（核心）
- Source上保持H75 `D4+Bstar`固定，SAM3输出掩码与Source真实GT三方对齐到同一1024图像，不引入不同尺寸引起的边界伪差。
- 对`SAM3_pred != H75_pred`分歧区域，统计四个像素集合：
  1. `SAM正确 / H75错误`（实际帮助数`help`）；
  2. `SAM错误 / H75正确`（实际伤害数`hurt`）；
  3. 两者都错误；
  4. 仅在类别prompt可判定区域做评估的coverage。
- 定义`net_correction = (help - hurt) / max(1, disagreement_pixels)`；报告逐类、近边界/内部、小目标/大区域、H300/R1099分层，并按**图片级bootstrap**给出95%置信区间。**不把这项Source纠错率外推为Target的真实纠错率**。
- 为候选监督另设保守质量门槛（供讨论，不是事实）：在**独立Source审核集**，准入mask类的高置信伪标签Precision≥0.85，且其图片级bootstrap 95%下界尽可能≥0.80，`net_correction`不为明显负值并有足够的分歧覆盖；达不到则标`PSEUDO_LABEL_NOT_TRUSTED`。阈值依据数据可靠性和样本量固定后不得根据Target结果挪动。

### 8.3 Target3594可选实验：先候选bias，再考虑蒸馏

1. **许可PASS且Source审计PASS后**，仅对经过可靠性验证的类生成Target候选mask及模型输出confidence；为mask标记`teacher_id/prompt/version/threshold/shape/image_sha/consent_reference`。无GT目标不生成“accuracy”列。
2. **B_sam**：使用P1 H75 `raw_native` logits，在固定、已验证的score协议中拟合**单个全局8维bias**；优化目标为可信SAM3区域的弱代理一致性+Source有GT保持项+相对Bstar的L2/幅度正则。**禁止以全Target可疑伪标签强迫H75全面改判**。
3. 预注册候选优化参数与幅度范围（例如每维相对Bstar限制`|Δb_c|≤0.5`先做保守对照；若需要放宽必须先Source模拟论证），输出bias中心化版本和最终完整向量。拟合时不能用来源于H75自身的预测当“独立教师纠错”。
4. **先离线诊断**：Source1399新的逐类变化（尤其Barren假阳性）、Target三部分面积/分歧、与Bstar/B_EM比较，只能说明候选风险，**不能写“已经提升Target”**。
5. **只有B_sam有合理目标域新信息证据、源域风险可控、正式SAM3用途合规且P0不受影响时**，提交`H75+B_sam`人工候选评审；官方实际分数决定成功/失败。
6. **蒸馏后置**：仅在mask独立纠错能力强、bias路线有合理证据、时间许可下做短受控训练；H75初始化、Source有GT CE锚定、只更新事先约定范围；通过正确性/伪标签精度筛选目标区域；保留NoTarget教师/NoSAM对照，保证不是重复自蒸馏。

**停线**：许可证/老师确认不通过、无类别通过审计、help≤hurt、覆盖太少、对齐错误、目标预测塌缩、算力影响P0/截止。失败时不强行报“生成了3594 mask所以有价值”。

**交付**：`SAM3_LICENSE_APPROVAL.md`、`SAM3_SOURCE_METRICS.json`、`SAM3_DISAGREEMENT_CORRECTION.md`、`SAM3_PROMPT_MAPPING.yaml`、`TARGET_MASK_MANIFEST.json`（仅若许可允许执行）、`B_SAM_CANDIDATE.json`、`SAM3_VERDICT.json`。

---

## 9. P4：DINOv3-SAT限时A/B，不提前承诺长训

### 9.1 实验事实和新假设

- 已有的是`DINOv3-L/16 LVD通用版 + PMT` **2000步**，Source1399有掩码 **68.531338%**、无掩码 **66.044216%**，并非SAT实验。
- 新假设：在相同PMT代码/数据/训练步数/seed/优化器/掩码协议下，把通用LVD的权重和输入normalize换成**SAT-493M卫星域预训练**，可能改善遥感/航拍特征。此为假设，**不是已证实提分**。
- 官方开源资料可核对权重标识`facebook/dinov3-vitl16-pretrain-sat493m`；SAT均值约`(0.430,0.411,0.296)`、标准差`(0.213,0.156,0.143)`，以实际模型卡和源码核对；不要只根据名称替换一个未经核验的文件。公开模型卡显示其**专用许可证**，不能笼统认为所有公开权重都允许比赛使用。

### 9.2 第一阶段A/B

1. 核验LVD/SAT模型配置完全同构、权重键映射、模型卡许可、预训练权重实际SHA；复用已有PMT训练代码，先做单批前向、loss/梯度有限性与显存smoke。
2. 以既有LVD 2000步配置为准锁定训练协议：除`backbone checkpoint + 对应normalize`之外不改学习率、数据顺序、随机种子、decoder、训练参数组；如两次运行的代码/环境不同，必须明确A/B不是严格单变量。
3. 在授权独占GPU上完成最多2000步SAT探针，记录Source1399**与LVD同协议**的mIoU/per-class、训练loss、耗时、checkpoint SHA和验证输入manifest。
4. Muse阈值`Source≥72继续；70–72加测；<70停止`可当**成本控制型screening**，**不得解释为预测目标域成绩**。同时给出Barren/Vehicle等弱类与Target风险报告；如果模型结构/许可证/速度不满足，直接停止。
5. 全量8000步、LoRA、Target蒸馏或3594线上候选都需要**三项前置**：A/B通过技术与资源筛选；M2F线上结果已知且无更紧急冠军材料风险；截止前能完成**训练+推理QA+复现归档**。若不满足，归档A/B，不强行开跑。

**交付**：`DINO_SAT_ASSET_LICENSE.md`、`SAT_AB_CONFIG_DIFF.json`、`SAT_2000_TRAIN_REPORT.md`、`DINO_SOURCE_AND_TARGET_DIAG.md`、`DINO_VERDICT.json`。**没有官方分数时状态必须保持`NOT_SUBMITTED`**。

---

## 10. 统一QA与实验判据（每条线共享，不重复造流程）

### Gate G0：身份、格式、数值协议（硬闸）

- 输入图片/标签/模型/SHA对应；全3594清单完整；输出像素1..8且`uint8 PIL L`、1024²、名称精确匹配、ZIP根目录无多余文件；无NaN/Inf，全部可完整读取、ZIP CRC通过。
- 正式预测的模型、TTA、bias、upsample、class mapping与manifest**唯一绑定**；同图重复运行差异以数值报告量化。
- 历史`qa_p0b_zip.sh`ROOT路径bug需要单独修复，使用原`qa_submission.py`显式路径实现可复用调用，不把包装器失败忽略掉。
- **未过G0，一律不得建议上线**。

### Gate G1：Source有GT灾变检查（不是择优排名）

- 每类IoU、全体混淆矩阵；B3衍生物Source1399严重下降（Muse建议`<72%`）；新架构`<65%`或关键类严重塌缩/任何类<10%、Barren<30%：标红并人工复核。
- 这些是**启发式、预注册的灾变阈值**，不是“超过就必能冲71%”。Source1399更高也不得单独当推荐上线条件。

### Gate G2：Target无GT风险与可证伪方法验证

- **核心必做**：P1目标类面积/分歧与已有H300/R1099分层有据可查；源域EM/BBSE留出模拟有报告；候选score语义已核查。若另行获批SAM3监督/偏置路线，其Source分歧纠错和许可状态必须过该路线专属门。**扰动稳定性需有已有单视图数据或成本核准；§5.3域分组/特征距离/重加权从未被授权时不得视为G2失败。**
- 类别面积极端塌缩、图片大片分歧、输入域处理错误、SAM mask无可验证映射→`RED`；没有充分证据→`UNKNOWN`，不是擅自放行。
- **无需强制“候选与H75预测差异必须0.5%~12%”**：差异比例过大时要排查，差异太小不代表不可能有收益。

### Gate G3：合规与提交前放行

- 独立权重/许可/预训练数据授权；官方原始问答或充分规则依据；最终单模型（可为离线参数平均后单checkpoint）+ 许可TTA + 固定bias；全链可追溯。
- 用户/队长**人工批准**使用一次排行榜额度；Claude只提供`SUBMISSION_CANDIDATE_CARD.md`和本地ZIP路径/SHA/完整性，不自行提交。
- 得到线上成绩后录入带截图/提交时间/模型SHA/ZIP SHA的`ONLINE_SCORE_LEDGER.csv`，只在同赛段比较；官方分数为唯一在线成绩判断。

### Gate G4：冠军切换与最终交付

- 新候选**官方分数严格更高**并且能用该权重/脚本/环境重建其预测，才可能替换H75。若成绩更高但不能按规则复现，先解决复现/合规并保留H75正式材料。
- 按11-01 15:00内部冻结进度、11-01 20:00官方ZIP截止、11-02 20:00材料截止设置提醒；禁止赛后补交幻想。

---

## 11. 任务与资源调度——可以并行，但不可争GPU

**默认优先顺序**：P0 Docker/可复现材料 **≥** P1原生缓存+金丝雀及EM/BBSE模拟 **≥** P2 M2F探针 **≥** P3 SAM3 Source两级审计 **≥** P4 SAT小试验；**§5.3域分组/距离/Source重加权排在四项核心交付完成之后，默认不做**。若P0已经E2E完成，可释放GPU给P1/P2；SAM3和SAT不抢占P0。

| 北京时间 | 主线动作 | 是否占官方提交额度 | 必须交付/决策 |
|---|---|---|---|
| **10/28** | Stage0资产/许可核验；**队长发送点名SAM3确认问并存发送记录**；P0搭建；P1原生缓存canary；M2F资产审计（**不开始新Source视觉分组**） | 不必提交；旧CTRL5400/soup原则上不优先 | `AUDIT`、`SAM3_CONFIRMATION_STATUS`、`P0 STATUS`、`P1 CANARY` |
| **10/29** | P0干净Docker；P1 EM/BBSE模拟&基础目标诊断；P2 M2F 40图速度+打包候选；SAM3在许可框架内Source两级审计 | 如P2 QA通过且人工同意可用1次；B_sam合规且成熟再考虑 | `EM_VERDICT`、`M2F_VERDICT`、`SAM_SOURCE_GATE` |
| **10/30** | P0完成全量复现；P2已有线上结果则决策；SAM3 B_sam候选审计；P4有空闲卡才做SAT2000 | 仅对过Gate且有研究增量的候选手工提交 | `P0 E2E`、更新冠军材料方案 |
| **10/31** | 成绩候选收尾；所有高分候选脚本/ZIP和模型可复现审计；技术方案PDF形成完整草案 | 不强制填满额度 | `FINAL_CANDIDATE_AUDIT` |
| **11/01 15:00** | 内部冻结最高分候选与对应材料清单；其后以防守为主 | 只有明确更强且来得及复现才经人工例外批准 | `WINNER_FREEZE_MANIFEST` |
| **11/01 20:00** | **官方预测ZIP截止** | 截止后0 | 正式排行榜+ZIP证据 |
| **11/02 20:00** | **官方作品/网盘截止** | 无 | 最高分材料上传成功、链接可访问截图 |

**额度计划不等于已提交记录**。每天2次、未用可以空置；不可为了用满额度牺牲QA。§5.3可选基建不抢占P0、缓存/金丝雀、EM/BBSE模拟、M2F与SAM3审计的工时。不要把10/31“新平台全6996复训”当成默认必做——其会破坏现行固定验证划分，而且未必能在截止前形成真实优势，除非另有严格研究设计、合法数据使用证明和足够时间。

### 11.1 GPU调度和停止条件

- Claude先报告`hostname/GPU UUID/physical GPU/显存空闲/他人进程`；GPU索引在不同`CUDA_VISIBLE_DEVICES`下会变化，不能仅凭“GPU0/3”判断拥有权。
- 每项训练/推理启动前写`RESOURCE_REQUEST.json`含预估峰值显存、预计消耗时间**来自实际smoke**、可用存储、抢占/退出方案。没有GPU授权或容量不足→不启动。
- 距预测截止不足以完成“训练+QA+上传+复现”时不再启动长训；若P0红灯，所有新增研究线可以暂停。
- 长任务必须日志落盘、PID记录、可安全中止；聊天CLI报错/掉线不等于真实任务退出，要核对进程和产物。

---

## 12. 统一脚本、配置和自动测试清单（Claude Code要真正实现）

### 12.1 共享Python模块/CLI（建议名称，可遵照现有工程小改）

```text
code/
  audit_assets.py               # 各关键sha/路径/输入清单/资源只读报告
  build_fixed_splits.py         # Source1399 699/350/350稳定图片分层
  cache_scores.py               # H75/M2F分数缓存，保存真实score定义
  replay_from_scores.py         # raw→bias→upsample→argmax→PNG
  evaluate_source.py            # 混淆矩阵/IoU/precision/recall/分组
  make_domain_bins.py           # OPTIONAL/DEFERRED：§5.3被单独批准后才创建
  compare_target.py             # 与H75逐像素分歧、类面积、稳定性
  prior_em.py                   # 稳定EM+收敛日志
  prior_bbse.py                 # simplex约束回归+病态诊断
  simulate_label_shift.py       # 有GT Source合成分布偏移与留出评估
  audit_sam3_source.py          # SAM3 prompt→mask映射+GT验证
  audit_disagreement.py         # SAM3对H75 help/hurt与bootstrap
  fit_b_sam.py                  # 合规后全局8维bias限制拟合
  probe_dino_sat.py             # 严格A/B参数diff和短跑测试
  qa_submission.py              # 复用原始已验证QA或独立严格等价实现
  make_manifest.py              # checkpoint/script/zip/protocol绑定
  tests/                        # 必须含下面关键单元/集成测试
configs/
  h75_protocol.yaml
  domain_bins.yaml              # OPTIONAL/DEFERRED：§5.3被单独批准后才创建
  em_bbse.yaml
  sam3_prompts.yaml
  m2f_protocol.yaml
  dino_sat_probe.yaml
```

不用为形式过度重构现有训练仓库。**先复用可验证的官方H75代码**，新增模块放到独立目录；有功能相同的现存脚本应记录原名、哈希和差异，而不是重造一份未经验证的推理链。

### 12.2 最低自动测试集合

1. **标签映射测试**：raw GT 0→ignore255、1..8→0..7；推理0..7→官方1..8；255不参与Source评分；提交PNG uint8模式L。
2. **D4逆变换测试**：各种几何组合可在合成非对称张量上原样还原；8视图顺序固定，确保旋转/翻转逆序正确。
3. **后处理顺序测试**：raw native → 加bias → bilinear upsample → argmax，不得出现低分辨率先argmax再nearest上采样；`align_corners=False`。
4. **bias gauge测试**：`argmax(z+b)==argmax(z+b+constant)`，所有对bias的比较使用中心化向量及像素结果；对于softmax也相同。
5. **EM单元测试**：合成已知纯label-shift后验数据，估计类别先验误差/归一性/数值收敛；有0频类与极端logits时不崩溃。
6. **BBSE测试**：已知可逆混淆矩阵可回收近似先验；奇异/病态矩阵必须发出`ILL_CONDITIONED`而非产生负概率并宣称成功。
7. **Source验证拆分测试**：1399张图片三级集合无交集且总数1399，同图片不同像素绝不同时出现在fit和holdout。
8. **Source混淆测试**：`sum(confusion_matrix)==有效GT像素数`，且逐类`TP/FP/FN`可从矩阵一致还原（注意多类错误像素同时计入一个FP和一个FN，**不能**把`sum(TP+FP+FN)`等同总像素）；混淆矩阵按图累加、类别缺失处理有明确策略。
9. **缓存重建集成测试**：40张canary对已评分H75输出，与FP32/FP16分别测差异；任何迁移环境变化有单独报告。
10. **提交包QA**：3594唯一同名PNG、1024、uint8 L、值1..8、ZIP根目录/CRC/样本总数/文件SHA全部校验；零失败才可生成候选卡。
11. **Source/Target隔离**：有GT的Source只读清单与Target3594无GT清单严格分离；目标域不能从目录中意外读取GT路径。
12. **状态机检查**：未上线时不能将`online_miou`写成数值；不能将`risk_score`写作`target_miou`；只有用户或正式回执可把状态从`READY_FOR_MANUAL_SUBMISSION`改为`OFFICIAL_SCORED`。

---

## 13. Manifest、状态机与上线审查格式

### 13.1 每个候选必须独立保存 `candidate_manifest.json`

```json
{
  "experiment_id": "H75_EM_V1_20261028",
  "status": "OFFLINE_DIAGNOSTICS_ONLY",
  "owner": "<实际负责人>",
  "model_family": "SegFormer-B3",
  "checkpoint_path": "<核实后的只读权重绝对路径>",
  "checkpoint_sha256": "<实际sha256>",
  "input_target_manifest_sha256": "<3594输入manifest sha>",
  "source_split_manifest_sha256": "<Source1399清单sha>",
  "code_manifest_sha256": "<实际代码清单sha>",
  "protocol_id": "h75_d4_native_fp32_bias_upsample1024",
  "tta_views": 8,
  "score_stage": "native_fp32_before_bias",
  "bias_vector": null,
  "label_mapping": "model_0to7_plus1_official_1to8",
  "source1399_miou": null,
  "target3594_risk_verdict": "UNKNOWN",
  "online_miou": null,
  "prediction_zip_path": null,
  "prediction_zip_sha256": null,
  "source_evidence": [],
  "license_approval_refs": [],
  "qa_status": "NOT_RUN",
  "created_at_cst": "<执行时填写>",
  "notes": "所有null必须以实际执行后填充，不允许猜测。"
}
```

状态机：

```text
PLANNED → ASSET_VERIFIED → OFFLINE_DIAGNOSTICS_ONLY
  → PASSED_G0_G1_G2_G3 → READY_FOR_MANUAL_SUBMISSION
  → MANUALLY_SUBMITTED → OFFICIAL_SCORED
  → (若超越且复现通过) WINNER_CANDIDATE → FINAL_WINNER

任何状态 → BLOCKED / NO_GO / ARCHIVED
```

`READY_FOR_MANUAL_SUBMISSION`绝不代表Claude已上传。报告提供`下一步人工操作`，不对外宣称已成功。

### 13.2 申请一次官方额度时的固定卡片

```text
experiment_id:
截止/当日已用额度（依据人工核实）:
模型+权重SHA:
推理脚本/config SHA:
唯一ZIP路径/大小/SHA:
同赛段已知最高基线: H75+Bstar=70.998665%
Source1399与协议:
Target风险诊断 (无GT):
G0格式/复现PASS:
G1灾变检查:
G2方法可靠性:
G3官方/许可证合规:
相比H75预测改变的面积/类别:
新信息来源和潜在风险:
是否能在剩余时间内复现和交付:
建议: SUBMIT / HOLD / NO_GO
正式平台上传: 仅由队长/授权队员人工操作
```

**台账：** `ONLINE_SCORE_LEDGER.csv`必须带`stage=semifinal_3594`、日期、提交账号责任人、实验ID、ZIP SHA、官方分数（若有）、截图位置，不能在未出分时写假的“0.0”或“预计71+”。

---

## 14. 失败、争议及止损处理

| 情况 | 立即动作 | 后续结论边界 |
|---|---|---|
| H75 SHA/推理协议不符 | 停研究线，优先P0原件定位 | 不用可能错误的raw cache建立所有后续结论 |
| EM在Source仿真不稳或BBSE病态 | 标`NO_GO`，保留Bstar | 不以目标无GT面积让EM“洗白” |
| EM与Bstar中心化方向反对 | 保持探索，不自动提交 | Bstar非GT；矛盾本身需留痕 |
| Target各类预测大幅塌缩 | G2 RED，查前处理/映射/校准 | 不能通过提交分数反向找借口 |
| M2F score语义/耗时无法保证 | 先修接口或降TTA，仍失败就停线 | 不能把M2F概率直接当H75 logits加bias |
| SAM3 Source整体不错但分歧纠错为负 | 停止SAM蒸馏/B_sam | 整体Precision不能证明对H75有新增知识 |
| SAM3合规或许可不通过 | 禁止相应目标训练/提交 | 只保留合法范围内的审计资料 |
| SAT @2000指标未过成本门槛 | 停长训，归档 | 不推断SAT线上真实表现 |
| 新候选Source涨、Target风险混乱 | 人工复核或不提交 | Source没有授权预测Target mIoU |
| 新候选线上超过H75但材料不可复现 | 暂不切冠军，优先修复 | 官方可能取消不可复现成绩 |
| 截止临近/P0未通过 | 立即停止长训/探索，优先H75材料 | 保住已确认70.998665% |

---

## 15. Claude Code 每阶段应返回的简报，不要只给口头“成功”

每阶段执行完成必须至少生成并回复：

```text
[阶段] / [时间CST] / [主机+GPU(若有)]
1. 已验证事实（命令、输入、SHA与原始报告路径）
2. 新产物的确切路径与SHA
3. 关键数值：canary、分类指标、分布审计或资源耗时
4. Gate结果（PASS / AMBER / BLOCKED / NO_GO）及触发依据
5. 哪些结论仍只是推断/无GT代理
6. 是否影响P0冠军保险
7. 建议的下一项**单一**动作和需要人工批准的事项
```

不允许只回“脚本已写好，理论可用”；也不允许因为聊天CLI退出码异常就断言训练失败——先查实际进程、checkpoint和日志。每次重要阶段将报告写进`09_reports/`并更新台账，便于队长和其它AI对照。

---

## 16. 第一条执行指令模板（**需队长另行明确发出，文件本身不授权**）

**队长决定启动Stage 0时**，可单独发送以下指令给Claude Code（不要把“发送文件”当成已经执行了这条授权）：

> **开始执行 Stage 0，只读资产与合规审计。** 你现在是AIC2026无人机低空航拍语义分割半决赛的工程执行助手。请完整阅读我提供的《AIC2026_UAV_半决赛_ClaudeCode_融合执行方案_v1.1_20261028.md》；将10月28日交接汇总和Muse《PLAN_FINAL_20261028.md》仅作来源背景，**有冲突以v1.1执行门槛、经核实原始证据和最新官方规则为准**。本次只授权Stage 0：不得正式训练、不得下载或全量运行未经批准的外部模型、不得自动向比赛平台提交。核对H75冠军权重、已评分ZIP、原生推理协议、Source5597/1399、Target3594、现有M2F/PMT资产、GPU/磁盘、官方十问原文及各许可证；对发现的路径和SHA输出机器可读清单，区分`VERIFIED / NOT_FOUND / PENDING_APPROVAL`。创建独立工作目录；输出`00_PREFLIGHT_AUDIT.md`、`ASSET_MANIFEST.json`、`OPEN_ISSUES.md`及命令/证据路径。**在`00_audit/`创建SAM3专项确认问与答复状态记录；提醒队长10月28日亲自向官方发问，Claude不可代发、不可虚构官方答复。** 不启动§5.3域分组/特征距离/Source重加权。若SSH或数据缺失，记录阻塞，不猜测、不修改旧数据。完成后给我一个Stage0审计简报，**等待下一次明确阶段授权**。当前已知官方3594最高成绩H75+Bstar=70.998665%，11-01 20:00截止预测，11-02 20:00截止材料。

无论Stage 0是否PASS，进入P0干净环境复现、P1大规模缓存、P2/P3/P4 GPU推理或任何正式提交之前，均需队长按本方案权限另行明确授权；Stage0的`READY`不等于任何线上`SUBMITTED`。

---

## 附录 A：研究与官方资料（不是比赛许可凭证）

- Saerens M., Latinne P., Decaestecker C. (2002), *Adjusting the Outputs of a Classifier to New a Priori Probabilities*, Neural Computation. DOI: <https://doi.org/10.1162/089976602753284446>。EM目标先验修正的理论来源。
- Lipton Z., Wang Y., Smola A. (2018), *Detecting and Correcting for Label Shift with Black Box Predictors*, ICML. <https://proceedings.mlr.press/v80/lipton18a.html>。BBSE和条件混淆矩阵的前提、病态风险。
- Meta SAM 3官方实现：<https://github.com/facebookresearch/sam3>；官方论文概述：<https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/>。需独立核对实际权重与许可。
- DINOv3官方仓库：<https://github.com/facebookresearch/dinov3>；卫星预训练模型卡：<https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m>。SAT均值/标准差来自官方仓库；模型卡注明专门的`dinov3-license`，**公开可获取不代表比赛可自动使用**。
- 赛事相关原件：`【发布版】算法挑战赛道复赛获奖名单及半决赛赛程安排通知.pdf`、`附件2：AIC算法挑战赛道半决赛作品提交要求-1.pdf`、官方赛题页面与赛题QQ群最新通知。**赛题最新官方规则高于本研究执行方案**。

## 附录 B：计划与事实的明确边界

- 文档中所有未来性能数值、阈值、进度和产物名称都是**计划/建议**，不是已经取得的新成绩。
- 原始远程checkpoint、GPU占用、剩余日提交额度、十问答复原文、SAM3权重许可、SAT权重SHA、M2F精确推理接口均需Stage0核实。
- **SAM3专项问答在得到原始回复前始终是`PENDING`**；10月28日是人工发问目标日，不是已发或已批的事实。`00_audit/`必须记录问句/发送情况/原始证据。
- **§5.3分域、距离与重加权是可选后置项**，不是当前必须完成的Stage0/P1交付，不得因它拖延H75复现、原生缓存、EM/BBSE、M2F、SAM3。
- **没有Target3594 GT，不能声称校准/伪标签选模预测了真实目标域mIoU，也不能把人工目检当目标真值**。
- **新方案成功不改变复现义务**：最终获奖必须绑定真实官方最高分所对应的单模型、源代码、环境、提交ZIP与可访问材料链接。