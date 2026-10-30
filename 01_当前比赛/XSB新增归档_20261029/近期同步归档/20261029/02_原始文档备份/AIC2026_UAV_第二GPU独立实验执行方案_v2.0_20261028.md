# AIC2026 UAV半决赛｜第二张GPU独立实验执行方案 v2.0

> **制定日期**：2026-10-28，北京时间（CST，UTC+8）  
> **文件定位**：面向**新开的Claude Code独立对话**的研究与执行任务书；不是立即执行授权，也不是之前v1.3的替代文件。  
> **新增主线**：第二张经核验且获准使用的RTX 4090运行**Mask2Former+Swin-L强化训练**；条件允许才开展**DINOv3-SAT/LVD + 冻结骨干Rein-G + Mask2Former的严格A/B**。  
> **唯一实际交付目标**：在2026-11-01 20:00前，力争获得**半决赛3594张无标签Target的、符合官方规格并通过QA的单模型预测ZIP**，供队长**手工**决定是否上传。实验可能NO-GO，**不保证新ZIP或新线上高分**。  
> **绝对不包含**：Docker/洁净环境复现、PDF技术报告、网盘材料、最终作品交付、比赛平台自动上传、未经授权的跨GPU/跨主机运算。上述事项独立留待后续安排，**不得为它们占用本线资源**。  
> **外部证据属性**：Muse《其他候选方案检索报告》为研究线索，GOOSE、Rein/Rein++为不同任务的公开研究，**不能直接把它们的论文mIoU收益或训练时长套用到本赛Target3594**。本计划中的超参数是受控起点，需以现有工程、许可证及200步实测确认。

---

## 0. 与现有主线的边界：两套Claude，两套看门狗，一条额外GPU

| 对象 | 现有主线v1.3（不修改） | 本文件第二GPU线（独立） |
|---|---|---|
| 服务器 | `teacher-4090`（实测hostname `ubuntu-Super-Server`，`jhy`账号） | **优先同一服务器**，但不同的获准物理GPU |
| 已确认GPU | **物理GPU3**，UUID `GPU-32853ccf-4c4f-1328-8d75-04efe9d5298f` | **未选定**：先核验GPU0/1/2实况、UUID、利用权、显存和正在运行的其他人任务 |
| 现有进程 | 原有GPU3推理/缓存/校准任务 | 禁止停止、接管、迁移、重启或更改它们 |
| 看门狗 | `ziponly-watchdog.service`、目录 `/home/jhy/aic2026_ziponly_20261028/` | **新名称/目录/服务/锁/队列**，只管理本文件GPU |
| 核心工作 | H75 logits、EM/BBSE、现成M2F E02探针和ZIP、SAM3受限审计 | **新M2F长训**；有余力时SAT/LVD+Rein-G新实验 |
| 产物 | v1.3自己的候选ZIP、评测、manifest | 本线独立checkpoint、日志、指标和候选ZIP，禁止互相覆盖 |
| 线上提交 | 队长人工处理 | **同样仅出ZIP，不自动提交**；仍共享每日最多2次的官方额度 |

**现有事实基线**：半决赛同一3594张线上最佳`H75+Bstar+D4-TTA8=70.998665%`；其它已评分`T3-FULL-ST=70.990138%`、`CTRL@3594+Bstar=70.318575%`、`H75+Zero=67.979564%`。M2F E02还没有经核实的半决赛线上分。本线**不得把Source1399的高分当成Target3594高分**。

### 0.1 唯一职责、一次性授权和节约tokens

- 新Claude首次只做**只读Stage A资产/GPU审计**；文件本身不授权GPU训练。队长后续**可以一次性授权完整M2F受控任务**（200步+达Gate后的固定预算长训+候选ZIP），之后无需逐步聊天批准、由服务器看门狗按既定规则推进。
- 阶段中正常进度全部保存在`state.json`、日志、manifest；**Claude不每隔N分钟聊天轮询、不主动发送逐epoch消息**。
- Claude新会话部署看门狗后提供一次“启动/状态查询方式”回执；任务结束或致命异常时保存最终报告。用户以后主动询问时，读取状态报告**一次即可**。不得依赖Claude会话保持在线。
- **Rein方向和新的大规模权重下载**没有因M2F一次性授权而自动获准；须先满足独立模型许可证/规则核验及后续授权。
- 任何自动任务**不能自动打开线上提交页面或上传候选ZIP**。

---

## 1. 数据、格式和证据的硬约束

### 1.1 唯一数据定义

- Source共**6996张有GT**，既定`train5597 / val1399`；验证集H300=300，R1099=1099。训练只能使用**5597官方训练GT**，不得合并1399训练（除非用户另行启动独立新研究，不属于本版）。
- Target半决赛共**3594张无GT**：test1=500、test2=1300、test3_new=1794；使用已核验清单/`subset`字段统一映射，不能另加1800张，也不得人工给Target标注再回训。
- GT原始`0`表示ignore，官方`1..8`→模型`0..7`，训练ignore为`255`。类别顺序：`Background, Building, Road, Water, Barren, Vegetation, Agricultural, Vehicle`。
- **正式预测ZIP**：恰好3594张与官方图片同名PNG；`PIL mode=L`、`uint8`、1024×1024、预测像素`1..8`、ZIP根目录平铺、无额外目录与文件、CRC通过、全文件名集合一一对应，保留ZIP SHA256和模型/代码/config/输入ID清单SHA。
- 原M2F E02文件线索（并非审计通过）：`/home/jhy/mmseg_uav/work_dirs/Exp00_M2F_SwinL_Baseline/`。原v1.3工作区与数据目录应由新Claude只读定位，而非从聊天记录猜测绝对路径。

### 1.2 赛事与预训练使用许可

- 使用**官方数据与允许的公开预训练权重**。Swin-L backbone必须核查原始发布者、下载地址、权重SHA、使用许可证、上游预训练集；尤其不能悄悄载入外部有监督**分割头**权重来训练正式候选，除非能够证明该特定用法符合赛事规则。
- DINOv3-SAT/LVD权重页面声明`dinov3-license`，并可能要求账户同意协议；公开可下载≠自动允许本竞赛使用。无清楚许可/官方认可则DINO路线仅做合规记录、不将相关模型用于正式提交候选。
- **禁止本线**：SAM3/SAM2作Target教师、外部数据或人工标注Target、商业闭源模型API、在线多模型融合、用官方线上分数反复搜索最优超参。
- 训练过程中EMA是**同一模型权重随时间平均**，最终导出单份EMA权重；属于训练技术，不是在推理时运行多模型；具体赛事准许情况以最新官方答复为准。

### 1.3 科学验证事实边界

- Source1399 mIoU可以用来查类映射、损失和灾变，也可用固定协议审计各类IoU，但**不能单独决定目标域最优checkpoint**。历史SegNeXt Source82.06而线上约67，CTRL也有反转。
- Target3594无GT，所算entropy、类面积、预测分歧、稳定性等**只能作为风险信号**，不得生成“预测Target mIoU=XX”或用几个旧线上点训练伪评分器。
- 用固定步数的EMA checkpoint作为主提交候选，而不是按val-best挑步数。**固定步数也不免于Source灾变/格式/合规闸门**。
- GOOSE给出的“预训练差4.7pp、EMA约+0.6、裁剪收益约+2.2”只用于设假说；Rein/Rein++在其他DG基准上的提升不作为本赛收益预估。GOOSE强DINOv3模型为端到端微调，不是Rein的受控证明。

---

## 2. Stage A：第二GPU与模型资产审计（只读，不启动长训）

### A1. 选择第二张4090

1. `ssh`连接现有主服务器；只读检查`hostname -f; date -Is; nvidia-smi -L; nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv; nvidia-smi`及`ps`；记录GPU0/1/2上**已有用户进程、实际归属和授权状态**。
2. **排除物理GPU3和其UUID**；选择一张经授权的RTX4090，保存`SECOND_GPU_UUID`、`SECOND_GPU_PCI_BUS_ID`、启动时`CUDA_VISIBLE_DEVICES=<实际UUID>`，在Python中再次用NVML/环境/实际device属性核对。`CUDA_VISIBLE_DEVICES`设置后PyTorch逻辑`cuda:0`≠物理GPU0。
3. 如果空闲GPU都属于他人、权限未明确或不能保证不冲突，输出`BLOCKED_NO_APPROVED_GPU`，不能抢占、杀进程或偷用GPU3。
4. 核对共享数据盘IO、独立工作区配额、可用空间、训练权重和中间日志存储预算，避免第二线拖慢GPU3的3594张推理。磁盘占用预计须包含模型EMA+optimizer checkpoint、至少2~3份固定步权重、可能的候选结果；留≥20%空间余量，实际可根据容量评估。

### A2. 只读审计E02与预训练来源

1. 寻找E02 B1/C0原checkpoint、optimizer、scheduler、EMA是否存在、配置、训练命令、seed、数值格式、标签映射、MMSeg版本、历史Source1399的79.5654/79.5325是否能与具体权重对应。
2. 判断E02 2000步是否为“短训练、完整长训练LR schedule的一段”还是“2000步为终点的短LR schedule”。**不要把E02权重来源不清的训练状态带入新主线**。
3. 默认**从经许可的原始Swin-L公开backbone权重干净启动**，Mask2Former decoder/head按明确配置新初始化。只有同时满足“E02来源/数据/优化器/EMA/RNG完整可信、学习率日程原本就为同一长训配方且续训有明确工程收益”时，才把E02续训作为**申请例外**；不能静默沿用。
4. 查现有MMSeg/Mask2Former源码对EMA、paramwise LR、mask matching/semantic inference的具体实现，不在训练中途临时换模型结构或损失标度。

**Stage A交付**：`STAGE_A_AUDIT.md`、`SECOND_GPU_IDENTITY.json`、`M2F_E02_PROVENANCE.md`、`BACKBONE_LICENSE_AUDIT.md`、`STORAGE_BUDGET.json`、`STAGE_A_VERDICT.json`。结论`PASS / PENDING_APPROVAL / BLOCKED`，无批准不能进入Stage B。

---

## 3. 新主线M2F-Swin-L：训练规格（预注册，不堆叠策略）

> **首轮的研究变量**：针对同一M2F+Swin-L已有短训练实验，按统一干净长训练日程测试“更充分的受控优化与EMA能否改善候选表现”。它本身包含多个预先固定的配方选择，**不能事后把提升完全归因于单独的EMA或训练长度**。若要严格单变量归因，须新增匹配对照，不能用本轮与历史E02的差异伪称单变量实验。

| 项目 | 主方案v2.0（需以实际工程smoke确认） | 锁定要求 |
|---|---|---|
| 模型 | `Mask2Former + Swin-L`，8类纯语义分割 | 原生mask分类到每像素8类语义输出流程要可审计 |
| Backbone初始化 | 经核验允许的官方/公开`Swin-L`预训练**骨干** | 不加载未经规则许可的外部数据分割头 |
| Head初始化 | 与训练配置兼容的新初始化Mask2Former decoder/head | strict加载检查，不使用未知E02权重混入 |
| Source train / val | 5597训练；1399只验证 | 图像ID清单SHA和GT映射固定 |
| 随机种子 | `42`（示例，若复现现有E02协议已锁其他seed则记录并冻结） | Python/NumPy/PyTorch、sampler记录 |
| 输入/裁剪 | 主方案`768×768`；不默认升级1024 | 1024仅在独立显存/速度闸门通过且重新注册为不同实验ID时考虑 |
| 采样 | 官方5597图像，正常随机裁剪/翻转/90°旋转及标准轻量色彩增强 | 训练图与标签几何同步；禁止过强合成变换 |
| 暂不加入 | Copy-Paste、ClassMix、MIC、UDA、Target伪标签、KD、不同seed soup | 先测基础训练配方的真实成本和行为 |
| Loss | Mask2Former原生分类CE + mask BCE + Dice及matcher配置 | 以经核验本地实现为准；不将外部论文loss权重未经检查直接套用 |
| Optimizer | AdamW，**候选起点**head`lr=1e-4`、backbone `lr=1e-5`（×0.1），`weight_decay=0.05` | 逐组打印实际lr和参数数；不出现意外冻结/重复分组 |
| LR日程 | 约1.5k**有效优化器更新**warmup + poly或cosine，**启动前固定一种** | 训练上限40k或60k时按固定总步制定全程schedule，不因看val变更 |
| 有效batch | **8**，单卡通过micro batch × gradient_accumulation实现 | 24GB显存优先smoke `micro=1, accum=8` 或 `micro=2, accum=4`，需实测 |
| AMP | BF16优先（如PyTorch/硬件稳定）；否则合适AMP | 非有限loss/grad必须硬停，不能默默忽略 |
| 梯度裁剪 | 以实际MMSeg配置确认，候选`max_norm=0.01`仅为Mask2Former经典配方参考 | 不能把不同`norm_type`/clipping实现混同，固定训练前验证 |
| EMA | 独立shadow权重，候选`decay=0.9999`，**每次有效optimizer.step后**更新 | optimizer被AMP skip不更新；包含可训练参数与必要buffer处理，记录EMA初始化和bias correction/热启动语义 |
| 评估 | 1399GT每类IoU与基础Target风险；不同checkpoint同协议 | **只作为健康闸**，不按val最高选正式checkpoint |
| 主输出 | 预注册固定步数的**EMA单checkpoint**（非val-best） | 保存step/EMA配置/RNG/optimizer/所有输入SHA |
| 最终推理 | 首选一个固定的单模型完整图推理协议，之后可合法D4-TTA8 | TTA档位由实测决定，禁止复制H75的Bstar到M2F概率空间 |

### 3.1 E02优化器配置继承与变更审计

以Stage A核验后的E02训练配置作为默认参考。所有优化器及学习率相关参数，包括head/backbone LR、LR倍率、weight decay、梯度裁剪、warmup、scheduler，必须在`M2F_CONFIG_DIFF_FROM_E02.md`中逐项记录：E02原值、新方案值、变更原因及预期影响。

若E02原配置针对2000步短训练，而新方案面向40k～60k步，允许调整学习率日程，但必须明确说明依据。不能直接将外部论文默认参数视为最优配置。

200步Smoke Test仅用于验证数值稳定性、显存和吞吐，不能证明学习率适合长期优化。正式训练须持续记录学习率曲线、梯度范数、loss及固定检查点指标，不得依据短期Source1399表现随意修改已锁定参数。

若E02配置来源无法核实，明确标记UNKNOWN，不得猜测或假装已完成配置对照。
### 3.2 Epoch换算必须使用5597分母

有效batch8时，约`5597/8 = 699.625`次optimizer更新覆盖一遍训练集（理想连续采样口径）；`40000 step≈57.17 epoch`，`60000≈85.76 epoch`，`90000≈128.64 epoch`。数据迭代、drop_last和随机重采样可能略变，**不能用6996计算“训练epoch”**。当前默认最多60k，**90k不在本次授权预算**。

### 3.3 固定EMA检查点，而非用val-best挑点

- 原则上保存`2k`（smoke）、`10k`、`20k`、`40k`、`60k`（仅若60k被获准且确实运行），每个检查点保留`raw model`与`EMA model`版本标记；对外优先EMA（参数平均后单模型）。可以在2k之后更高频地保存**恢复专用**状态（例如每2k/5k，取决于存储），不当作候选搜索。
- **正式候选目标在长训开始前锁定**：若Gate选40k则`ema_step40000`为primary；选60k则`ema_step60000`为primary，`ema_step40000`仅是预注册备份。若硬截止触发仅能到20k，`ema_step20000`是预注册兜底候选，必须标`EARLY_STOP_BY_DEADLINE`。
- Source1399可在10k/20k/40k等做事先固定的健康评估；明显的NaN、结构性类别塌缩、标签映射错/实验非法→停线；**不能根据1399哪一步mIoU最高事后改定主checkpoint**。
- 不在本轮做EMA+多个checkpoint的soup混合。任何导出的EMA文件都须对应唯一明确权重和推理链；EMA首次进入性能路径前做同样输入两次推理可重复性检查。

---

## 4. Stage B：先做200个有效更新步的速度/稳定性闸门

### 4.1 探针准备

1. 先最小数据/标签smoke：一个batch正反向、loss/mask可视化、训练忽略值正确；训练/验证图片集互斥；模型8类输出、无nan。
2. 探针必须使用**计划长训的同一模型、裁剪、optimizer、真实DataLoader workers、AMP、梯度累积、EMA、checkpoint开关**，独立实验ID。不要用未开EMA、未读磁盘的toy benchmark代替真实速度。
3. 跑**200个有效optimizer更新**（不把microbatch、梯度累积小步和被AMP skip的尝试算为有效更新）。前50步作为warmup/缓存噪声观察，后150步报告单步wall-clock分位数p50/p90、平均有效step时间、GPU峰值显存、IO等待/加载瓶颈、CPU内存与磁盘写入吞吐。
4. 测试一次**原子checkpoint写盘→独立加载→恢复1步**，核对optimizer/scheduler/EMA/有效step计数都正确，文件SHA稳定；证明看门狗能接续，而不是仅能重新开始。
5. 如设置固定步长总训练日程，性能探针的初始训练状态不能与未来正式长训**重复计步或重用错误scheduler**。两种安全策略择一并写清：`DRY_RUN_AND_RESTART_CLEAN`（200步性能测试后从相同公开骨干重新开始，重置所有状态），或`SAME_RUN_RESUME`（探针本就是从0开始的正式长训，所有状态完整，并与锁定总步相同）。**默认推荐前者**，便于先定40k/60k日程，避免事后改scheduler破坏归因。

### 4.2 在截止前选择40k还是60k（硬公式）

- **训练完成硬截止：2026-10-31 12:00北京时间**。若距离截止不足以跑完整日程，宁可保留此前检查点并停机，不能熬到11-01临界点才想到打包。
- `T_wall_budget = 训练截止 - 探针完成后的实际北京时间`；从中扣除**至少4小时**给训练期间固定评估、模型导出、checkpoint IO及不可预见停顿；剩余预算再乘`0.8`做保守安全系数；根据实测**每有效step wall-clock**（建议用p90与均值综合保守估）计算计划总成本。
- 以`(目标步数×实测step耗时)+预计评估/保存成本 ≤ 安全预算`为标准，择**最大可完成的预设档**：首选60k，未通过改40k，仍不通过改20k兜底；如果连20k也来不及，**不启动长训**，保留探针、报告`NO_GO_TIME`。
- 为避免“40k容易赶上，60k几乎来不及”的博弈，**不在跑40k后偷偷延长为60k**。只有在训练启动前已批准明确的`40k`或`60k`计划，才执行对应scheduler；到期前若因异常必须提前停，取最近**预注册**EMA检查点并标记原因。
- 必须把**两套估算**都写进`TRAIN_FEASIBILITY.json`：`40k/p50,p90, total h`、`60k/p50,p90, total h`，连同当前时刻、剩余预算、评估开销和资源占用；不可直接继承Muse的20~36h外推。对于每个任务，无论做不做，报告`GO_60K / GO_40K / GO_20K / NO_GO`。

> **参考纯算术，非实测**：若每有效step`1/1.5/2/3秒`，40k约`11.1/16.7/22.2/33.3小时`，60k约`16.7/25/33.3/50小时`，不含评估、checkpoint、推理与QA。真实单步还受microbatch、EMA、IO及资源影响。

### 4.3 Gate B，全部通过才允许长训

- `B1`：独立GPU授权+UUID校验通过，且显存不会抢占其他用户进程；
- `B2`：训练数据5597身份/GT映射正确，pretrain来源、许可证及分割头初始化合规；
- `B3`：200个**有效**step运行稳定，loss/grad有限、输出有意义、EMA持续更新；
- `B4`：checkpoint包含足以断点续训的model/optimizer/scheduler/EMA/scaler/RNG/sampler及有效step；
- `B5`：40k/60k/20k档位按硬时间与存储预算推导，**不是先选60k再赌能跑完**；
- `B6`：第二线独立看门狗健壮，且对GPU3现有进程/单元无写权限、无影响；
- `B7`：队长已明确授权在本实验ID及固定档位内开展长训（可以在200步阶段提前授予“若通过自动长训”条件授权）。

---

## 5. 独立看门狗：服务器运行、Claude只报告关键里程碑

### 5.1 架构必须独立

建议新目录（可依现有可用磁盘调整；**不可改既有主线目录**）：

```text
/home/jhy/aic2026_secondgpu_m2f_20261028/
  00_audit/                   # GPU/资产/许可证/200步测试
  configs/                    # 冻结的实验配置与SHA
  src/                        # 最小实验代码、差异补丁、测试
  ops/
    watchdog.py               # 独立进程监控与有限状态机
    tasks.yaml                # 固定任务DAG和Gate
    status.sh                 # 只读汇总，不触发计算
    stop.sh                   # 受控停止/不删checkpoint
    state.json                # 原子写入+锁；明确实测步数
    events.jsonl              # 事件、错误及重试证据
    watchdog.lock            # flock防重复实例
    watchdog.log
  run_m2f_clean_<id>/
    checkpoints/              # raw/EMA+optimizer，原子写盘
    train.log
    val_metrics/
    gpu_metrics/
    checkpoint_manifest.json
  rein_ab/                    # 只有独立许可+授权后才可运行
  candidate_zips/             # 单模型3594预测及QA
  reports/                    # 少量必要审计/结果；不生成最终作品PDF
```

服务名**另取唯一名称**（例如`secondgpu-m2f-watchdog.service`），在systemd user下运行并核查user linger；不具备权限则使用独立tmux/nohup并验证SSH断开后存活。**绝对禁止**修改`ziponly-watchdog.service`、其锁、`state.json`、GPU3任务、代码和模型文件。

### 5.2 看门狗固定调度序列（DAG）

```text
A_READ_ONLY_AUDIT
   └─ B_SMOKE
       └─ B_BENCHMARK_200_EFFECTIVE_STEPS
           └─ B_CHECKPOINT_RESTART_TEST
               └─ B_SELECT_FIXED_BUDGET
                   └─ [条件授权+所有Gate PASS] LONG_M2F_CLEAN_TRAIN
                       └─ EXPORT_FIXED_STEP_EMA
                           └─ SOURCE1399_HEALTH_EVAL
                               └─ TARGET3594_INFER_SINGLE_MODEL
                                   └─ ZIP_QA_ALL_3594
                                       └─ READY_FOR_MANUAL_SUBMISSION

(REIN研究独立队列：默认DISABLED / PENDING_AUTHORIZATION)
```

- 若当前计划是40k或60k，只有在**固定目标step对应EMA文件**存在且健康闸通过时才自动运行全量Target推理。若提前截止，只能按前述预注册20k/40k兜底档，不能自动挑Source最高点。
- 独立看门狗每90秒检查**任务PID+启动时间、心跳/有效step时间、GPU UUID及其显存/他人进程、磁盘空闲、logger最后时间、checkpoint原子写与manifest一致性**；看到进程退出`0`且产物符合Gate才推进下一个节点。
- 采用`flock`、持久化任务状态、`run_id`，保证**只有一个同GPU长任务**；不依赖Claude Code会话在线。
- 只允许针对已明确恢复机制的瞬时网络/磁盘IO/断开问题在**相同实验配置**下从最近完整checkpoint最多自动重试一次；OOM/CUDA错误、NaN/Inf、SHA不符、GPU被其他用户占用、磁盘不足、数据污染、关键文件损坏、deadline超限→`HALTED`并停止启动后续GPU任务。不要为了过闸自动改batch/LR/crop/step/EMA。
- 在每次checkpoint前后记录实际耗时、ETA（仅服务器本地状态）、有效step与尝试step，监控EMA未更新、loss长期无有限数、输出严重类别塌缩等异常；不要因为短期val波动自动停线。
- `status.sh`返回单屏：`host / GPU UUID / state / effective_step / locked_target_steps / median_s_per_step / ETA_CST / disk_free / latest_ema / latest_error / ready_zip_path`。**status只读，不能隐式启动任何任务**。
- 启动/后台运行后Claude**只回复一次**：具体第二GPU身份、看门狗服务名/PID、run目录、固定档位与单条状态查看命令。结束/出错时写本地报告；用户无需持续向Claude发起“再检查一下”请求。看门狗能写状态文件，但**不宣称能凭空主动在聊天里通知用户**。

### 5.3 资源隔离细节

- 独立工作目录和独立Python venv（若已有兼容环境可只读复用），不执行系统级pip升级/替换GPU3的共有venv。
- 输入数据挂载为只读；若存在GPU3对Source/Target数据目录的高强度IO任务，本线应限额prefetch、workers数量，减少I/O碰撞。
- 任务子进程从启动至结束固定`CUDA_VISIBLE_DEVICES=<已核准UUID>`；看门狗本身只运行CPU检查，不在CUDA环境里擅自创建其它GPU进程。
- 必须在提交前对该GPU做**PID+GPU UUID一致性验证**，单纯`CUDA_VISIBLE_DEVICES=0`或“第二张卡”这种描述不足以授权。

---

## 6. M2F中途校验、推理与ZIP生产（只为在线评分）

### 6.1 训练健康闸（无需等官方线上分）

- 每个固定节点记录`loss`、grad norm/AMP skip、学习率及EMA更新次数、checkpoint SHA、各类Source1399 IoU/混淆矩阵、`H300/R1099`分层、Vehicle/Barren/Agricultural风险。
- `G_source_disaster`：如预测全空、类别映射错、任一类近乎持续消失、训练loss非有限、Source性能显著异常，停止候选生产并人工核查。**不把Source比H75高当成目标域提分认证**。
- `G_target_no_gt`：全体与test1/2/3的类面积、置信/熵、与H75+Bstar预测分歧、粗边界异常；分歧率本身不设0.5%~12%硬门；目标域无GT不得宣称真实纠错率。
- 预训练实际使用的ImageNet-22K或其它预训练数据必须有书面/公开证据归档；禁止来源不明的外部有标注分割预训练权重。

### 6.2 M2F输出score的必要确认

- M2F语义输出通常涉及**query类别概率与query mask概率组合**，不是与SegFormer B3同意义的原生logits。记录`model_score_semantics`；候选首测采用**模型自身未经B3 bias干预**的单模型预测。
- 不能直接把H75的`Bstar`加到M2F的概率图或mask-classification query输出；要做M2F专属校准，需单独Source模拟与score空间验证，本版不作为必做工作。
- 初次正式候选：**固定EMA单模型 + 单尺度整图（或明确记录的合规滑窗）+ 正确上采样 + argmax+1**。若已实测可负担D4 TTA8，且不影响截止，可生成一个固定的D4版；不能把多尺度TTA当作无成本，也不能因效果传闻不断生成很多TTA ZIP。

### 6.3 ZIP产出G0/G4条件

- 官方Target3594的ID清单与ZIP条目集合严格相同（3594，唯一文件名）。
- 每张PNG`PIL mode L`、uint8、1024×1024，类别集合`1..8`；无空图、损坏或额外数据文件；CRC测试、解压测试通过。
- 输出`M2F_EMA_<fixed_step>_<protocol>_3594.zip`和同目录`SUBMISSION_QA.json`、`candidate_manifest.json`、`SOURCE1399_METRICS.json`、`TARGET3594_DIAGNOSTICS.json`、`SHA256SUMS.txt`、`CANDIDATE_DECISION.md`。
- manifest至少绑定`experiment_id, checkpoint_sha, ema_decay, fixed_step, model_arch, pretrain_sha/license, source_train_ids_sha, target3594_ids_sha, prediction_script_sha, train_config_sha, class_mapping, score_semantics, TTA, resize, zip_path/size/sha256, QA status, online_mIoU=null`。
- 仅当合法、身份一致、数值稳定、完整QA PASS时状态转为`READY_FOR_MANUAL_SUBMISSION`；官方分数需要队长人工提交后才能填写。
- 若截止前只达到20k备份步数但模型Source/Target风险很高，可直接`NO_GO`，**不为凑包胡乱提交**。

---

## 7. 备选研究：DINOv3 SAT vs LVD + Rein-G + Mask2Former（受控A/B）

> **研究假设，尚无该组合在本赛的已知有效结果。** GOOSE展示DINOv3等骨干在其数据集上重要，但优胜主线是**骨干端到端微调**；Rein/Rein++的冻结骨干DG收益来自其他实验。二者组合不能说“已有论文验证”。

### 7.1 只在三个条件全满足时考虑启动

1. M2F长训不再占用本线GPU，且M2F对应预测ZIP生产预算/主线正常（可以另安排队长明确同意后的新资源，但**不抢GPU3**）；
2. DINOv3-SAT和LVD权重许可、赛事用途许可、预训练潜在重叠风险核对通过（如有正式专项澄清，则归档）；
3. **严格A/B的两次训练+评估+目标预测ZIP若有需求**可以在2026-10-31 12:00的长训截止前完成，否则只允许**不影响M2F的代码审计/CPU配置预研**，不在倒计时末段启动无望完结的新任务。

### 7.2 A/B仅变更预训练来源及其相应输入归一化

| 项目 | A SAT | B LVD |
|---|---|---|
| backbone | DINOv3 ViT-L/16 SAT-493M | DINOv3 ViT-L/16 LVD-1689M |
| 模型卡 | `facebook/dinov3-vitl16-pretrain-sat493m` | 官方`facebook/dinov3-vitl16-pretrain-lvd1689m`相关权重：**必须核对准确模型ID和同构性** |
| normalize | 官方SAT模型对应统计 | 官方LVD模型对应统计 |
| 特征接口 | 同4层token特征抽取+同SimpleFPN+同Mask2Former语义头 | 完全相同代码路径、token/shape定义 |
| 训练策略 | 冻结DINOv3骨干、仅训Rein-G及新初始化的FPN/Mask2Former头 | 同样冻结骨干、相同Rein-G模块容量/初始化策略 |
| 其它参数 | 相同Source5597/val1399、裁剪、batch、optimizer/scheduler、固定step、seed、TTA、指标 | 与A相同 |

- **严禁**拿A的normalize直接给B，或用不同训练总步、不同裁剪、不同EMA、不同DINO权重尺寸而宣称“只比预训练”。
- 第一阶段必须完成`backbone frozen checksum`、训练参数白名单（仅Rein/adapter/FPN/head）、200个有效step性能/有限性；`Rein-G`不能无故切换到含SAM辅助的`Rein-A`。
- 若原始Rein代码只支持DINOv2，需要对DINOv3 token接口和多层feature一致性做明确单测；没有现成代码时，不能把“冻结骨干+线性head”冒称Rein-G。
- A/B初步建议**相同固定2k步探针**，如冻结骨干刚起步无法体现优势，只报告`UNDERTRAINED`并不解释为SAT不适用；必要时做更长等预算，但必须在时间闸门下批准。
- 结果：按Source逐类/目标无GT风险/模型吞吐/许可证综合记录，**不做仅依赖Source1399最高mIoU的武断选型**。如仅完成A未完成B，标签`SINGLE_ARM_EXPLORATORY`，不许标为严格A/B。

### 7.3 DINO与主M2F相互独立的启动/停线

- DINO整体**默认`DISABLED_PENDING_LICENSE_AND_AUTHORIZATION`**；不能因M2F完成就自动开始下载/训练。
- 只有队长额外明确授权、两边均能在预算内闭环并有GPU空余，才启用A/B队列。任务依次跑SAT和LVD，严格同资源配置；若不够同时完成，就不启动新长训。
- DINO如果最终形成候选ZIP，同样走§6全量QA，正式候选仍是**单模型**；不参与在线多模型融合，且不做SAM3训练期监督。

---

## 8. 两项低成本补充审计（不得抢主训练）

### 8.1 图块空间连续性（仅诊断）

- **最多30~60分钟墙钟时间，CPU低资源优先**；对官方Source与Target图像做固定规则的感知hash、下采样边缘相似度、局部边缘拼接候选检索，重点是是否存在**非常明显**的近重复/边界邻接。
- 数据只读、不人工标注、不从Source GT传播到Target、不做测试图拼接推理/标签传导系统；不从“视觉相似”直接推定同一场景空间坐标。若无元数据或特征不足，标`INCONCLUSIVE`；不拖延M2F。
- 成果`SPATIAL_ADJACENCY_SCREENING.md`，明确证据强度、候选图对数量、假阳性风险，无后续自动训练分支。

### 8.2 目标先验重加权Source切片（探索性）

- **仅当v1.3 EM/BBSE已有可靠合法的先验估计时**，可以采用已存在的Source GT逐类混淆矩阵做无标签目标先验敏感性分析；默认限时30分钟。
- 需要比较源域原prior、EM/BBSE两个备选prior以及合理扰动区间；若得到相反候选排名，立即写`UNRELIABLE_FOR_SELECTION`，**不得把它作为选模/自动提交硬门**。
- 禁止为此启动新embedding聚类平台、重新全量跑H75、使用任意Target GT或训练一个“目标线上mIoU预测器”。

---

## 9. 总时间线与优先级（无最终作品阶段）

| 北京时间 | 第二GPU线目标 | 硬边界 |
|---|---|---|
| **10/28** | Stage A审计→M2F配置锁→200有效step真实测速→决定主训练目标40k或60k或20k（可能NO-GO） | 只在独立GPU上执行；v1.3不动 |
| **10/28–10/30** | 符合Gate的M2F干净长训自动运行；固定EMA checkpoint保存；基础验证/日志 | 不能根据val-best临时变更计划 |
| **10/31 12:00** | **新增长训硬停止点**，按已冻结计划优先目标EMA或预设兜底 | 未完成不延长训练，不为了多跑而撞ZIP截止 |
| **10/31下午–11/01** | 从合规固定EMA checkpoint完成Target3594推理、ZIP与全量QA；若已完成且有明确额外资源/许可，DINO仅能做能及时完结的短探针 | 此阶段不做Docker、PDF、网盘或正式平台上传 |
| **11/01 20:00** | **官方预测结果截止**，候选ZIP必须在此前让队长有充分人工操作时间 | 不保证自动得到平台得分 |

**优先级**：`现有GPU3保持安全` > `Stage A合规与身份` > `M2F 200step+长训` > `固定EMA预测ZIP与QA` > `DINO/Rein A/B（条件触发）` > `两项低成本审计（可完全跳过）`。绝不以DINO研发耽误M2F候选ZIP。

---

## 10. 任务状态机、自动化验收和失败策略

### 10.1 通用状态

```text
PLANNED → AUDITED → GPU_APPROVED → BENCHMARK_PASS → TRAINING
  → EMA_CHECKPOINT_READY → SOURCE_HEALTH_PASS → TARGET_INFERENCE
  → ZIP_QA_PASS → READY_FOR_MANUAL_SUBMISSION

其它终态：BLOCKED, NO_GO, HALTED, OUT_OF_TIME, ARCHIVED
DINO研究队列：DISABLED_PENDING_LICENSE_AND_AUTHORIZATION
```

### 10.2 关键自动化测试

1. Source5597/1399互斥，数据清单sha与标签ignore255正确；官方8类映射无歧义。
2. EMA权重真更新、正确step、非EMA checkpoint不会被误当EMA、EMA与AMP skip交互正确。
3. 最小模型训练前向/后向，匈牙利匹配不崩溃、loss有限/梯度合理、M2F semantic inference输出8类1024图。
4. GPU UUID与获授权设备精确一致；GPU3的CUDA进程、服务、checkpoint不会被第二线访问写入。
5. 看门狗锁和超时：制造一个可安全的虚拟失败和进程正常退出测试，证明不会无限重试、不会误判断点文件完整、不会重复启动。
6. 训练恢复：从独立checkpoint完整恢复model/optimizer/scheduler/EMA/scaler/RNG/sampler/step一致性，随机性无法完全恢复的部分如实标记。
7. Source基础混淆矩阵与逐类IoU、H300/R1099池化、ignore处理正确；Source无GT/Target数据通道隔离。
8. 固定EMA checkpoint的推理前处理与score转换明确；不把M2F概率图当成B3 logits添加Bstar。
9. 3594 PNG逐图L/uint8/1024²/1..8/同名；ZIP根目录/CRC/数量/不重复/无多余；ZIP SHA必须实测而非估算。
10. `READY_FOR_MANUAL_SUBMISSION`只表示包在本地可用；从未人工上传时`online_miou=null`。

### 10.3 失败时的处置

| 异常 | 看门狗与Claude应做的事 |
|---|---|
| 无法取得获准第二GPU | `BLOCKED_NO_APPROVED_GPU`，不动GPU3 |
| E02不匹配、Swin-L公开预训练许可不清 | 报来源与证据，停在Stage A，不猜权重 |
| 200步不稳定/模型输出错 | `NO_GO_TECHNICAL`，不启动长训 |
| 40k连时间预算也不满足 | 按预注册规则选20k或`NO_GO_TIME`，不私自启动超时计划 |
| 长训NaN/Inf、OOM、他人GPU冲突 | `HALTED`，保存最后有效checkpoint、日志、故障时间；不自行改配置/占用别卡 |
| EMA检查点有效但Source灾变/Target预测严重塌缩 | 报`NO_GO_CANDIDATE`，不强行生成可提交状态 |
| ZIP检查任一条FAIL | `ZIP_QA_FAIL`，不标READY；定位根因后在权限范围内修复，不覆盖原文件 |
| 时间到10/31中午 | 安全退出长训，记录兜底EMA及风险；不自动推迟硬截止 |
| DINO许可证未确认 | `DISABLED`，不训练、不给主线上“允许运行”的错误结论 |

---

## 11. 本线输出清单与Claude最少汇报格式

### 11.1 阶段文件（Claude写盘，用户不用频繁聊天查看）

```text
STAGE_A_AUDIT.md
SECOND_GPU_IDENTITY.json
M2F_E02_PROVENANCE.md
BACKBONE_LICENSE_AUDIT.md
M2F_CONFIG_DIFF_FROM_E02.md
M2F_BENCHMARK_200.json
TRAIN_FEASIBILITY.json
WATCHDOG_STATUS.json / events.jsonl / ops/status.sh
M2F_TRAIN_RUN_MANIFEST.json
EMA_CHECKPOINT_MANIFEST.json
SOURCE1399_METRICS.json
TARGET3594_DIAGNOSTICS.json
SUBMISSION_QA.json
candidate_manifest.json
candidate ZIP + SHA256SUMS.txt     # 仅通过全部Gate后出现
FINAL_SECONDGPU_RESULT.md          # SUCCESS / NO_GO / BLOCKED / TIMEOUT
```

### 11.2 Claude只需两次必要汇报

**一次：** 部署看门狗/200步测速与训练预算决策完成时，报告`GPU UUID、benchmark p50/p90、档位40k/60k/20k/NO_GO、预计训练结束CST、看门狗服务/PID、工作路径、一条status命令`。若还没获得正式长训授权，等待队长单次授权；已满足条件授权则自动继续。

**一次：** 最终结束时（或者出现必须人工处理的HALTED），报告`已完成有效step、EMA checkpoint SHA、模型/训练协议、Source健康、Target风险、ZIP绝对路径/大小/SHA、3594 QA、status、未处理问题`。**不每隔几十分钟主动发送聊天消息**；服务器状态文件才是持续监控的信息来源。

**本文件到这里结束。不要额外生成“现在给Claude的提示词”或自行宣布已授权开跑。**

---

## 附录：主要研究依据（用于新Claude核对；不构成比赛许可）

- Muse《其他候选方案检索报告》（2026-10-28）——多模型证据与GPU计划均属于队伍讨论和跨数据集参考，非本队已复现成绩。
- GOOSE挑战赛技术报告：<https://arxiv.org/abs/2606.23113>。支持“预训练+长日程+EMA+大裁剪+TTA值得研究”，**没有证明DINOv3+Rein完整组合**。
- Rein：<https://arxiv.org/abs/2312.04265>；Rein++：<https://arxiv.org/abs/2508.01667>。Rein++中的Rein-A含SAM信息，本计划仅研究不依赖SAM的Rein-G。
- 官方DINOv3 SAT模型卡：<https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m>；许可证字段`dinov3-license`，需另验赛题许可。
- Mask2Former正式论文/实现参考：<https://arxiv.org/abs/2112.01527>，<https://github.com/facebookresearch/Mask2Former>。优先复用本地经审计MMSeg工程，不强行迁移已归档旧仓库。
- 大赛半决赛官方结果截止：**2026-11-01 20:00北京时间**。本文件只做**预测ZIP产出**，不做另一期作品材料交付。