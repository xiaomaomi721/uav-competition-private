# AIC2026 UAV半决赛｜第二GPU独立实验最终执行指令

你现在负责一条独立于现有主线的新研究与训练任务。

**主目标：利用teacher-4090服务器上另一张经核实且获准使用的RTX 4090，开展Mask2Former + Swin-L强化训练，并在2026年11月1日20:00（北京时间）之前，争取产出3594张预测PNG的合格候选ZIP。**

请先完整阅读：

1. `AIC2026_UAV_第二GPU独立实验执行方案_v2.0_20261028.md`
2. "E:\A人工智能大赛\A半决赛\队友交付资料\T3_FULL_ST_DINOv3_V1_V2_完整实验总结与队友交接_20261028(1).md"
3. 其他已提供的赛事规则与资产交接材料。

本消息是v2.0的最终执行补充。如发生冲突，以本消息的新增限制和修正为准。附件未提供或无法读取时，应如实记录，不要假装已经审阅。

## 一、任务边界与GPU隔离

现有主线已经在`teacher-4090`物理GPU3运行：

- GPU3 UUID：`GPU-32853ccf-4c4f-1328-8d75-04efe9d5298f`
- 主线目录：`/home/jhy/aic2026_ziponly_20261028/`
- 现有服务：`ziponly-watchdog.service`
- 任务包括H75缓存、EM/BBSE和M2F E02候选ZIP。

**这条主线绝对不能修改、停止、重启、接管或抢占。**

你的工作必须满足：

- 优先使用同一台teacher-4090服务器；
- 在GPU0/1/2中实测并选择一张经授权、资源可用的RTX 4090；
- 核对GPU物理编号、UUID、PCI Bus ID、进程归属、显存和使用权限；
- 使用CUDA_VISIBLE_DEVICES绑定具体GPU，并由程序启动时二次核验；
- 只允许使用一张新增GPU；
- 不得占用他人任务，不得杀死其他进程；
- 独立工作目录、独立服务、独立锁文件、独立状态和日志；
- 所有历史资产只读使用，禁止覆盖。

如果没有可确认权限的第二张GPU，记录`BLOCKED_NO_APPROVED_GPU`，不得改用GPU3。

## 二、Stage A：先审计，复用已有成果

### 1. M2F E02资产审计

审计现有Mask2Former + Swin-L E02 B1/C0权重、配置和训练记录，核实此前Source1399约79.5654%/79.5325%的成绩与具体checkpoint是否对应。

重点检查：

- 权重SHA、来源及许可证；
- 模型结构和八类标签映射；
- 原始Swin-L预训练权重；
- optimizer、scheduler、warmup、梯度裁剪；
- E02实际训练步数和学习率日程；
- EMA是否存在及实现方式；
- Python、PyTorch、MMCV、MMEngine、MMSeg和MMDet版本；
- checkpoint能否完整恢复训练。

**默认从已审计且合规的Swin-L公开骨干权重干净启动新训练。**

只有E02数据血缘、优化器状态、学习率日程、EMA/RNG等完整可信，且续训有明确理由时，才可把E02续训作为另行申请的例外，不得擅自继承。

### 2. 优先复用已有环境

队友交接显示teacher-4090已有多套M2F环境，包括：

`/home/jhy/.venvs/mask2former`

以及：

`/home/jhy/aic2026_h76_20261026/experiments/T3_ST_DINOV3_SELECTKD_V1/environment/m2f-venv`

请先只读盘点这些环境及其他既有环境。

能够安全复用的就不要重复安装，不得无故新建第四套M2F环境。若现有环境不兼容，只能在本线独立环境中解决，不能修改GPU3共享的运行环境。

### 3. 队友DINOv3历史实验

历史结果已经明确：

- DINOv3-B教师完成2500步Head训练+2500步部分解冻训练；
- Source1399 mIoU为60.0826%；
- V1：`STOP_NO_KD_COVERAGE`；
- V2：Source虚拟纠错+0.1059pp，但Target有效覆盖仅0.015709%，最终`STOP_TARGET_LOW_COVERAGE`；
- 未发生学生CONTROL/TREAT续训；
- 没有产生新学生checkpoint或线上成绩。

**关闭这条DINOv3-B选择性KD路线。**

不得重复V1/V2，不得通过降低既有A1门槛重新启动同一实验。

这些负面结果不等于DINOv3-L SAT/LVD + Rein-G被证伪，但足以要求我们降低其优先级。

## 三、M2F强化训练配置

采用v2.0中的受控配方，优先参数为：

- 架构：Mask2Former + Swin-L；
- Source训练：5597张官方GT；
- Source验证：1399张，仅作评估；
- Crop：768×768；
- 有效batch：8；
- Optimizer：AdamW；
- 骨干与Head使用分组学习率；
- EMA：候选decay 0.9999；
- AMP：优先BF16，需验证数值稳定性；
- Mask2Former原生分类与mask损失；
- 预注册固定步数EMA checkpoint作为正式候选；
- 首次训练不加入Copy-Paste、ClassMix、MIC、UDA或DINO蒸馏。

### 重要新增规则：E02优化器锚点

**不要直接把论文中的head LR=1e-4、backbone倍率0.1、梯度裁剪0.01等候选起点当成已经验证的最佳参数。**

必须在`M2F_CONFIG_DIFF_FROM_E02.md`中逐项写明：

- 参数名称；
- E02已核验原值；
- 新方案数值；
- 修改理由；
- 预期作用与主要风险。

涵盖学习率、paramwise倍率、weight decay、warmup、scheduler、梯度裁剪、batch、AMP和EMA等。

E02是本项目已有的实证参考，但2000步短训配置也未必适合40k～60k长训，因此允许有充分依据的调整。

200步测试只能证明稳定性及吞吐，不能证明长期学习率最优。

## 四、先做200个有效优化步的真实测速

在锁定训练配置后：

1. 完成数据、标签、前向、反向和loss最小测试；
2. 用真实模型、真实DataLoader、真实梯度累积、AMP和EMA运行200个有效optimizer更新；
3. 区分有效optimizer step与microbatch；
4. 前50步作为预热，重点统计后150步的平均值、p50和p90；
5. 记录显存峰值、GPU利用率、CPU/IO瓶颈和磁盘需求；
6. 完成checkpoint原子保存、加载与恢复测试；
7. 验证optimizer、scheduler、EMA、RNG及有效step计数恢复一致。

默认使用`DRY_RUN_AND_RESTART_CLEAN`，避免200步探针与正式训练不同的scheduler总步数相互污染。

生成`M2F_BENCHMARK_200.json`和`TRAIN_FEASIBILITY.json`。

## 五、训练预算与自动放行

新增M2F长训的硬截止为：

**2026-10-31 12:00，北京时间。**

采用实测的有效step耗时进行预算估计，另外保留至少4小时评估与保存余量，再施加20%安全余量。

预注册档位：

- 60k：只有预计可以安全完成时采用；
- 40k：60k不能安全完成时采用；
- 20k：40k也不可行时的兜底；
- NO_GO：20k仍不可行、资源不合规或技术测试失败。

训练目标必须在正式启动前锁定。

不得在训练中依据Source1399最高mIoU临时变更学习率或目标步数。

固定EMA检查点按v2.0保存，正式候选不采用事后选择的val-best。Source只做灾变、类别和数值健康检查。

### 本次条件授权

**我在此一次性授权Stage A审计、200步测速，以及全部闸门PASS后，在所选固定预算内自动启动M2F干净长训、评估和候选ZIP生成。**

不需要每个阶段向我重新索取执行批准。

但是，以下情况必须暂停受影响任务，保留证据并等待人工处理：

- 无法确认第二GPU的使用权限；
- 预训练来源或赛事许可不清；
- E02/模型资产存在无法解决的身份冲突；
- 200步测试发生严重异常；
- 时间预算不满足任何档位；
- OOM、NaN、数据污染或不可安全恢复的训练错误。

本次条件授权不包括DINOv3正式训练、SAM3目标域监督、使用更多GPU或任何官方平台提交。

## 六、独立部署看门狗，降低Claude会话消耗

请在本线独立部署服务器端看门狗。

建议目录：

`/home/jhy/aic2026_secondgpu_m2f_20261028/`

建议服务名称：

`secondgpu-m2f-watchdog.service`

要求：

- 优先systemd user service，并验证SSH断开后仍运行；
- 无法使用时，采用可靠的独立tmux等方式；
- 每90秒自动检查一次；
- 监控PID、退出码、heartbeat、有效step、GPU UUID、显存、磁盘、checkpoint和日志；
- 使用flock防重复启动；
- 每次只允许运行一个本线GPU任务；
- 按前置闸门自动推进任务；
- 对可安全恢复的瞬时异常最多自动重试一次；
- OOM、NaN、SHA不符、权限冲突、磁盘不足时HALT；
- 禁止为了通过闸门自动改变超参数；
- 记录完整事件流及可恢复状态。

必须提供：

`bash ops/status.sh`

它要能一次显示当前阶段、GPU UUID、有效步数、训练速度、预计完成时间、最新checkpoint和最后异常。

**不要让Claude Code本人每隔几分钟进行SSH轮询。**

由服务器看门狗完成监控、恢复和任务交接。Claude完成部署及初次验证后报告一次即可，之后我需要检查时再主动询问。

## 七、正式训练完成后自动生成候选ZIP

根据预注册固定EMA checkpoint：

1. 验证模型、EMA、类别映射及Source健康；
2. 对官方Target3594执行单模型推理；
3. 保持Mask2Former本身的semantic score语义；
4. 不得把H75的Bstar直接加到M2F概率输出；
5. 如D4-TTA确有时间预算且已核实数值正确，可使用合法单模型TTA；
6. 生成3594张PNG并打包。

完整QA必须确认：

- 3594张且文件名集合准确；
- ZIP根目录平铺；
- PNG灰度L、uint8、1024×1024；
- 像素类别1..8；
- CRC与文件完整性；
- 模型、训练配置、推理脚本、ZIP的SHA256；
- 每份ZIP可以追溯到唯一EMA checkpoint。

只有全部通过才标记`READY_FOR_MANUAL_SUBMISSION`。

不得声称Target真实mIoU已经提升，不得自动上传官方平台。

## 八、DINOv3-SAT/LVD + Rein-G的后续限制

该研究方向保留，但默认`DISABLED_PENDING_LICENSE_AND_AUTHORIZATION`。

仅在M2F优先工作完成、有足够时间、许可证核验通过且我再次授权后，才可启动严格A/B：

- SAT-493M对LVD-1689M；
- 同为DINOv3 ViT-L/16；
- 相同冻结骨干+Rein-G+Mask2Former结构；
- 同一数据、batch、步数、seed和训练策略；
- 各用对应预训练权重的正确归一化统计；
- 重点审计Barren、Agricultural、Vehicle；
- 不使用Rein-A中的SAM辅助路线。

不允许把GOOSE的端到端DINOv3结果宣称为DINOv3+Rein已有验证结果。

此前HF在teacher-4090上存在网络不可达记录。若未来获得额外授权需要下载SAT/LVD权重，必须预先规划官方来源下载及Colab→Windows→SCP等合规中转方案，完整核验revision与SHA。不得使用来源不明的镜像权重。

本次不要为DINO额外下载大型权重或启动训练。

## 九、低成本附属审计

仅在不影响M2F训练和IO的前提下进行：

- 图块空间连续性检查：最多30～60分钟，只读诊断，不做标签传播；
- Target先验重加权Source诊断：仅复用已可信的EM/BBSE估计，默认最多30分钟，不作为自动选模门槛。

两项均可跳过，不能为此推迟M2F或额外建立大型研究基础设施。

## 十、最终汇报要求

你只需要在以下节点向我提供简洁报告：

**第一次：** Stage A、200步测速和看门狗部署完成。

说明GPU身份、资产结果、配置差异、实际step耗时、选定40k/60k/20k档位、训练结束时间预算、服务名及唯一状态查询命令。

如满足条件授权及全部闸门，直接让看门狗自动执行正式M2F训练，无需再次询问。

**第二次：** 我主动检查时，或最终任务结束时。

报告真实有效step、最终EMA权重及SHA、Source评估、Target无标签风险、ZIP路径/SHA/QA以及最终READY、NO_GO或HALTED状态。

所有详细过程保存到服务器文件，不需要在聊天中大量输出日志。

## 最终执行原则

- 不重复Stage A已经通过的项目；
- 不重复队友已关闭的DINOv3选择性KD；
- 不抢占GPU3；
- 不重新安装能够安全复用的M2F环境；
- 不把Source高分当作Target成绩；
- 不为了完成60k而突破10月31日中午截止；
- 不为了交付ZIP而绕过质量闸门；
- 不做Docker、PDF、网盘和最终作品材料；
- 不自动向赛事官网提交。

**现在开始执行第二GPU独立实验。优先完成Stage A和200步测试，并部署能够脱离Claude会话持续工作的看门狗。通过全部条件后，按照预注册队列自动完成M2F长训与候选ZIP生产。**