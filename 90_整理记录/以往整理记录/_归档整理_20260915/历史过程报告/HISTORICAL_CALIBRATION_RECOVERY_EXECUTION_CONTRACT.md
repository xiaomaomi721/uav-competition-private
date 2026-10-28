# HISTORICAL_POSITIVE_CALIBRATION_RECOVERY_V2 — 审计与执行约定

状态：只读身份/缓存审计进行中；不是完成后的Inventory，也不代表已启动CPU扫描。

## 隔离

- 不训练、不修改checkpoint、不修改GPU2/GPU3的formal或队列、不发进程信号。
- 本轮不启动任何GPU forward；有价值但缺可信缓存的模型只登记QUEUE_TTA。
- 不自动线上提交，不删文件，不批量保存冗余PNG。
- CPU工作使用CUDA_VISIBLE_DEVICES空、nice19、ionice idle，串行扫描，线程数最多2（根据资源审计可降至1）。

## 分类与证据

- RUN_NOW：模型身份、必要的正效益证据、完整验证集logits及其协议/精度/来源均闭环；CPU资源与增量磁盘门禁可通过。
- QUEUE_TTA：符合本轮核心优先级，但缺少可信完整缓存；不自动补GPU。
- RULE_BLOCKED：Pseudo/Target Adaptation缺官方规则闭环；禁止候选包生产。
- SKIP：明确排除路线/线上负效益/非独立core；若因证据不足暂缓必须写明UNKNOWN，不能把UNKNOWN写成负效益。
- 只有PNG/argmax预测不能重建任意8D偏置的logits landscape。
- 不把已知精度门禁失败缓存或1182/1399部分缓存冒充完整匹配验证缓存。
- 精确69.281290的checkpoint+bias身份仅凭完整服务器证据闭环认可；否则UNKNOWN。

## 全模型共享的后续扫描规则

- 历史固定中心：zero、LOCKED8D、EXP41_STYLE。扩展审计已找到EXP41_STYLE数值与历史build_p100/69.281290账本的服务器产物对应关系，详见HISTORICAL_GOLDEN_69281290_IDENTITY_RECOVERY.md；尚无独立平台评分回执，不把该关联当作新FP32匹配验证结果。
- 固定seed42；对规范化排序的完整image stem按SHA256(seed,stem)排序，轮流分配5fold。所有模型共用同一份fold manifest。
- 每fold由整张image组成，绝不pixel随机split。
- 保存每image 8×8 confusion（int64），可复算总体/fold mIoU、8类IoU、预测比例及相对zero/两个历史中心的delta。
- 所有候选共享fold，仅用于稳定性筛查；因搜索也使用这批val，它不是独立留出集，不声称是无偏泛化估计或独立交叉验证成绩。
- 先三中心，再8轴正负偏移，再敏感轴局部联合搜索，再邻域稳定性/替代盆地；不做5^8穷举。
- 数值去重和整套val预测内容去重；等价角色保留别名，不重复计算/出ZIP。
- 不预设Bg/Barren、Bg/Agri一定最敏感，按实际轴敏感性决定变体。
- 首先保存小体积CSV/JSON/manifest/每image confusion，不默认保存每候选PNG。
- test缓存完整可信才进行CPU出包；新test forward仅排队，需以后具备明确GPU许可。
- 真正出包前逐次检查空间、官方格式、500张预测内容去重、权重/缓存/向量/协议SHA绑定。

## 当前起始保护快照

服务器时间2026-09-14 16:49:37：RareExposure GPU2 PID283448 step48872；Paramwise01 GPU3 PID2115724 step2200。两formal配置SHA未变，可用空间10,893,778,944字节（约10.15GiB）。

证据：HISTORICAL_RECOVERY_TRAINING_GUARD_01.json。
