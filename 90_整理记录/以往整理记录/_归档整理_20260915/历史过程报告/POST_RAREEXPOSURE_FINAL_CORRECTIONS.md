# POST_RAREEXPOSURE 最终解释性修正 — 执行约束

本文件落实用户最新修正，优先于上一版自动own8D扫描描述。

## 不可变训练与现有调度

- GPU2 RareExposure formal283448、GPU3 Paramwise formal2115724不停止、不重启、不resume、不改配置、不换GPU。
- 当前storage-aware streaming supervisors2399435/2399454保持有效，不暂停/替换，不热改其冻结文件。
- 不恢复281314/2113375/2187127等旧scheduler。
- 不开启第三条56k训练、不自动线上提交、不保存全分辨率pixel logits长期缓存。

## Rare checkpoint真实范围

只有best_miou.pt与terminal last.pt。训练自然结束后严格核验entry exit/56000/updates+skips/metrics/manifest/权重SHA。last.step=56000后认证为last/56000；按模型tensor内容SHA与best去重。52k记NOT_SAVED，不补造、不改trainer。

Paramwise继续真实best/52k/56k。

## 三包规范名称

1. TTA8_ONLY：zero bias。
2. TTA8_CHAMPION8D_69_281290：[-0.2,0.1,0.0709,0.7,3.6,0.4,3.3215,-0.5]。
3. TTA8_LOCKED8D_69_276229：[-0.2,-0.0384,0.0709,0.7,3.6,0.1932,3.3215,-0.5]。

冠军分数边界：69.281290来源于历史项目账本，未取得独立平台评分回执。

不能热改活跃冻结driver来改旧文件标签。新增规范索引按精确vector绑定已有ZIP路径和SHA，为用户提供canonical_name，不复制/重命名历史或正在写的文件。旧EXP41_STYLE只能作为溯源别名；新factory报告和新包使用冠军正式名。Exp41_EDGE_BCE_CORE独立训练身份不与bias混淆。

## 自动截止状态

formal audit -> matched three-package generation -> **MATCHED_PACKAGES_COMPLETE_WAITING_ONLINE_RESULTS**。

真实online matched结果才可计算最终DeltaCore/DeltaChampionCal/DeltaLockedCal；没有结果就写UNKNOWN，不根据val/local标winner或启动Rare PassA/PassB。

需要真实同协议Stage23线上baseline；历史AMP成绩不冒充新FP32同协议成绩。历史Goldenfactory三固定角色也保留，用于以后用户手工提交建立匹配对照。

## 等待期间历史回收

新增独立observer只读等待Rare已通过官方检查的三包/完整结果。GPU2确实空闲且满足容量门禁后：

1. Golden step56000。
2. Exp47 standalone step56000（CWFA=false）。
3. Stage66 step52000。
4. Exp45 MixUp alpha0.1/p0.5 step51800。

每core完整streaming PassA -> PassB -> finaltest多包。共享磁盘/postprocess锁避免和Paramwise同时写包，不影响其训练。

若main根据用户真实online结果生成资格通知，observer仅在当前完整core任务结束后的安全边界停止并标WAITING_RARE_HANDOFF，由main核验并部署Rareownscan。任何通知文件的出现可以触发保守停队，但不构成资格证明，更不能直接启动新训练或未批准scan。不中断半途TTA forward。

Stage66的51800/56000不自动跑，先报告52000实际潜力/敏感性证据再决定；第二批MixStyle/MaxEnt/DINO/Exp41保持QUEUE_TTA_SECONDARY。Pseudo保持RULE_BLOCKED。

## 实现和验证必须分开报告

- 当前已部署：固定三包streaming及旧标签输出，终态等待online；未部署新历史factory。
- 本地新增实现进行中：canonical registry、observer、完整factory、planner/统计与fold选择。
- 本地GT1399 histogram已逐文件SHA/像素计数，可生成真实320图subset。
- 生产GPU路径只能在Rare释放后运行；CPUfixture不冒称GPU完成。
- 当前Auto模式拒绝新增remote SCP，即使已明确用户授权；不绕过。部署待用户切换可逐次审核模式。
