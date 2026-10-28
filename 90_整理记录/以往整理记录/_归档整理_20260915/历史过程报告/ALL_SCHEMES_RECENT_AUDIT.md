# 近期方案状态与成绩补充审计

**编制日期：2026-09-14。只读取本地已有证据，不访问服务器，不上传外部AI。** 本表是近期方案补充，非全部历史实验总表。分数、状态、ZIP存在性分开记录；没有线上成绩不填0，也不推断算法失败。

## 零、最早分组分析路线（仅诊断，不计新训练/新线上成绩）

| 路线 | 归档结论 | 成绩含义 |
|---|---|---|
| Phase1 HDBSCAN / Louvain group discovery | 6996图；Louvain 17组、噪声0；HDBSCAN 2组、噪声2289图/32.7187%；14行人工检查完成，回执CANDIDATE_PASS | 分组诊断与候选验收，不是分割模型训练，不产生新比赛预测ZIP或独立线上分 |
| Phase2 分组重加权验证 | 复用8个历史模型及旧8个线上成绩；原1399固定val Spearman=0.785714，Louvain组均值=0.714286，差−0.071429，**没有改善** | 只是原固定val上的group-aware reweighting，**不是leave-group-out**；分析完成不等于新训练或新线上提交 |

来源：`历史归档/_return_package/phase1_final_candidate.json`、`历史归档/_phase2_return/PHASE2_RETURN_TO_CHATGPT.txt`。源回执无内嵌时间，仅按历史归档阶段描述。

## 一、训练方案与成绩

| 方案 | 最后证据时间/来源 | 训练与包状态 | 本地验证成绩 | 线上成绩 |
|---|---|---|---|---|
| Stage59 RegionC，类别对[[4,0],[6,0]] | 历史包目录20260911；清单本地mtime 2026-09-14 08:24:36 UTC | **本地10个原包实际存在，本次Glob确认**；协调方说明历史SHA核验完成，本轮未重hash。last/s56000路径记录不替代独立完成回执；本轮恢复排除不等于算法失败 | 本次未找到可引用数值 | 所查材料未找到 |
| Stage66 SEMEDA+RareCrop+延迟CWFA | 历史包目录20260911；S1审计本地mtime 2026-09-14 08:59:07 UTC | 审计已读metrics/completed；best51800、52000、56000身份独立。**本地10个原包实际存在，本次Glob确认**；协调方说明历史SHA核验完成，本轮未重hash。不能混用不同包族或checkpoint | 见下表 | 所查材料未找到 |
| Full6996 refit | 优先包回执 2026-09-12 01:47:59 UTC | 69997和65000各3包，共6包PASS；last是69997别名。**本次6个本地ZIP SHA全部匹配**。`all_done=false`，不代表55000/60000或整条管线完成 | 未找到可直接比较的1399独立holdout数值；不能与原5597/1399协议混比 | 所查材料未找到 |
| RareExposure v2 | **2026-09-14 18:56:07.258181**，源时间无时区 | **55111/56000**，`FORMAL_RUNNING_STREAMING_INSTALLED`；最新清单仅best_miou.pt，无更晚完成回执/最终metrics/新matched包证据 | 尚无最终评估成绩；不把训练loss当mIoU | 未有可引用线上成绩 |
| Paramwise01 UUID-fixed | **2026-09-14 18:56:07.258181**，源时间无时区 | **8630/56000**，`FORMAL_RUNNING_STREAMING_INSTALLED`；最新清单仅best_miou.pt，无更晚完成回执/最终metrics/新matched包证据 | 尚无最终评估成绩 | 未有可引用线上成绩 |

最新训练证据来自协调方保存的 `ALL_SCHEMES_LIVE_STATUS_20260914.json`，不是本审查重新远程观测。18:39快照只保留作历史来源，不覆盖18:56记录。**不能根据预计结束时间声称现在已完成。**

### Stage66 已有本地验证成绩（非线上、非本轮新扫描）

| checkpoint / 协议 | mIoU原值（0–1） | 百分数显示 |
|---|---:|---:|
| best51800：历史训练评估 | 0.8125200681968625 | 81.252007% |
| best51800：独立per_checkpoint CSV | 0.8125296965690411 | 81.252970% |
| 52000：per_checkpoint CSV | 0.8125501791930202 | 81.255018% |
| 56000：per_checkpoint CSV | 0.8124672938336803 | 81.246729% |
| 52000：历史1399图true-FP32 D4，zero bias | 0.8131277295176083 | 81.312773% |
| 52000：同FP32 reference，champion固定向量 | 0.7876702342447008 | 78.767023% |
| 52000：同FP32 reference，locked固定向量 | 0.7875682049879659 | 78.756820% |

两份best51800记录存在微小差别，保留其独立来源，不抹平。D4固定向量结果不能代替无TTA模型本体成绩。

**关键：Stage66 FP16 precision FAIL 是缓存量化精度门禁失败。** 1399图诊断完成、旧程序exit2，只禁止该缓存下游复用；不等于训练失败、模型本体无效或比赛算法失败。历史FP32参考分数仍可报告，但不能从这些汇总值恢复任意新bias预测。

## 二、工程验收、受阻与仅审计项目（不是独立训练方法成绩）

| 项目 | 状态与边界 |
|---|---|
| Rare smoke1 | 产物契约不匹配：要求至少8张可视化但配置仅2；工程验收失败，不是算法失败。 |
| Rare smoke2 | 原准备脚本只调整max_val_samples和prediction_visualization_count从2到8；后续formal运行快照说明已越过smoke入口。四步smoke不构成正式比赛成绩或500图包。 |
| Param旧protocol_v2全量缓存 | 容量CPU预检BLOCKED：需215,605,739,520字节，当时可用11,097,174,016；没有新扫描/包。后被流式调度替代，不能写成训练失败。 |
| 历史可信缓存恢复V2 | `BLOCKED_NO_RUN_NOW`，scans_started=0、new_gpu_forwards=0、new_packages=0。仅审计，不是校准已完成。 |
| Storage-aware两队列 | 已有交接证据；最后18:56仍在等待各自formal自然完成。工程接入不等于未来GPU出包必成功；线上未知时不自动模型专属8D。 |
| POST_RARE新factory/observer | 仅本地工程实现；CPU合成fixture为17候选、10角色，明确FIXTURE_ONLY_PASS。不是生产1399/500图结果，未有新生产ZIP；无生产GPU验证，不声称已部署运行。 |
| POST_RARE queue静态问题 | canonical语义角色丢失与+0.10浮点边界两项**未修复**；按要求暂停修改/部署，详见`POST_RARE_QUEUE_REVIEW_FINDINGS.md`。 |
| 第二批/规则受限 | MixStyle、MaxEnt、DINO、Exp41仍为第二批待排；Stage66 51800/56000不自动扩展；Pseudo仍RULE_BLOCKED。不是已取消训练或已出新包。 |

## 三、本次可确认的本地ZIP

目录：`提交包/Full6996_priority_20260912/`

| 本地文件 | SHA256 |
|---|---|
| 65000_TTA8.zip | de78b5d8665649182575b434b6e21076d0a8648a73e7235b06672f52b2278df9 |
| 65000_TTA8_FIXED8D_EXP41_STYLE.zip | 0bcfe2a11b3bf5f519e6f3fd5490b3e86a0c283db3208d4b0552000a0f8358c4 |
| 65000_TTA8_LOCKED8D_69_276229.zip | c32a410e39716a7ce0e46f075ee3cb1d3e91448ca9d36036e3355e8d78e087ed |
| 69997_TTA8.zip | 435f98cb0ea38ad5832b7dd9bc189c866db5a69c9a70a1bc2d623057e730e91c |
| 69997_TTA8_FIXED8D_EXP41_STYLE.zip | 5167656552e6ac3e395f244fca19da9897527be3bc45363927da75f56f6f4a1a |
| 69997_TTA8_LOCKED8D_69_276229.zip | 241f618b7a980fb8862460847d092b286bf6c1404edef337d038706793dfa36d |

以上Full6996六包是本次真实本地读取与SHA匹配，官方检查PASS引用旧回执而非本次重跑。文件名保留原样；不能因后来canonical名调整就复制或改名历史包。

**Stage59/66本地包也已存在**：本次Glob确认`提交包/stage66_stage59_20260911/`中四个原始目录共20个ZIP，另有`UPLOAD_ORDER/`中的10个去重上传顺序包。它们不是额外10个模型；协调方说明历史SHA核验已完成，本轮未重新hash。这纠正了早期窄范围查找造成的“未下载”误述。目录包含：`stage59_submission`、`autopack_stage59`、`latecwfa_submission_stage66`、`autopack_stage66`、`UPLOAD_ORDER`。

## 四、证据索引与限制

- `ALL_SCHEMES_LIVE_STATUS_20260914.json`：18:56最新训练观察副本。
- `POST_RAREEXPOSURE_ACK_SNAPSHOT.json`、`STORAGE_AWARE_HANDOFF_REPORT.md`：18:39及更早交接证据。
- `提交包/Full6996_priority_20260912/FULL6996_PRIORITY_PACKAGES_REPORT.json`：6包PASS、last别名、all_done=false。
- `HISTORICAL_RECOVERY_S1_AUDIT.json`：Stage66训练/包身份与单独协议成绩。
- `HISTORICAL_CALIBRATION_RECOVERY_RESULTS.json`：缓存恢复受阻、无新执行。
- `CANDIDATES.json`、`DISK_AUDIT.json`：Stage59/66历史包存在性清单，SHA缺失不能补造。
- `paramwise01_protocol_v2/CPU_PREFLIGHT_REMOTE.json`：历史容量受阻副本。
- `post_rare_recovery_v1/FACTORY_SYNTHETIC_02/SYNTHETIC_CLI_RESULT.json`：仅CPU合成样例。

机器可读明细见`ALL_SCHEMES_RECENT_AUDIT.json`。本次只生成报告，没有修改任何queue、部署、远程操作、新训练、推理或线上提交。
