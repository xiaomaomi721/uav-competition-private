# PARAMWISE01_PROTOCOL_UPDATE_ACK

核验时间：2026-09-14 16:44:39（服务器时间）。

## 当前训练：保持有效，未停止、未重启、未改参数

| 实验 | 物理 GPU | Formal PID/SID | 已记录 step | 状态 |
|---|---:|---:|---:|---|
| RareExposure v2 | 2 | 283448 | 48614 / 56000 | 正常运行 |
| Paramwise01 | 3 | 2115724 | 1934 / 56000 | 正常运行 |

两次交接后观察：RareExposure 48584→48614，Paramwise01 1903→1934。GPU进程归属正确。

正式配置SHA保持不变：
- RareExposure：`fc6005a5e8a59e03f134a17b832e620e2764a36697e789b5dd5c60d9a181fe09`
- Paramwise01：`b13dabef0c44ac802bff16443d8fc7c1d81606540c6dde2c3cc5b22321f4068d`

## 磁盘清理：STOP CLEANUP，未达到容量目标

- 已完成只读审计、逐路径计划及清理报告。
- safe_to_delete=yes：0项、0字节。
- 实际删除：0项；实际回收：0字节。
- 31个历史ZIP（137,282,345字节）具有来源/SHA闭环，但打开文件检查可见性不足，仍为safe_to_delete=no，全部保留。
- 清理报告时可用10.3438GiB；最新可用10,894,372,864字节，约10.15GiB。持续写入导致的空间变化不是清理收益。
- 20GiB目标和25GiB优选目标均未达到。所有受保护checkpoint、logits/cache、formal目录、日志及manifest未删除或移动。
- 未以权限提升、删除其他用户数据或身份不明历史资产来凑空间。

## 新协议：已冻结并接入实际调度器

`matched-control -> positive-effect gate -> model-specific 8D scan -> multi-package bank`

- Stage23相同D4/FP32协议三bias基线需要重建；已审计历史候选不能直接当作匹配基线。
- best/52k/56k按模型tensor内容去重；40k/44k/48k保留但不默认test推理。
- 正效应门槛：DeltaCore≥0.001 或 DeltaFixedCal≥0.001（即+0.10个百分点）。
- 符合门槛后按预注册匹配指标选择，执行CPU轴敏感性/联合局部/稳定盆地搜索，不做5^8穷举。
- 严格一次forward设计保留各候选val缓存；选中checkpoint的test只forward一次，固定三角色+七校准角色复用缓存。
- 500张实际预测内容去重；保留角色别名，不凑重复ZIP；无自动线上提交。

### 实际交接

- 仅对身份核验后的旧Paramwise调度器2113375发送pidfd SIGSTOP，现状态T；原锁仍持有，不自动恢复。
- Formal 2115724未接收信号；RareExposure的formal与调度器未触碰。
- 新调度器PID/SID2187127，PPID1，独立successor.lock已持有。
- 旧状态和报告已归档，新状态保留原smoke PASS及唯一formal历史。
- 实际`--validate-only`退出0/PASS；实际调度器接管后两条训练step继续增长。
- 第一次交接准备的只读检查因预期argv漏写Python `-B`而拒绝；未发信号、未创建交接产物。读取真实argv后精确修正启动检查，再执行交接。未改任何训练代码或参数。

## 容量阻塞：生产后处理尚未开始

真实CPU CLI预检退出3，`BLOCKED_INSUFFICIENT_STORAGE`，`CALIBRATION_STORAGE_GATE_BLOCKED`。

| 预留项 | 字节 |
|---|---:|
| 最多三个FP32 val缓存 | 140827951104 |
| 选中checkpoint的FP32 test缓存 | 16777216000 |
| NPZ容器开销 | 153911296 |
| 最多16角色PNG/ZIP保守上界 | 52477952000 |
| 安全余量 | 5368709120 |
| 合计所需可用空间 | 215605739520 |

约200.80GiB需求；当前约10.15GiB，约缺190.65GiB。即使清理到25GiB也不足。这是当前严格缓存/保留协议的保守门禁，不是声称最终压缩文件一定占用这么多。未计入压缩收益，不降FP16、不自动删除缓存、不启用metrics-only替代。

- 预检覆盖1399验证图、1399 GT、500 test图。
- `torch_imported=false`，`GPU_started=false`，`stageA_status=NOT_STARTED`。
- 错误协议SHA被实际CLI拒绝；重复输出目录被拒绝且原回执保留。
- GPU推理、真实大缓存CPU扫描和正式出包均尚未执行，不能标为PASS。
- 新调度器等待原formal自然结束并审计后重查容量；不足则停止并保留阻塞回执，不自动重试、不回退旧协议。
- RareExposure原后处理门禁约14.16GiB；当前空间也低于该门槛。该队列未修改，不能承诺其出包能在现有容量下完成。

## 冻结身份与证据

- PROTOCOL SHA：`8c5411cf1ae368ae613f147990243c1128af1e2e5ba4dee84b025446a6bae4ee`
- DEPENDENCIES SHA：`c266d091191e7c647a493370958036b9dc35ea5050fb25f754cb64c6a8c598d9`
- HANDOFF_POLICY SHA：`5a02b29fbc5eb8849ae208922fcd8ae7bb21818df25f91167efb1d78c54e6ca2`
- 新调度器源码SHA：`12b7adf240226c7f70a8bf884b9fdb6ca74002b6a30ba81cd5cff15bc1ee2293`

本地：
- `DISK_CLEANUP_PLAN.md` / `DISK_CLEANUP_REPORT.md`
- `paramwise01_protocol_v2/CPU_PREFLIGHT_REMOTE.json`
- `HANDOFF_LIVE_OBSERVATION_01.json` / `HANDOFF_LIVE_OBSERVATION_02.json`

服务器：
- `/home/jhy/gpu3_paramwise01/protocol_v2/HANDOFF_LAUNCH.json`
- `/home/jhy/gpu3_paramwise01/protocol_v2/HANDOFF_VALIDATION.json`
- `/home/jhy/gpu3_paramwise01/protocol_v2/handoff_execution/STATUS.json`
- `/home/jhy/gpu3_paramwise01/queue/PARAMWISE01_FINAL_REPORT.md`（持续更新）

**总结：训练参数完全未改；协议部署完成；清理已安全停止；容量问题未解决；生产后处理与线上提交均未执行。**
