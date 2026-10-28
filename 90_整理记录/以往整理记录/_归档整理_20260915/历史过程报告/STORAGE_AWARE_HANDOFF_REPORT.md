# STORAGE_AWARE_HANDOFF_REPORT

**状态：STREAMING_HANDOFF_VERIFIED。** 两个post-training scheduler已切换到新FP32流式直写ZIP路径，formal没有停止、重启、resume或配置变更。

## 2026-09-14 18:33:26服务器实测

| 实验 | GPU | Formal PID/SID | step | 新supervisor PID/SID | 新supervisor PPID |
|---|---:|---:|---:|---:|---:|
| RareExposure v2 | 2 | 283448 | 53999/56000 | 2399435 | 1 |
| Paramwise01 | 3 | 2115724 | 7483/56000 | 2399454 | 1 |

两者状态均FORMAL_RUNNING_STREAMING_INSTALLED，新锁持有，实际GPU进程归属正确。当前free10,870,472,704字节，约10.12GiB。

## 旧调度器

- Rare旧watchdog281314已精准pidfd SIGSTOP，状态T，原锁保持。
- Param旧stream-cache controller2187127已精准pidfd SIGSTOP，状态T；更早原parent2113375继续保持T。
- 所有formal PID未接收信号；没有process-group signal。
- 旧状态按唯一文件名归档并记录SHA，新状态只写新namespace，不热改旧代码/冻结文件/训练历史。

## 新状态位置

- `/home/jhy/storage_aware_calibration_v1/rare/supervisor_work/STATUS.json`
- `/home/jhy/storage_aware_calibration_v1/paramwise/supervisor_work/STATUS.json`

旧报告停留于交接快照，不再是最新进度源。

## 冻结身份

| 项目 | SHA256 |
|---|---|
| Rare DRIVER_POLICY | 91d3ec8fdebd5dc61c500cc2d8090a163194591ead61289db19180a0b1756bf5 |
| Rare SUPERVISOR_POLICY | c0a1de2d6c9f170c0e6e1589c4a85775fb129b9454dc40ffa90fe6ae88f03111 |
| Param DRIVER_POLICY | 5cda42eef0babcacb6ec503fb1c329ed803916625a68ce3b7722c94ded7978e8 |
| Param SUPERVISOR_POLICY | 759b43bb7f501166fa5c8b4fb5de2ac41da3db20f51cb65042ba8c6c8443cc46 |

正式配置SHA完全未变：
- Rare fc6005a5e8a59e03f134a17b832e620e2764a36697e789b5dd5c60d9a181fe09
- Param b13dabef0c44ac802bff16443d8fc7c1d81606540c6dde2c3cc5b22321f4068d

## 自动执行范围

等各自原formal自然完成→entry/artifact/metrics严格检查（Param保留CPU milestone审计）→等共享后处理锁→所授权GPU仍空闲才执行→模型权重去重→每checkpoint一遍500test D4 FP32 forward同时生成固定三包→原官方checker ZIP适配验证→实际500图内容去重→等待用户提供真实线上结果。

Rare只存在best/last，不补造52k；last完整56k经步数认证后作为56k别名。Param best/52k/56k。

模型专属8D仅在真实线上winner确认后进入后续两阶段流程。当前policy planner已验证；真实两阶段GPU执行/最终角色选择尚未接入，未冒称已完整实现。历史无可信cache的模型继续QUEUE_TTA，不抢GPU。

## 当前尚未执行

- 新matched test GPU推理：未开始（formal仍运行）。
- 新候选ZIP：未生成。
- 低分辨率32图等价性审计：未执行/未启用。
- 线上提交：0。

容量/CPU接口与ZIP兼容性预检详见STORAGE_AWARE_CALIBRATION_PREFLIGHT.md。GPU新完整链条尚未运行，因此此报告只证明调度交接、训练连续性与已执行的CPU表面验证，不证明未来GPU出包一定成功。
