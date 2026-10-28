# 历史Stage66 step52000已有TTA参考指标（非本轮扫描）

来源：`/home/jhy/gpu2_day_queue/tta_stage66_partial_v2/TTA_ANATOMY.json`，SHA `9f9292da5863274c974674c9085d9f5e81d7dbaefc04a8fbc9f4a89211c5dee3`。

Checkpoint SHA：`311919e410d43cfaef5138570c754010a62a3c5b84c8a3a736af016157ce8afb`，step52000。不能代替best51800或step56000。

| 历史1399图真FP32 TTA8 reference | mIoU fraction | 百分数 |
|---|---:|---:|
| zero bias | 0.8131277295176083 | 81.312773 |
| LOCKED8D | 0.7875682049879659 | 78.756820 |
| Golden build_p100 / EXP41_STYLE | 0.7876702342447008 | 78.767023 |

两个历史固定bias在此验证集上均低于zero：约−2.556和−2.546个百分点。因此模型专属扫描必须保留zero控制，不能把历史线上强bias当作一定更好的验证中心。这不代表线上一定同方向，也不是对训练core正效益的 matched-control判断。

这些是**旧forward在量化前已记录的参考分数**。本轮没有重新计算这些分数、没有新CPU搜索、没有新fold统计。落盘FP16缓存的精度门禁FAIL、`downstream_cache_use_allowed=false`；从汇总分数不能重建任意新bias的逐像素结果。

因此：保留已有指标用于证据回收，模型分类QUEUE_TTA；不能因已经知道三个分数而标RUN_NOW。
