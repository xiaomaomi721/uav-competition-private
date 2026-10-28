# POST_RARE queue review — 未修复问题

本记录来自本地只读静态审查，不代表生产运行或 GPU 验证。按协调方要求暂不修改、部署或运行 queue。

## 1. canonical 索引丢失语义角色（未修复）

文件：`post_rare_recovery_v1/recovery_queue.py`，审查版本第 216–235 行，`index_packages()`。

当 `VAL_PEAK`、`STABLE_1` 等语义角色的向量与固定对照完全相同时，索引按向量匹配选中固定角色名称，且不保留原始 `item['role']`。多个角色因此得到同一 canonical 名称，不能表达“三个固定角色 + 七个语义角色”。共享向量/ZIP 是合法别名，不应删除角色身份。

建议：索引显式保留 `source_role`；历史语义角色始终使用其角色名称，不因向量等同固定对照而改名；使用独立 `alias_of`/原 ZIP 路径与 SHA 表达共享文件。禁止改名或复制历史 ZIP。

## 2. +0.10 个百分点阈值的浮点边界（未修复）

文件：`post_rare_recovery_v1/recovery_queue.py`，审查版本第 171–172 行，`boundary_stop()`。

直接计算 `rare_score - golden_score >= .10` 会受到二进制浮点误差影响。例如 69.30 − 69.20 在浮点计算中略小于 0.10，可能错误拒绝刚好达到授权阈值的用户结果。

建议：在确认分数单位为百分点、数值有限且范围有效后，使用 `Decimal(str(score))` 相减，并与 `Decimal('0.10')` 比较。不得降低阈值、伪造在线成绩或自动启动 Rare 校准。

## 验证范围

- factory/planner/scan_stats 接口未发现其他已确认阻断问题，不等于运行正确性保证。
- final factory 审查 SHA256：`002e4fb4594cc568f0e6f01fc1ca5a11f172c8810ba343b89c55e71de8047d1c`。
- 仅查阅了作者提供的 CPU 合成样例回执；未自行运行生产 GPU。
- 两项 queue 问题均未修复；当前不应标记 queue 为可部署完成。
