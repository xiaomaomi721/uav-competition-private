# 实验治理模板（2026-09-26）

这些文件用于约束外部 AI/执行 AI 的实验行为，不是自动执行器。

## 使用顺序

1. 复制并填写 `METHOD_CARD_TEMPLATE.md`，在 eligibility diagnostic 前锁 SHA。
2. 每个 RUN_ID 独立复制 `ENTRY_CONTRACT_TEMPLATE.json`；只有用户批准且 preflight 完成后才能把状态改为授权。
3. 运行无论成功失败都填写 `EXECUTION_REPORT_TEMPLATE.md`。
4. `FAILURE_TEMPLATE.json` 必须存在；成功时 failure 字段明确为空/none，失败时不得省略。
5. 人工例外用 `EXCEPTION_RECORD_TEMPLATE.json`，不能反写原 FAIL。
6. 平台提交由用户人工执行；用 `PLATFORM_RECEIPT_TEMPLATE.json` 绑定 checkpoint、ZIP 和平台回执。
7. 所有被报告、提交、作为 parent/teacher 或因果对照端点的产物写入 `RETENTION_MANIFEST_TEMPLATE.json` 并锁定。

## 强制规则

- 只允许物理 GPU2：`CUDA_DEVICE_ORDER=PCI_BUS_ID`、`CUDA_VISIBLE_DEVICES=2`、内部 `cuda:0`。
- GPU0/1 属于他人，GPU3 禁用；不得探测、停止或复用其进程。
- 输出目录必须预先不存在；失败产物不得覆盖。
- 不得静默修改 config、数据、steps、LR、scope、阈值或推理协议。
- 不得因早期效果不佳终止 formal；工程身份、安全或数值错误除外。
- 缺 `EXECUTION_REPORT.md`：`RESULT_UNTRUSTED_MISSING_EXECUTION_REPORT`。
- 不得自动提交；test2 无 GT 统计不得表述为准确率。
