# Stage 03 脱敏验收摘要

## 状态

- `core_training_status: PASS`
- `evidence_gate_status: CONDITIONAL_PASS`

核心训练、配置、日志连续性和混淆矩阵复算均通过；条件项来自正式权重到 500 张预测的证据链尚未独立关闭。

## 核心核对

| 项目 | 结论 |
|---|---|
| 验收 ZIP SHA256 | `AC8BA3D4985B54753FB6F607666730C26982E39A476009194C00A3C056B41618`，MATCH |
| 配置 | 26/26 项通过，运行时绝对路径已脱敏 |
| 训练日志 | step 1–56000 连续，40 次验证，无 OOM/NaN/Inf |
| 混淆矩阵 | 8×8，CSV/NPY/metrics 一致 |
| 重算 mIoU | 0.7123106568503285 |
| 重算 Pixel Accuracy | 0.8527743645537557 |
| 可视化 | 8 张三联图自洽；仅归档 3 张纯训练曲线 |
| 原始验收目录 | 只读保留，未修改 |

## 官方 ID 8 修正

原验收脚本把测试预测按训练 ID 0–7 检查，因此将官方 ID 8 误记为非法。正确提交 ID 是 1–8，官方 ID 8=Vehicle，必须视为合法。本轮逐张复检 500 张预测后确认：实际 ID 为 1–8，ID 8 出现在 100 张中，非法 ID 文件数为 0。

该错误只影响测试预测格式统计，不影响 Stage 03 的本地 mIoU、Pixel Accuracy、混淆矩阵或逐类 IoU。仓库不复制原始 `acceptance_result.json`，仅保留本修正版脱敏摘要。

## 权重证据

- 正式 best 预期 SHA256：`5846D7625FC92CCFDBB065BFC9F8910348F4B81C0010AE253AD598443DCBB4A1`
- 正式 last 声明 SHA256：`1DB63E34CF899DAF1F60F654DAA98B119932A3DE72E9196D80ABE6716E6E2C7A`
- 本轮搜索并流式哈希 29 个同名候选，匹配数为 0。
- `weight_hash_verified=false`；正式权重仍待曾昭虹提供或重新备份。

现有 500 张合规预测能证明完整推理曾执行，但不能单凭 PNG 独立证明其由上述 best checkpoint 生成。这一证据缺口不否定 Stage 04 已记录的线上成绩。

## 仓库安全

摘要不含机器绝对路径、完整训练日志、官方图像或权重。真实本地路径仅存放在 `local_records/` 的 Git 忽略文件中。
