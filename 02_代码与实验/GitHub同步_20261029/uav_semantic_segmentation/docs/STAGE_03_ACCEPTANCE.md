# Stage 03：独立验收与证据门禁

## 验收结论

| 门禁 | 状态 |
|---|---|
| core_training_status | PASS |
| evidence_gate_status | CONDITIONAL_PASS |

核心训练完整性和本地指标通过；正式权重到 500 张测试预测的完整证据链尚未独立关闭，因此不得描述为无条件完整 PASS。

## 独立验收范围

- 只读检查正式验收目录的 20 个文件、2 个子目录和 19,792,248 bytes 内容。
- 核对配置、completion、56000 行训练记录、metrics、逐类 IoU、CSV/NPY 混淆矩阵和 8 张验证三联图。
- 核对独立验收报告、机器可读验收结果与文件 manifest。
- 检查外部 500 张测试预测是否存在且格式可读。
- 未训练、未推理、未修改原始验收目录。

## 验收包 SHA256

- 验收 ZIP SHA256：`AC8BA3D4985B54753FB6F607666730C26982E39A476009194C00A3C056B41618`
- 实际计算值与声明值一致。
- ZIP 内 20 个文件与只读验收目录的文件名、数量和内容 SHA256 完全一致。

## 配置检查

26/26 项配置核对通过：实验 ID、正式阶段、模型、类别数、crop、纯 Cross Entropy、AdamW、学习率、固定划分、batch、step、验证间隔及禁用的 Rare Crop/Domain Aug 均符合正式基线要求。原 config 中的三个机器绝对路径已在仓库快照中替换为占位符或相对配置路径，训练实质参数未改变。

训练环境快照属于曾昭虹执行 Stage 03 正式训练的电脑，不应与本轮在另一台电脑上执行的静态检查环境混为一谈。

## 混淆矩阵独立重算

- shape：8×8，行=真值、列=预测。
- 总有效像素：1,445,194,394。
- 重算 mIoU：0.7123106568503285。
- 重算 Pixel Accuracy：0.8527743645537557。
- 重算 Vehicle IoU：0.6690723797071482。
- 重算 Barren IoU：0.4835624404398186。
- CSV、NPY、metrics.json 与 per_class_iou.csv 一致，逐类最大误差为 0。

## 日志连续性

- step 1→56000 连续，无断层、重复、倒序或归零。
- 验证 step 为 1400、2800、…、56000，共 40 次。
- 无 OOM、NaN 或 Inf。
- 后期进入平台期但最后一次验证创最佳；学习率在 56000 step 降至 0。
- 单点耗时异常不伴随 step 重置或指标损坏，不构成中断证据。

## 可视化检查

8/8 张三联图均可解码，几何和调色板自洽，未发现系统性错位或全局坍缩。可视化支持 Building→Background、Agricultural/Barren/Vegetation 混淆和 Vehicle 小目标漏分等错误模式。三张训练曲线均小于 2 MB、可读、无官方原始图像/标签/三联图内容，也无机器路径水印，允许进入仓库。

## 原始包只读保真

正式验收目录和独立验收目录均保持只读，未被本轮修改。仓库不复制权重、完整日志、三联图、官方数据或原始 acceptance_result.json，只生成脱敏摘要。

## 条件项原因

验收摘要声明正式 best checkpoint 已用于测试推理、strict load 的 missing/unexpected keys 均为 0；但验收包内没有权重及未编辑的重载/推理日志。500 张既存预测可证明完整推理曾完成，却不能独立证明其由预期 SHA256 的 checkpoint 生成。

## 官方 ID 8 检查错误说明

原验收脚本曾按训练 ID 0–7 检查测试预测，把官方 ID 8 记录为 `invalid_class_ids`。这是口径错误：

- 训练内部 ID：0–7。
- 官方提交 ID：1–8。
- 官方 ID 8 = Vehicle，是合法类别。
- 本轮逐张复检显示 500 张预测实际 ID 为 1–8，ID 8 出现在 100 张中，非法 ID 文件数为 0。
- 该问题只影响测试预测格式统计，不影响 Stage 03 本地 mIoU、Pixel Accuracy、混淆矩阵和逐类 IoU。
- 仓库归档摘要已按官方 ID 1–8 修正解释；原 acceptance_result.json 不复制、不修改。

## 权重搜索与哈希状态

- 正式 best checkpoint 预期 SHA256：`5846D7625FC92CCFDBB065BFC9F8910348F4B81C0010AE253AD598443DCBB4A1`。
- 验收摘要声明的正式 last checkpoint SHA256：`1DB63E34CF899DAF1F60F654DAA98B119932A3DE72E9196D80ABE6716E6E2C7A`。
- 本轮在限定范围内搜索并流式计算了 29 个同名候选的 SHA256；没有候选匹配上述正式哈希。
- `weight_hash_verified=false`。正式权重仍待曾昭虹提供或重新备份，不能把 quick pilot 或其他项目的同名文件误认为正式权重。

## 路径与 Git 策略

仓库跟踪文档不记录训练电脑、本机数据、权重或输出目录的绝对路径。真实路径只保存在被 `.gitignore` 排除的 `local_records/stage03_stage04_paths.json`；模型权重永不复制进 Git 仓库。
