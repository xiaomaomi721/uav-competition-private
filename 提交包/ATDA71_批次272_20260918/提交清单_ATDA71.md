# ATDA-71 提交清单 — 批次 272–278（相对 theta261）

> 生成时间 2026-09-18。base 冠军 **theta261 = 0.702836**（包 261）。
> 所有 delta 均相对 theta261 组合：`theta261 + alpha*(theta_atda − theta261)`。
> 部署算子统一 **Bstar** = `[-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]`。

## 建议提交顺序（编号 = 文件名前缀，逐字节一致）

| 建议编号 | 文件名 | 定义 | SHA256 (前16) |
|---|---|---|---|
| 272 | `atda_2000_a1p0_Bstar.zip` | 2k, α=1.0 | 65f0e34bc7115ab9 |
| 273 | `atda_2k_a0p5_Bstar.zip` | 2k, α=0.5 | d82f93132989b00b |
| 274 | `atda_4000_a1p0_Bstar.zip` | 4k, α=1.0 | 593510462b4fb1c9 |
| 275 | `atda_4k_a0p5_Bstar.zip` | 4k, α=0.5 | 1b212c79e6d64e02 |
| 276 | `atda_8000_a1p0_Bstar.zip` | 8k, α=1.0 | fd90085cbaf82a1c |
| 277 | `atda_8k_a0p75_Bstar.zip` | 8k, α=0.75 | 8a8b6c08a72717b2 |
| 278 | `atda_8k_a1p25_Bstar.zip` | 8k, α=1.25 | 3daa0b4a64898ff6 |

## 训练与出包完整性（已在 GPU2 验证）

- 训练 8000 步，`frozen_identical_to_theta261=True`，`theta_ref_unchanged=True`，`amp_skips=2`。
- 8k 训练 checkpoint SHA256 `c3e87f6c51bd1fb32e6249783618d9bec7f1fc431a4a36d4ef3066d54e480eac`。
- 组合层 `out_of_scope=0.0`（stage0/1/head 从未动过）；`max_delta_in_scope` 2k 0.0066 / 4k 0.0107 / 8k 0.0135。

## 判据（相对 baseline 0.702836）

- `< 0.702336` 明显失败 → STOP
- `0.702336 ~ 0.702836` 灰区 → 只允许再走一个 gate
- `> 0.702836` target-domain 正信号
- `≥ 0.703836` 明确成功（+0.10pp）
- `≥ 0.704836` 强成功（+0.20pp）

## 折叠后最终单模型（Q 节）

finalize 完成后，3 个 8k 候选各自有：
- `finalize/<tag>/composed_atda_*.pt_folded.pt` — b_star 已折叠进 `decode_head.classifier.bias`
- `finalize/<tag>/FINALIZE_REPORT.json` — 张量级 + 逐像素一致性验证

winner 确定后，用对应的 `_folded.pt` 作为最终单模型提交（已无二次 bias）。
