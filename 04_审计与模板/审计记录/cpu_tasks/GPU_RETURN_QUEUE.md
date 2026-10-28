# GPU_RETURN_QUEUE

> 生成于 2026-09-19（CPU 阶段结束）。**不要启动**。下一次 GPU2 窗口按此顺序执行。
> 当前冠军冻结：`278 = 0.706154`（base `261 = 0.702836`，ATDA 贡献 +0.3318pp）。
> 距 0.710000 还差 0.3846pp。

## 队列（按优先级）

### 1. 免费候选推理（零训练，先测这些，总数 ≤ 15）

所有候选已构造好、out-of-scope 相对 theta261 全部 byte-identical，见
`atda_20260918/cpu_candidates/`（`manifest.json` + `manifest_layerwise.json`）。
只需 D4 TTA8 前向 + Bstar 出包，复用 `_atda_export.py`。

| 组 | 候选 | 文件 | 构造公式 |
|---|---|---|---|
| D alpha-narrow | a1.10/1.15/1.20/1.30/1.35 | `d_s8k_a1p*.pt` | theta261 + α·(θ8k−θ261) |
| C EMA | ema8k-direct / 0.5s+0.5e / 0.75s+0.25e / 261+1.25·ema | `c_*.pt` | EMA 8k 混合 |
| E late-extrap | X025 / X050 | `e_X0*.pt` | θ8k + k·(θ8k−θ6k) |
| F layerwise | s2/s3 拆分 4 个 | `f_*.pt` | stage2·α2 + stage3·α3 |

**优先级逻辑**：先测 D（alpha 窄化，判断方向是否继续单调涨）；若 a1.30/a1.35
再涨，直接锚定"幅度增益"，无需再训。再测 E（late 外推）判断 8k 后增量是否仍
正。F 用来定位增益在 stage2 还是 stage3。C 是 EMA 的备选。

### 2. exact 8k→10k resume

先给 `_atda_train.py` 加 `--resume`（唯一缺失，见
`cpu_resume/RESUME_READINESS.json`），然后跑 `cpu_resume/resume_8k_to_10k.sh`。
`total_steps` 保持 10000。

### 3. 根据 10k 平台判据

- 若 10k 继续明显上涨（≥ +0.1pp over 278）→ 上 `ATDA_16k_horizon.yaml`。
- 若 10k 平台（涨幅 < 0.1pp）→ 转 `Pseudo500-v2`（teacher=278 预测，见
  `pseudo500_v2_cpu/`，一致率 98.46%）。

### 4. 仅当 ATDA 与 Pseudo 轴都不足，才允许讨论 clean B5。

## CPU 阶段全程禁止（已遵守）

- no-decay / RareExposure / SEMEDA 旧 gate / calibration 大扫
- DINO / MixStyle 旧资产 / B5 实现
- CPU SegFormer 全量 forward / 新训练 loss 发明 / 其它 GPU / 旧 donor 路线

## 关键结论摘要（CPU 阶段产出）

1. **278 为什么成功**（`cpu_audit_261_vs_278/`）：改动集中在低置信像素
   （changed 置信 0.581 vs unchanged 0.954）；净面积 Barren −0.060pp、
   Vegetation +0.062pp（对齐历史 Barren 过预测方向）。
2. **收益 = 方向成熟 × 幅度**（`cpu_geometry/`）：累计方向 cos 0.87→0.98 高度
   一致；alpha×RMS 单调 0.00019→0.00065 对应分数 0.7033→0.7062。**同一方向继续
   放大仍是最优延续**。
3. **8k 仍未收敛**（`cpu_log_analysis/`）：loss 窗口均值 0.214→0.168 单调降。
4. **curriculum 分层**（`cpu_curriculum/`）：299 stable / 173 medium / 28 high-shift。
5. **源域覆盖**（`cpu_domain/`）：见 `domain_analysis.json`。
6. **复现资产**（`release_278/`）：DRY_RUN 全通过。

## 资产清单

| 路径 | 内容 |
|---|---|
| `SCORE_LEDGER/SCORE_LEDGER.{csv,json}` | 分数账本（SHA 锚定） |
| `release_278/` | 复现资产 + hashes.sha256 |
| `cpu_candidates/` | 15 个构造好的候选 |
| `cpu_resume/` | resume 脚本 + 16k yaml + readiness |
| `pseudo500_v2_cpu/` | Pseudo500-v2 标签/manifest |
| `cpu_audit_261_vs_278/`, `cpu_geometry/`, `cpu_log_analysis/`, `cpu_curriculum/`, `cpu_domain/` | 各分析产物 |
