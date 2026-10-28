# METHOD_CARD — TAST-E

> 状态只能是 `NO_RUN`、`DIAGNOSTIC_ONLY` 或 `APPROVED`。必须在 eligibility diagnostic 前填写并锁定 SHA256；结果出来后不得改参数或 PASS 条件。

## 1. 身份

- method_family: `TAST-E`
- card_status: `NO_RUN`
- prepared_at:
- prepared_by:
- method_card_sha256:
- paired_run_ids:
  - `TASTE_SOURCE_DONOR_A05_S42_8K`
  - `TASTE_TEST2_DONOR_A05_S42_8K`

## 2. 假设与已有证据

- 假设：test2 与 source 之间存在超出 source 内部波动、且可被低频 Fourier amplitude 操作覆盖的外观差异；保留 source phase 与 GT 的 styled source 可用于早层适配。
- 已有证据：`审计记录/cpu_tasks/ROUND2_SHIFT_REPORT.md` 证明存在中等 domain shift，但尚未证明 TAST-E 算子前提，故不得直接进入 GPU。
- 本轮目标：寻找相对 A05_WEIGHT 至少 `+0.20pp` 的新杠杆，而非小分微调。

## 3. 固定干预

```text
content = labeled source image
phase = source phase
A_mix = source amplitude outside centered beta window
A_mix[window] = (1-rho) * A_source + rho * A_donor
beta = 0.01
rho: warmup from 0, then frozen-RNG Uniform(0.25, 0.50)
```

- control donor: independent/permuted source; no self-pair.
- treatment donor: unlabeled test2.
- 唯一允许的因果差异：`donor_domain`。

## 4. 共同训练合同

- parent: `A05_WEIGHT_alpha05.pt`
- expected parent SHA256: `023b65800d434069f0fb0ad881083fd439b89011f731feebc88d6e9217727134`
- trainable: B3 `segformer.stages.0.*`, `segformer.stages.1.*` 及属于这些 stage 的 normalization parameters。
- frozen: `segformer.stages.2.*`, `segformer.stages.3.*`, `decode_head.*` 及其 buffers。
- loss: `0.5 * CE(original_source, GT) + 0.5 * CE(styled_source, GT)`，ignore=255。
- optimizer: AdamW, LR `5e-6`, WD `0.01`, polynomial schedule。
- seed: 42。
- endpoint: `8000 successful optimizer updates`；loop steps 和 AMP skips 必须分开记录。
- inference: one checkpoint, A05 frozen D4 TTA8+Bstar, no new bias search。
- forbidden additions: KD, entropy, pseudo-label CE, prototypes, class weights, EMA, checkpoint/model/logit fusion。

## 5. Eligibility diagnostic（CPU-only）

比较 source-source、source-test2、TAST(source,test2)-test2：

1. centered low-frequency log-amplitude radial bands；
2. RGB/Lab moments and histogram distances；
3. luminance/contrast and label-free texture-spectrum summaries。

PASS 必须全部满足：

- 至少两个独立统计族的 source→test2 距离超过 source-source bootstrap 95th percentile；
- 方向在预注册 bootstrap 中稳定；
- 冻结 TAST-E 算子使至少一个 primary 距离降低 ≥20%，其他 primary 距离无 >10% 系统性恶化；
- clipping ≤0.5%，无 NaN/Inf，结构保持检查通过。

FAIL 固定写为 `METHOD_PRECONDITION_NOT_MET`；不得放宽阈值或看结果后换 beta/rho。

## 6. 工程阶段与停止规则

- 每臂独立 4-step actual-entry smoke，只检查 GPU/shape/finite/gradient/frozen/checkpoint。
- 不设置以早期效果为门槛的 250/500-step pilot。
- formal 启动后必须跑满，除非：GPU/身份/hash/数据错误、NaN/Inf、不可恢复 OOM、无效梯度、冻结参数漂移。
- source mIoU、KL、class share、test2 flip 等不得触发 formal 早停。

## 7. 线上问题与成功阈值

```text
Delta_target_style = online(TEST2_DONOR) - online(SOURCE_DONOR)
Delta_champion = online(TEST2_DONOR) - 70.775011
```

- strong target-style lever: 两个 delta 均 ≥+0.20pp。
- candidate lever: treatment 相对 champion ≥+0.20pp。
- ≥+0.40pp：large lever。
- <+0.20pp 不扩张为方法族失败；不自动扫参数。

## 8. 结论边界

本实验可以判断：在冻结 recipe 下，test2 donor 相对 source donor 是否产生因果增益，以及 treatment 是否超过 A05。

本实验不能证明：所有 FDA/style transfer/UDA 方法有效或无效；source 指标能预测 test2；test2 无 GT 行为变化正确；论文中的完整方法已经复现。

## 9. 合规与提交

- 最终推理只加载一个 checkpoint。
- 两包必须同协议、同日成对问题。
- 不得自动提交；必须由用户人工确认。
- 平台结果只有绑定 submission ID、时间、ZIP SHA、checkpoint SHA 和截图/导出后才是 `PLATFORM_BOUND`。
