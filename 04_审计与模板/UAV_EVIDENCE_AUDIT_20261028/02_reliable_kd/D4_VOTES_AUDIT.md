# D4 votes 统计口径审计

对象：`rkd_common.py::d4_stack_with_votes` 与 `rkd_teacher_diag.py` 中的"14.45 亿像素中仅 6065 个一致性≤0.99"。

---

## 1. votes 的生成方式（VERIFIED_NOW，读源码）

```python
@torch.inference_mode()
def d4_stack_with_votes(model, x):
    zs = []
    for v in range(8):
        zs.append(inv_d4(model(pixel_values=d4(x, v)).logits.float(), v))   # (1) 8 次独立前向
    zT = sum(zs) / 8.0                                                      # (2) FP32 均值
    cal = zT + b
    cal_arg = cal.argmax(1)
    votes = zeros_like(cal_arg, uint8)
    for z in zs:
        votes += ((z + b).argmax(1) == cal_arg).to(uint8)                   # (3) 逐视图独立 argmax
    q = cal.softmax(1)
    hn = -(q.log()*q).sum(1) / log(8)
    raw_arg = zT.argmax(1)
    return zT, votes, hn, (raw_arg != cal_arg)
```

**结论**：
- 每个视图是**独立的一次 `model(d4(x,v))` 前向**，无复用、无近似。
- `votes` 用**每个视图自己的 argmax** 与"均值 logits 的 argmax"比较，不是用均值去投票。
- `votes` 在 **native 256×256** 网格上生成。
- **该实现没有把 mean logits 重复当作各视图**。这一点是干净的。

---

## 2. "6065 / 1,445,188,329" 的分母口径问题（**存在口径不一致**）

`rkd_teacher_diag.py` 中的统计代码：

```python
a_up = up(votes[0])        # votes 是 256×256，此处 bilinear 上采样到 1024
...
bucket["a_lo"] += (v & (a_up <= 0.99)).sum()      # 分母 v = valid(GT != 255) @1024
```

**问题**：
1. `votes` 值域是整数 0..8，**被当作连续量做 bilinear 上采样**到 1024，这与"投票数"的离散语义不符。
2. 判定阈值 `a_up <= 0.99` 在 upsampled 空间上，等价于"上采样后仍 ≤0.99"，会**系统性低估**低投票区域（插值后中间值落在 0.99 附近的极少）。
3. 因此 "6,065 / 1,445,188,329" 这个比例的**物理含义不清晰**，不宜作为"一致性项无方差"的定量证据。

**更稳妥的替代口径（未执行，标 `NEEDS_AUTHORIZED_COMPUTE`）**：
在 **native 256 网格**上直接统计 `votes` 的直方图（0..8），分母用 native 有效像素（GT 下采样到 256 的 valid 计数）。这只需 CPU + 已有缓存，但本任务未获授权重跑统计。

---

## 3. 对 C1 结论的影响评估

**不影响 C1 的核心结论，只影响其"证据强度"。**

理由：
- C1 的最终效果是在 **source1399 mIoU** 上直接测量的：C1 − C0 = +0.0066pp。无论 votes 口径如何，这个结果不变。
- "`a² ≈ 1`，实际生效的是熵项 + 翻转降权"这一解读**依赖上述有问题的口径**，因此应降级为**合理推断**，而不是已验证事实。
- 若要严格证明 `a` 项退化，应在 native 256 上重算 votes 直方图。

---

## 4. 逐视图 logits 是否可离线复核

**不可（BLOCKED）**：`teacher_d4/*.npz` 只持久化了 `z_d4`（均值）、`z_single`、`votes`、`hn`、`flip`，**没有存 8 个视图各自的 logits**。

因此"逐视图两两分歧百分比""raw/+Bstar 两套投票分布"无法仅凭现有缓存重算 → 标 `NEEDS_AUTHORIZED_COMPUTE`。

最小复核方案（需授权、约 5 分钟 GPU）：对固定 20 张图，重跑 8 视图前向并保存逐视图 logits，再计算：
- 各视图 vs 均值预测的分歧率
- 视图两两分歧率（28 对）
- raw / +Bstar 两套 votes 直方图

---

## 5. 结论

| 项 | 判定 |
|---|---|
| votes 是否由 8 次独立前向真实产生 | **是**（源码 VERIFIED_NOW） |
| 是否复用均值 logits 冒充视图 | **否** |
| "6065 / 14.45 亿" 口径 | **存在上采样 + 离散量插值问题，不宜作为定量证据** |
| C1 − C0 = +0.0066pp | **不受影响**（直接测量） |
| 逐视图分歧可复核性 | **BLOCKED**（缓存未持久化，需授权重算） |
