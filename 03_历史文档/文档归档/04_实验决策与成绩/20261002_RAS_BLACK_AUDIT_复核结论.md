# 2026-10-02 RAS-KD / BLACK_CORE_DROP 完整审计复核

## 结论先行

- **线上结果（用户报告）：RAS-KD TREAT `71.042244`，相对 Arm B 为 `−0.003071pp`；BLACK_CORE_DROP `71.029651`，相对 Arm B 为 `−0.015664pp`。两者都没有刷新冠军。**
- **两个候选共同带有 zero-safe `xlogy` 修复，因此在 RAS CONTROL 得分前，不能把相对历史 Arm B 的差值全部归因于 RAS 或黑区删除。**
- **今日用一个槽补交 RAS CONTROL，即可同时计算 `Δxlogy`、`ΔRAS`、`ΔBLACK`；不再继续扫 RAS 强度或 BLACK 变体。**
- **TCA 不吸收 RAS 的 D4-JS 权重，并恢复原预注册的全网格 selected/complement，不加入 BLACK_CORE_DROP；保留 `xlogy` 数值修复。**
- 分数证据级别为 `USER_REPORTED`，当前未归档平台 submission ID、时间或截图。

预注册计算：

```text
Δxlogy = RAS_CONTROL − 71.045315
ΔRAS   = 71.042244 − RAS_CONTROL
ΔBLACK = 71.029651 − RAS_CONTROL
```

## 1. 审计包身份

文件：`C:\Users\Lenovo\OneDrive\Documents\桌面\20261001_RAS_BLACK_AUDIT.zip`

- 外层 SHA256：`2d1a20c184d32c6d2d2489251d7108bb5a2c67015f9af0245af9f7728ae6844d`
- 大小：11,619,731 bytes
- 成员：91
- ZIP CRC：PASS

当前审计包包含 RAS CONTROL 提交包、CONTROL/TREAT 代码/日志/摘要/manifest、缓存 manifest 和 BLACK_CORE_DROP 全套报告。它**没有包含 RAS TREAT 提交包本体**，因此当前 ZIP 单独看并非双提交包完全自包含。TREAT 包曾在先前 `方案1.zip` 中独立通过格式与像素审计，其 SHA 与本审计包 manifest 一致。

## 2. RAS 包身份与提交格式

### CONTROL

- checkpoint SHA256：`5bc7a35303859dea5e0091bd35ac855ccdf4a74795db17897e079b11a7755548`
- ZIP SHA256：`1700504fbc9659bbb00c386a63c5624902fbf0f45ba1cce09bd8f2727be648b4`
- 当前审计包内 payload 重新计算 SHA：一致
- 1300 个文件，`test2_1.png` 到 `test2_1300.png` 数字序
- flat root、unique names、CRC PASS
- 全部 `L`、1024×1024、uint8、标签 1–8
- single model / one checkpoint / no inference ensemble

### TREAT

- checkpoint SHA256：`8832f2ce79f9c1e913f03592d2d829a593cc664d25fb38a5d8c9e8a53bcc39dc`
- ZIP SHA256：`50175d994eba7fa8a67c87bacda76d6289e8316c0431445923d5f12df6e164e3`
- SHA 与先前独立审计的 `方案1.zip` 内包一致
- 先前审计已确认同样满足 1300 图、数字序、flat root、L/1024²/uint8/1–8

## 3. CONTROL/TREAT 是否为可信匹配对

共同条件：

- A05 初始化与固定 A05 teacher；
- seed `20260922`；
- source=0.1，target=1.0，500-step ramp；
- 2000 attempted loops；
- 同一 trainable scope、optimizer、LR、阈值、Bstar、增强和 endpoint；
- 两组共同使用 `torch.xlogy(q,q)-q*logp_student`；
- 两组均 1997 次实际更新；
- AMP skip 完全一致：`[310, 1136, 1225]`；
- 非有限 loss 均为 0。

匹配证据：

- 日志中 41 个记录点的 step、target stem、D4 view、mask coverage、blind coverage、stepped 全部一致；
- history tail 的最后 10 步 target stem、D4 view、stepped 全部一致；
- 八类 candidate 与 selected 累计计数逐类完全一致；
- TREAT 分支不新增随机调用，只读取固定 disagreement cache 并作确定性权重运算；
- 1300 个 disagreement cache SHA 全部核验，teacher SHA、图像列表、D4 views、Bstar、grid、dtype 均绑定一致；4 个重算样本伪类别和 selection 均 0 像素变化。

限制：

- 没有保存全 2000 步 target/source 样本序列 SHA；
- 没有保存每一步光度增强参数或增强结果 SHA；
- 当前审计包没有 checkpoint 本体与 TREAT ZIP 本体。

因此结论是：**匹配对证据充分到可以相信 RAS 权重是两组的唯一训练变量，但未达到逐步加密重放级别。**

## 4. RAS 权重是否真正进入反传

代码路径为：

```python
w = 1.0 - 0.5 * d[None]
weights[selected] = w[selected] / w[selected].mean()
lt = (kl[mask] * weights[mask]).mean()
loss = source_coeff * lsrc + target_coeff * ramp * lt
```

因此权重确实参与 `loss.backward()`，不是只用于日志。

类内归一化精确保留每张图、每个已入选伪类别的权重和，但**不会**取消空间重加权，也不保证 KL、梯度方向或梯度范数不变。

全程统计：

- selected 累计：104,067,122
- TREAT raw `w` 均值：0.99905293
- normalized `a` 均值：0.999999999
- normalized 范围：约 0.849–1.062
- TREAT unweighted KL 累计：2,212,275.62
- TREAT weighted KL 累计：2,195,836.34
- 比值：0.992569，即累计下降约 0.743%
- 12-step 抽样中 `|a-1|>0.05` 仅 0.308%，`|a-1|>0.10` 仅 0.0104%

所以准确表述是：**干预真实存在，但绝大部分像素的干预很弱；类内归一化没有数学上完全抵消它，却把作用限制在很窄的空间重排。**

## 5. 三方预测比较

总像素数均为 1,363,148,800。

### RAS CONTROL vs RAS TREAT

来自先前双包独立审计：

- changed：503,951（0.036970%）
- identical images：11 / 1300
- CONTROL→TREAT 主转移：Agricultural→Background 201,157
- 反向 Background→Agricultural：5,351

### 历史 Arm B vs RAS CONTROL

本次从审计包 CONTROL 与冻结 Arm B 重新计算：

- Arm B ZIP SHA：`9b426a5bda7fcc8b6b96af9a7a6c2a8e86467dbb4d8bb61441717538f29f1885`
- changed：120,722（0.008856%）
- identical images：19 / 1300
- CONTROL − Arm B 类别份额百分点：
  - Background −0.001810
  - Building −0.000161
  - Road +0.000247
  - Water −0.000251
  - Barren −0.000187
  - Vegetation −0.001118
  - Agricultural +0.003287
  - Vehicle −0.000007
- 最大转移：Arm B Background→CONTROL Agricultural 45,623

### 历史 Arm B vs RAS TREAT

来自先前独立审计：

- changed：441,079（0.032357%）
- 主要变化：Agricultural→Background 159,297
- Agricultural 份额约 −0.01230pp

三角关系与方向一致：RAS 的主要额外行为是把少量 Agricultural 改成 Background，但无 test2 GT，不能把这一行为解释为纠错。

## 6. 为什么 RAS CONTROL 不能直接当历史 Arm B

历史 Arm B 与 RAS CONTROL 的训练脚本差异中，唯一会改变数学结果的共同修复是：

```python
# historical
q * (q.log() - logp)

# RAS pair
xlogy(q, q) - q * logp
```

二者在实数数学上等价，但 `xlogy(0,0)=0`，可避免 FP32 softmax 下溢为 0 时出现 `0 * log(0) = NaN`，且有限值路径也可能有最后位数差异。其余主要差异是统计、日志和 fail-stop，不参与梯度。

历史 Arm B 与新 CONTROL 虽然 AMP skip 都为 `[310,1136,1225]`，最终预测仍有 120,722 像素不同。因此：

- RAS TREAT 不能只和历史 71.045315 做单包解释；
- 若要在线回答 RAS 因果问题，必须提交新 CONTROL 与新 TREAT 两包配对；
- 也正因为需要消耗两个槽，证据门应更高。

## 7. 线上结果与今日提交决策

昨日已人工提交并由用户报告：

- RAS TREAT：`71.042244`，相对 Arm B `−0.003071pp`；
- BLACK_CORE_DROP：`71.029651`，相对 Arm B `−0.015664pp`。

它们都没有刷新冠军，但两者相对历史 Arm B 都混入了共同 `xlogy` 数值修复。现在只需补交现成 RAS CONTROL 一个包，就能同时隔离 RAS 与 BLACK 的净效应。因此今日第一槽固定为 RAS CONTROL（ZIP SHA `1700504fbc9659bbb00c386a63c5624902fbf0f45ba1cce09bd8f2727be648b4`）；它是因果闭环，不是新的参数扫描。

第二槽只保留给 TCA v4 全网格正式包。无论 CONTROL 得分如何，都不回头扩大 RAS 强度或继续 BLACK 变体，也不根据该分数临时修改 TCA。

## 8. BLACK_CORE_DROP 结论

继续关闭：

- 黑区 target KL 贡献约 0.53%；
- 本地变化约 +0.0052pp，处于噪声量级；
- 相对 Arm B 预测变化约 0.098254%；
- 不是当前突破瓶颈，不提交、不继续加码。

## 9. TCA v4 修订与交付

在线结果并不允许在缺少新 CONTROL 分数时把 RAS/BLACK 的差值拆开，但足以确认两个复合包都没有刷新冠军。更关键的是，原 TCA v3 实现实际排除了 black core，与预注册的“不加入 BLACK_CORE_DROP”不一致。为恢复唯一变量，TCA v4：

- 不吸收 RAS D4-JS 权重；
- 不排除 black core；
- selected/complement 严格划分完整 256×256 teacher 网格；
- 保留 zero-safe `xlogy`；
- 不改变 mu 公式、训练预算、A05 teacher、Bstar 或出包协议。

固定公式版本：

```text
tca-v4-xlogy-zero-safe-all-pixels-selected-plus-gradient-calibrated-complement
```

本地验证：

- py_compile：PASS
- Ruff：PASS
- 11/11 unit tests：PASS
- CLI import/help：PASS
- black/source/network/upload 静态禁用项：无命中
- standalone delivery tests：11/11 PASS

更新交付：

- 代码目录：`C:\Users\Lenovo\OneDrive\Documents\桌面\TCA_TARGET_COMPLEMENT_ANCHOR_V4_队友执行_20261002`
- ZIP：`C:\Users\Lenovo\OneDrive\Documents\桌面\TCA_TARGET_COMPLEMENT_ANCHOR_V4_执行包_20261002.zip`
- ZIP SHA256：`25f5a1c9546492311c49b8b1f68681c25da03e4f7dfa9a2dbd7cf25aacfe7881`
- receipt SHA256：`f321a3aed2cc06bd0cd6e62e7cb64c6b285c6a252acebf59fec7d1ac6bc72ecb`

旧 TCA bundle SHA `b64e4c...` 与 `d433878...` 均已过期；旧 calibration receipt/checkpoint 不得与 v4 混用，必须从全新 calibration 开始。
