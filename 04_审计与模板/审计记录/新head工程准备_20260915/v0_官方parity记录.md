# 官方criterion数值一致性测试（2026-09-15）

> ## ⚠️ 2026-09-15 晚 provenance 更正（原文保留，不删除）
>
> 本文件的**掩码BCE parity 一项已失效**，原因见 §7。其余各项仍然有效。
>
> | 项目 | 状态 |
> |---|---|
> | Hungarian 匹配 parity | ✅ **仍有效** |
> | 分类损失 parity | ✅ **仍有效** |
> | 采样 semantics parity | ✅ **仍有效** |
> | **掩码BCE parity** | ❌ **INVALIDATED** — parity test used an input shape that did not reproduce the reference implementation's real `(M,P)` call path |
> | Dice | ✅ 仍注明：训练采用 **documented correct per-pair semantics**，而非 installed-version broadcast-defective literal behavior |
>
> **本文件作为整体并未作废**，只是其中一项的结论不再成立。

目的：确认我们的set-prediction实现**在无Ignore输入时逐位等价于已安装的Mask2Former criterion**。这样将来新head若失败，不会首先怀疑损失本身写偏。

**CPU，只读，未启动GPU、未改冻结trainer、未触碰KD。**

## 0. 长期固定的表述（任何接手者以此为准）

> **「官方 parity 通过」≠「我们训练时完全使用官方 criterion」。**
>
> Matcher / class loss / sampling semantics 已与当前官方实现做数值 parity；
> **Dice 训练采用文档语义正确的逐对实现，而不是已安装版本中带广播缺陷的字面实现。**
> 只有 parity 模式才逐字复现官方行为。
>
> **⚠️ 2026-09-15 更正：原地还写着「BCE …已与当前官方实现做数值 parity」——该说法已被推翻。**
> 当时的 BCE parity 使用了 `(M,1,P)` 形状，不是参考实现真实调用路径产生的 `(M,P)`；
> 因此 **v1 训练用的 mask BCE 归约（point-sum）与参考实现（point-mean）并不一致**。
> 详见 §7。**不要再用「BCE 已与官方 parity」这句话。**

不得把本文件表述为“训练目标与官方代码逐字相同”。

## 1. 结果

| 项目 | 我方 | 官方 | 绝对差 |
|---|---:|---:|---:|
| Hungarian匹配 | src=[2,3,5] tgt=[2,1,0] | 完全相同 | 0 |
| 分类损失 | 0.813935 | 0.813935 | **0.00e+00** |
| 掩码BCE | 1513.962036 | 1513.962036 | **0.00e+00** |
| Dice（逐字复现） | 1.848006 | 1.848006 | **0.00e+00** |
| 辅助层 分类 | 1.359728 | 1.359728 | **0.00e+00** |
| 辅助层 掩码BCE | 1404.141968 | 1404.141968 | **0.00e+00** |

## 2. 测试过程中发现并修正的三个真实缺陷

### (1) 匹配器softmax维度错误（会改变匹配结果）

官方在 **C+1维（含no-object）** 上做softmax后取类别代价；我原先先去掉no-object再softmax。两者不同，会直接影响匹配。

### (2) 掩码损失归一化错误（量级差约6000倍）

官方是**对每个匹配对求和，最后统一除以目标掩码数**；我原先用逐对权重 `len(src)/num_masks`，既少乘了点数，又会在多类别图像上把权重放大数倍。

### (3) 采样`padding_mode`不一致

官方用 `grid_sample` 默认的 `zeros`，我原先显式用了 `border`。已统一，并把所有采样参数收口到单一 `SAMPLING` 常量，**模块内不允许任何地方再直接调用 `grid_sample`**——否则将来某处改了 `align_corners`，Ignore重归一化的“精确性”会静默失效。

## 3. 关于已安装版本官方Dice的一个实现问题

官方 `loss_masks` 把标签以 `(N,1,P)` 传入一个按 `(N,P)` 设计的 `dice_loss`，于是发生两次广播：

- `probs * labels` → `(N,N,P)`，**非对角（跨掩码）项也进入求和**；
- `probs.sum(-1) + labels.sum(-1)` 把 `(N,)` 与 `(N,1)` 广播成 `(N,N)`，导致 **`denominator[i,j] = probs_sum[j] + labels_sum[i]`，两个下标来自不同掩码**。

对角项 `(i,i)` 仍然是正确的逐对dice，也就是文档所描述的语义。我按字面**含缺陷地**复现，使一致性检查能达到逐位相同；训练默认使用的是文档所述的逐对dice。

这不是我们改动官方代码，而是如实指出：已安装版本的实际dice与它自己的文档不一致。

## 4. 本次一致性测试**不能**证明的部分

- 官方内部采样点被我显式传入的点替换，因此**点采样策略本身未被验证**；
- 辅助层那一路，我的字段报告的是该层自身算出的匹配，官方则对每一层重新匹配；两者在该构造下按定义一致，因此**辅助匹配是防重排错误的守卫，不是逐位门槛**；
- 逐位一致只说明“无Ignore时数值等价”，**不说明该目标函数适合本比赛、也不说明新head会有效**。

## 5. 状态

| 项目 | 状态 |
|---|---|
| 四套CPU测试 | 全部通过 |
| 代码 | **冻结**，不再扩展 |
| 独立训练入口 | 未开始 |
| 真实数据训练 | 未运行 |
| GPU显存/速度 | **未测**，不做任何承诺 |

## 6. 冻结哈希（SHA256前16位）

| 文件 | 哈希 |
|---|---|
| `__init__.py` | `be5c8a099f8c4c18` |
| `head.py` | `e1e8189956148b9a` |
| `loss.py` | `4073203120c53120` |
| `semantic.py` | `10cd67aedf103e4d` |
| `smoke.py` | `508abfdb2b364eab` |
| `test_cpu_invariants.py` | `03164ad40c5f35e4` |
| `test_encoder_integration.py` | `841db0f81dbc5e0b` |
| `test_parity_official.py` | `5aad6eb867f01143` |
