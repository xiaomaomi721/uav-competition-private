# MCCA 注意力融合 + distance_transform 边界 研究结论

> 来源：deep-research 工作流（100 agent，3票对抗验证，拿到了官方开源代码 github.com/ZhengJianwei2/MCCANet）
> 日期：2026-08-31
> 用途：定"注意力融合解码头"和"升级边界生成"两个方案前的研究依据

---

## 一、MCCANet 官方开源代码已找到

**github.com/ZhengJianwei2/MCCANet**（论文 IEEE TGRS 2023, DOI 10.1109/TGRS.2023.3292112）

这解决了一个关键问题：之前抓不到原文，现在有官方 pytorch 代码，实现细节全部可复现（3-0 投票验证）。

## 二、MCCA 核心机制（已确认，含代码级细节）

MCCA 不是单一模块，是**两个注意力机制的组合**：

### ① CCA（channel-wise cross attention，通道交叉注意力）
对 skip 特征 x 和高层特征 g：
```python
avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)))   # 全局平均池化到 1×1
channel_att_x = self.mlp_x(avg_pool_x)                 # 单层 Linear(F_x, F_x)，无压缩
avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)))
channel_att_g = self.mlp_g(avg_pool_g)                 # Linear(F_g, F_x)
channel_att_sum = (channel_att_x + channel_att_g) / 2.0
scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
x_after_channel = x * scale                            # 乘性门控
```
**本质**：用高层语义 g 的通道注意力去门控 skip 特征 x，是"乘性门控单一特征"，不是加权求和。

### ② AFF（Attention Feature Fusion，注意力特征融合）
```python
# local_att: 1×1 conv → BN → ReLU → 1×1 conv → BN
# global_att: AdaptiveAvgPool2d(1) → 1×1 conv → BN → ReLU → 1×1 conv → BN
xa = x + residual
xl = local_att(xa); xg = global_att(xa)
wei = sigmoid(xl + xg)
xo = 2 * x * wei + 2 * residual * (1 - wei)   # 加权求和
```
**本质**：局部+全局注意力算权重，加权融合两个特征。默认 channels=64，压缩比 r=4。

### ③ 论文动机（3-0 验证，与我们判断一致）
> "Simple summation or concatenation techniques, such as skip connections, fail to address semantic gaps and even impose negative impacts on multiscale feature fusion."

**这正是我们想替换 SegFormer concat+1×1 的依据。**

## 三、边界监督机制（已确认，代码级）

边界 GT 用**欧氏距离变换**（不是 Sobel、不是形态学）：
```python
dist = distance_transform_edt(mask_i) + distance_transform_edt(1.0 - mask_i)
dist[dist > radius] = 0      # 超过 radius 的置 0
edgemap = (edgemap > 0).astype(np.uint8)
```
**关键**：这是**软边界**（距离越靠边缘值越小/越近中心越大），比我们 Exp38 的"膨胀-腐蚀"硬二值边界更好。

## 四、⚠️ 关键限制：MCCA 是为 ResNet-50 设计的

- MCCANet backbone = **ResNet-50**（output_stride=32，4 stage 输出 256/512/1024/2048）
- MCCA 加在 **U-Net 式解码器的 skip 连接处**
- **不能直接照搬代码到 SegFormer**，需把核心思想（CCA 门控 + AFF 加权融合）适配到 SegFormer 的 MLP 解码器。

## 五、我们两个方案的适配设计

| MCCA 原文（ResNet-50） | 适配到 SegFormer |
|---|---|
| CCA 门控 skip 特征（256通道） | 对 SegFormer 4 个 stage 的 768 通道特征，用深层特征 channel attention 门控浅层特征 |
| AFF 加权融合 | 替代 concat+1×1，用局部+全局注意力加权融合 4 尺度 |
| distance_transform 边界 | 替换 Exp38 的形态学边界，升级为软边界 |

## 六、结论

1. **注意力融合头**（CCA+AFF）有官方代码、机制清楚、对症"替代简单融合"，值得做。
2. **distance_transform 边界**比我们 Exp38 的形态学更好，值得升级。
3. FPN/ASPP 已判定不推（FPN 逐层相加=同类简单融合，文献认为有天花板）。

关联 [[多尺度上下文与边界监督调研报告]] [[SegFormer解码头源码事实]] [[uav-boundary-supervision]]
