# SegFormer 解码头源码事实（避免再记错）

> 用途：改多尺度上下文融合前，必须先搞清 SegFormer 解码头到底已经做了什么、局限在哪。
> 来源：transformers modeling_segformer.py 源码实读（2026-08-30）
> 更新：2026-08-30

---

## SegFormerDecodeHead 实际结构（源码级）

### __init__
1. `linear_projections`：4 个 `SegformerMLP`，把 4 个 stage 各统一到 `decoder_hidden_size`(768) 通道。
2. `linear_fuse`：`Conv2d(768*4 → 768, 1×1)` —— **4 个尺度 concat 后融合**。
3. `batch_norm` + `ReLU` + `dropout`。
4. `classifier`：`Conv2d(768 → num_labels, 1×1)`。

### forward
1. 每个 stage 过 linear_proj 统一通道。
2. 每个 stage 上采样到 **stage1 分辨率 (H/4)**（bilinear）。
3. `all_hidden_states[::-1]` concat（注意是**逆序**：stage4→stage1）。
4. `linear_fuse` 1×1 卷积 → BN → ReLU → dropout → classifier。

## 关键结论（纠正之前的错误表述）

**SegFormer 解码头已经是多尺度融合的**（4 尺度 concat + 1×1 fuse），不是"只用 stage4"。

## 它的局限（= 我们可改的方向）

1. **融合太粗**：4 尺度 concat 后只一个 1×1 卷积，没有逐尺度交互。1×1 = 加权求和，学不到"块状 vs 长条"这种跨尺度形状关系。
2. **上采样朴素**：直接 bilinear 4/8/16/32→4，中尺度(stage2/3)的形状/纹理细节被模糊。而灰色团靠形状几何区分，正需要这些中尺度细节。
3. **无显式位置/形状编码**：concat 后直接分类，没引导模型学"建筑=块状、道路=长条"的几何先验。

## 特色性改动的锚点

在已有 concat 基础上，把「粗融合」升级成「逐尺度形状感知融合」，而不是从零造 FPN/ASPP（那是 FCN 系用的，SegFormer 已有雏形）。

候选方向（等深度研究报告细化）：
- 自顶向下逐层融合（FPN 式）替代一次性 concat
- 多孔洞率并行（ASPP 式）捕捉多感受野
- 位置/形状编码注入（让模型显式学几何先验）

关联 [[赛题难点与图像质量复盘]] [[方法集成决策清单]]
