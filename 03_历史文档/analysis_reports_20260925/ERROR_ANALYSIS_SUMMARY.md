# Error Analysis 摘要（alpha050，H300+R1099，1399张）

## mIoU = 0.7203

## 错误类型分布
- 边界 ≤3px：21.3%
- 过渡 3-10px：19.8%
- **区域内部 >10px：58.9%**

→ 主要是语义混淆，不是边界问题。

## 各类别 IoU / Precision / Recall

| 类别 | IoU | Precision | Recall | 内部错误% |
|------|-----|-----------|--------|----------|
| Vehicle | 0.293 | 0.91 | **0.30** | 91.6% |
| Barren | 0.516 | **0.55** | 0.89 | 69.3% |
| Agricultural | 0.767 | 0.80 | 0.96 | 73.0% |
| Background | 0.718 | 0.90 | 0.78 | 56.7% |
| Road | 0.841 | 0.91 | 0.92 | 37.5% |
| Building | 0.873 | 0.92 | 0.94 | 50.2% |
| Vegetation | 0.860 | 0.90 | 0.95 | 48.6% |
| Water | 0.895 | 0.93 | 0.95 | 35.5% |

## Top 混淆（GT → Pred）
1. Background → Vegetation: 21.8M
2. **Vehicle → Vegetation: 18.5M**
3. Background → Building: 17.8M
4. Background → Agricultural: 17.6M
5. Background → Barren: 15.4M
6. Building → Background: 11.2M
7. Background → Road: 9.9M
8. Vegetation → Background: 9.9M
9. Road → Background: 7.7M

## 结论
- **不是 boundary 问题**：58.9% 错误在区域内部
- **主要是 feature separation 不足**：Background 被过度分割成各类，Vehicle 被 Vegetation 吞掉，Barren 误报多
- Road/Water 有边界问题但 IoU 已高，提升空间小
- **下一步方向：class-aware feature separation（contrastive/prototype），不是 boundary refinement**
