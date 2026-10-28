# B4 追溯执行报告 (Retrospective Execution Report)

**日期**: 2026-09-27
**实验ID**: b4_probe_6296_pretrained_seed42
**模型**: SegFormer-B4 (nvidia/segformer-b4-finetuned-ade-512-512)
**审计性质**: CPU-only retrospective, 不修改base_probe

---

## 1. 身份纠正

| 项目 | 错误认知 | 实际值 |
|------|---------|--------|
| train count | 6296 | **6097** = 5597 real GT + 500 pseudo |
| train split SHA | — | `0fbcc0fca84b68a88bb978996a3d6312c39bb7968276d8e8762695c1831efccc` |
| val split SHA | — | `2fc7f230a4ce389767dfc9cba7be6b48b219a0606ac136c31cbd861f3ccd1e74` |
| attempted steps | — | 56000 |
| optimizer updates | — | 55980 |
| AMP skips | — | 20 |
| effective epochs | 40 | **36.7394** |
| parameter count | ~90M | **63,999,176** |
| peak VRAM | — | 10362.44 MiB |
| physical GPU used | — | **GPU3 — POLICY_VIOLATION** |
| ENTRY_CONTRACT at launch | — | **ABSENT** |
| EXECUTION_REPORT at launch | — | **ABSENT** |

## 2. Checkpoint 审计

| Checkpoint | Step | Updates | SHA256 | Size |
|-----------|------|---------|--------|------|
| best_miou.pt | 53200 | 53180 | `d7a63b0c1124f5ab213b027125322ace8ce7cf834f48fcfb45e34b20fdef6000` | 256MB |
| last.pt | 56000 | 55980 | `7e4d3376ec3fc12a0aa3c7eb418b6dd7f5b805eacedb180069522b5614a7f259` | 769MB |

## 3. B4 vs B3 本地对比

| 指标 | B3 best (step54600) | B4 best (step53200) | Delta |
|------|-------------------|-------------------|-------|
| local mIoU | 0.809790 | 0.815468 | **+0.5677pp** |

best-to-best local delta ≈ +0.5677pp
56k validation endpoint delta ≈ +0.5692pp

### B4 per-class IoU (best step53200):

| Class | IoU |
|-------|-----|
| Background | 0.7546 |
| Building | 0.8808 |
| Road | 0.8472 |
| Water | 0.8955 |
| Barren | 0.6234 |
| Vegetation | 0.9006 |
| Agricultural | 0.8008 |
| Vehicle | 0.8209 |

## 4. 可比性声明

**不得称"纯capacity delta"**。原因：

1. 500张pseudo mask由可变symlink提供
2. 历史B3在9月5日运行
3. 9月8日symlink被重指向round-2三教师`pseudo_labels_v2_multi/masks_official`
4. B4在9月26日读取当前payload
5. 正确分类：**RETROSPECTIVE_QUASI_MATCHED**
6. 差异 = architecture/pretrained difference + pseudo-payload confound

## 5. 治理状态

- B4训练完成可信（metrics.json完整，confusion matrix存在，无OOM）
- **但执行治理不合规**：
  - prospective ENTRY_CONTRACT absent
  - EXECUTION_REPORT absent at launch
  - physical GPU3 used (policy violation)
- 当前**没有**test2预测、ZIP、平台回执或线上分数
- 本审计为事后追溯，不改变训练事实

## 6. 下一步

等待B3 CONTROL/TREAT pair完成后，在GPU2用best step53200生成D4 TTA8 ZERO-bias单checkpoint包。
