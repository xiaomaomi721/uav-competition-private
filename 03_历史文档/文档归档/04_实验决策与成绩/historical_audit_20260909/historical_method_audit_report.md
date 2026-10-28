# Historical Method Interaction Audit — 诊断报告

> 历史实验复审 + 方法有效性重分类 + 方法组合关系分析。2026-09-09。
> 数据源：线上成绩记录表.md（权威online分）+ 失败方案归因/根因复盘/方法集成/赛题本质复盘 + 跨会话记忆判决。未知标UNKNOWN，文档矛盾显式标注。
> 状态：HISTORICAL_METHOD_INTERACTION_AUDIT_PASS（历史数据充足；online成绩全、config大部分、checkpoint有、实验报告全）

---

## 1. 最可靠的历史正收益 Top 10（按online+repeatability+transfer consistency）

| # | 方法 | online证据 | 稳定性 | 状态 |
|---|---|---|---|---|
| 1 | 8维校准(测试先验) | +1.9~2.5(总成69.28) | 多轮复验但**FRAGILE** | CONFIRMED_TARGET_SPECIFIC + TRANSFER_FRAGILITY |
| 2 | no-decay | +0.673 | 单用复验 | CONFIRMED_USEFUL(禁叠CWFA) |
| 3 | CWFA | +0.632 | 高复验 | CONFIRMED_USEFUL(训练天花板) |
| 4 | MixStyle α0.1 | +0.492 | α扫描复验 | DOSE_FAILURE(≤CWFA) |
| 5 | 旋转TTA8 | +0.470 | 高 | CONFIRMED_USEFUL(TTA层) |
| 6 | 伪标签(贴测试分布,stage55b) | +0.68 | 复验(stage56/62续) | CONFIRMED(翻案版) |
| 7 | 边界分支(单独) | +0.313 | 无TTA口径 | CONFIRMED_USEFUL(叠加即崩) |
| 8 | MixUp | +0.270 | 单次 | 小收益需复验 |
| 9 | DINOv2蒸馏 | +0.252 | λ扫描 | DOSE_FAILURE(价值有限) |
| 10 | MaxEnt λ1.0 | +0.40(vs CWFA) | 校准挖尽68.48 | 训练时自适应=复赛稳 |

## 2. local-online ranking reversal
- **B3 vs B2**: B3 local高于B2但online记录不全→UNKNOWN(疑RANK_REVERSAL)
- **Lovász**: local略涨 → online −0.415（区域损失对视觉不可分类毒药）
- **CopyPaste**: local 0.816最高 → online −0.311（学到粘贴伪影）
- **B4**: local 0.8174最高 → online仅66.951（同域验证奖励过拟合）
- **伪标签原版**: local涨 → −0.136（方向反了）
- **6796全数据(exp51)**: local 0.8182最高 → online −0.76（加同域验证图终极铁证）

## 3. Transfer Shrinkage
- **1024 vs 768**: local明显涨、online小涨——但记录不全，TE标UNKNOWN
- **SEMEDA**: 标准+1.056异常高但校准增益仅+1.28(最低)=标准高≠校准空间大
- **B4容量**: local 0.8174但online只+0.212

## 4. Negative Interaction（有充分历史证据）
| 组合 | A+B结果 | 机制 |
|---|---|---|
| CWFA + no-decay | −0.543 | 同类正则剂量翻倍 |
| 边界分支 + CWFA | −0.450 | 结构监督×特征扰动打架 |
| MaxEnt × 伪标签 | −0.27 | 都做分布对齐 |

## 5. 过去被过早判死的方法
- **TTA**: 曾记"全负"→实为复杂多尺度TTA负，简单翻转TTA +0.275/旋转TTA8 +0.47（**IMPL 翻案**）
- **伪标签**: 原版−0.136判死→方向反了，贴测试分布版+0.68（**翻案CONFIRMED**）
- **no-decay校准空间**: 老版V1e只到68.57被过早放弃→锁定向量69.2638（差0.0175）
- **MixStyle**: α0.1嫌小→实为α越弱越好但≤CWFA（收益见顶非判错）
- **CopyPaste**: 随机版判死→约束版未跑(翻案候选)
- **DINOv2蒸馏**: +0.25嫌小→λ扫描证明价值有限（判断正确非过早）

## 6. Implementation Failure（实现错非机制错）
- **复杂TTA/多尺度TTA**失败→简单D4/TTA8有效（最典型）
- CopyPaste随机版失败→约束版是不同实现
- 注意力头: 自研魔改(MCCA)失败，**标准FPN/ASPP从未试**（但仍高危，SegFormer头勿动）
- 高分辨率推理: 2048失败主因细路像素仅0.013%非真瓶颈（对症次要）

## 7. Dose Failure 候选
- **MixStyle**: α0.1过强→0.05略好仍≤CWFA（剂量错）
- **DINOv2蒸馏**: λ1.0过强→λ越小越好（剂量+前提错）
- **CWFA**: ε6有伤→ε4最优（剂量已找到顶点）
- **EMA**: decay0.999太钝≈没平滑（剂量错，0.99/0.995未试）
- **MaxEnt**: λ0.1太弱/λ5.0崩/λ1.0甜点（剂量扫描明确）

## 8. Stage Timing Failure 候选
- EMA锚定训练分布（时机/平滑钝）
- MaxEnt λ0.1太弱≈无MaxEnt（剂量弱，非时机）
- 伪标签过早/无conf过滤=确认偏差（弱模型时代）
- 多数失败属剂量/变体而非strict timing；无强timing证据

## 9. 证据足够差可降优先级（非滥用MECHANISM_REJECTED）
- **注意力融合头**: MECHANISM_REJECTED(2实现证据:魔改头与标准头都关闭,SegFormer头勿动纪律)
- **Focal**: MECHANISM_REJECTED(−0.342,难例=视觉不可分+噪声机制冲突)
- **Lovász/Dice**: MECHANISM_REJECTED(标注噪声前提不符+区域损失全负)
- **色彩增广**: MECHANISM_REJECTED(攻击不存在靶子:训练测试几乎同色)
- **B4/B5容量**: 结案(容量>数据硬约束,全路试过)
- **加权CE**: 降级(微负但prior-shift已解决,不放P0)

## 10. 小正收益不值得直接进未来冠军
- 蒸馏+MixStyle集成 +0.177（无协同）
- MixUp +0.270（单次）
- 边界分支 +0.313（叠加即崩）
- 这些**REQUIRES_REVALIDATION_BEFORE_STACKING**

## 11. 机制高度重叠方法
- **域泛化族**: CWFA/MixStyle/no-decay/色彩增广 → 都主攻style_robustness/domain_generalization → **HIGH_OVERLAP**(同类互斥)
- **区域损失族**: Lovász/Dice/Focal/RegionContrast → 都激进惩罚难分 → 全负（非重叠问题是机制冲突）
- **分布对齐族**: MaxEnt/伪标签(贴测试) → 都target_adaptation → 叠崩

## 12. 最互补两两组合候选（只列候选不训练）
- **伪标签(贴测试) + 8维校准(残差级)**: 训练期注入×测试时先验，正交（stage55b已证可合但校准微降→校准降为残差）
- **伪标签 + CWFA**: 目标注入×域泛化，不同层
- **旋转TTA8 + 校准**: 已证叠加+1.9（到顶）
- **SEMEDA + 伪标签**: 边界/嵌入×目标域
- **MixUp + CWFA**: 输入×特征
- **伪标签 + v3 rescue校准打标**: stage65正在做

## 关键规律汇总（机制地图结论）
1. 训练期方法天花板≈+0.67，无1+1>1
2. 校准=最大分来源但FRAGILE（压力测试7/7负迁移）
3. "同类不可叠、不同层可叠"但不保证协同
4. local涨≠有效（同域val奖励训练分布）
5. 训练时自适应(MaxEnt/伪标签)换分布稳=复赛解药
6. 超参第一枪=论文值一半；越弱越好规律
