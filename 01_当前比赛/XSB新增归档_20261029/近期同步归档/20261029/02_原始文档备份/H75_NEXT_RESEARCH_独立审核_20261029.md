# H75_NEXT_RESEARCH_20261029 独立审核与下一步决策

日期：2026-10-29。依据：用户上传的原始ZIP内JSON/CSV/脚本，以及Model Soups、BBSE、MIC、HRDA、DACS原始论文摘要及官方页面。本文件的数值由 `/mnt/data/verify_h75_next_package.py` 从ZIP中的两套完整Source混淆矩阵和Target统计复算；**未连接用户Linux服务器，未重新推理3594张图片或重新读取权重**。因此，“独立复算交付数据”不等于“独立验证产生交付数据的服务器过程”。

## 结论分级

- **可保留**：Source真实GT的全量类别占比；Zero/Bstar的Source1399混淆矩阵与逐类表现；Target预测分布表（作为提交材料中自报的计算结果，尚需检查缓存身份）；CONTROL/TREAT参数键及形状兼容、400张Source上的五点插值成绩（均来自可检查的脚本/结果）。
- **不能据此下结论**：70.3与70.13指定组合失败；所有Model Soups均无意义；Target荒地“真实先验亏缺”；只换Source先验就能解决BBSE；Source更高必定Target更低；MIC、HRDA和DACS在本赛题均无潜力。

## 1. 历史70.3 / 70.13没有完成用户指定的组合验证

`01_复赛两模型融合/70_3与70_13模型身份核查.md`只有70.272000（BlindRecall CONTROL）的候选身份，未查到70.13；实际插值对象却是CONTROL 70.272000与TREAT 70.867100。两个分数不等于用户点名的70.3和70.13。

CONTROL/TREAT插值Source R1099前400：α=[0,.25,.5,.75,1]，mIoU=[80.4839,80.3007,80.0824,79.8544,79.6026]。这个结果仅拒绝**这对模型、这个Source子集、这个Bstar推理协议、这五个α**的中间最优假设。它不能代表Target，也不能用来拒绝尚未确定身份的70.13模型。

即使stage0/1的166个张量逐位相同、stage2/3相对L2仅0.4–0.9%，这只说明所比较模型的参数结构接近，不能数学推导插值后准确率必然位于两端之间；H75本身历史上即由参数平均形成。判决“所有模型天然没有融合上限”是错误泛化。

`FUSION_PROBE.json`还可重建“互补”的方向，而不能只报XOR：CONTROL相对TREAT独有正确像素约3,894,306，TREAT相对CONTROL独有正确像素约2,106,758，TREAT净少1,787,548正确像素。说明存在两个方向的互补，但在这个Source测试中TREAT付出的错误更多。**并非“完全没有互补性”。**

## 2. 目标荒地预测比例不能代替真实标签先验

`R1_REVERIFY.json`显示Target的Zero预测荒地为test1 0.6912%、test2 0.7084%、test3_new 0.7407%；Bstar预测为1.8435%、1.8577%、2.0473%。这都是**模型的预测占比**，Target没有真实GT。

因此，报告写“Bstar补的是真实先验亏缺（VERIFIED）”不成立。可能是Target的真实类分布改变，也可能是同类外观改变导致模型漏检，或二者同时发生。Bstar提高模型预测最大置信度不能证明输出更正确，也不能区分两种机制。

另：`r1_reverify.py`在输出JSON时直接写死`bias_applied_in_cache=False`，没有读取CACHE_META、校验该字段和对应生产脚本/文件SHA。已有说明和文件名可能支持“raw缓存”这一前提，但**本ZIP脚本本身没有完成缓存身份校验**。

## 3. 可以立即完成的类别重加权模拟

使用交付包`R0_A3_BSTAR_EFFECT.json`中两套8×8真实GT混淆矩阵，将每一类真实GT条件下的预测分布归一化为`P_source(pred|GT)`，再用不同假设GT类先验对行进行加权、重算mIoU：

| 重加权假设 | Zero mIoU | Bstar mIoU | Bstar−Zero |
|---|---:|---:|---:|
| Source1399原始分布 | 80.4418 | 77.5311 | −2.9107 pp |
| Source训练5597真实GT比例 | 80.2018 | 76.9542 | −3.2476 pp |
| Target Zero的预测比例（仅做代理） | 77.6599 | 73.4571 | −4.2028 pp |
| Target Bstar的预测比例（仅做代理） | 79.9055 | 77.3646 | −2.5409 pp |

这里后两项不是Target真实IoU，因为目标真实先验未知，而且还假定Source的类条件混淆矩阵可以迁移到Target。它们**不能完全否定标签偏移假说**，但已经证明Claude所建议的几个具体“只改变类别比例”实验不能合理地解释“Target Bstar比Zero高3.0191pp”。

更进一步，若从Source-val的8×8混淆矩阵重加权到任意目标GT类别分布，应先对Source-val各GT行归一化，再乘目标先验；不能将`p_target/p_source_train`直接乘Source-val原始混淆矩阵，否则混入val/train类先验差异。

## 4. 新报告内部的可定位错误

- `H75错误机制研究.md`第1.3节写Source `Background→Agricultural=4,674,174`、`Background→Barren=1,549,034`；但同包原始`R0_A3_BSTAR_EFFECT.json`及`Bstar纠错与致错分析.csv`确实分别为**21,750,444**和**25,545,301**。这不是四舍五入，需回源检查错误表生成过程。
- `H75错误机制研究.md`把test3_new的Bstar荒地写成约1.86%，与`R1_REVERIFY.json`的**2.0473%**不一致。其“Bstar变化幅度三个子集<0.5pp”也不完全成立：Background的变化分别为−4.7083pp和−5.2486pp，相差0.5403pp。
- “Source分数越高→Target越低、五点完全单调”是**选择性样本现象**：加入已知的M2F E02 B1（Source79.5654、线上67.6237），就破坏了简单的单调排序。不同架构、推理协议也会混杂这个关系。
- Source/Target RGB亮度和清晰度均值的差异，不能直接推断`P(x|y)`的差异：没有Target GT，不能按真实类别比较其外观；也不能说这些改变“量级小”就不会影响分类准确率。
- 源端统一官方标签映射，不等于不同数据来源不存在标注定义差异。
- MIC报告写“仅在未遮挡区计算一致性损失”，与MIC原论文强调从上下文恢复**被遮挡区域**的语义不一致，必须核对正式loss实现。
- HRDA不要求官方原图尺寸必须大于1024；可以使用缩小全局图像与原生分辨率局部patch学习尺度注意力。是否值得应用需另做工程/效益判断，不能以“1024原图”直接判不可用。
- DACS来源被误写为ICCV 2021并链接无效；正确是Tranheden等人**WACV 2021**。

## 5. 推荐的低成本验证顺序（不授权新训练/不授权提交）

1. **证据核对，先不做方案筛选**：对R1中的Target原始缓存路径核对`CACHE_META`、模型SHA、preprocessing、D4视图、score语义，最重要的是随机取少量图与已评分H75-Zero/Bstar输出逐像素复现，确保报告中的raw/加bias标签确实对应已评分方案。
2. **纠正报告与历史定位**：修复Source转换矩阵错读、test3数值错读，并搜索70.3和70.13真实提交文件身份。仍缺身份时不做指定两模型融合结论。
3. **对域偏移做可证伪分解**：在Source独立外观子组，检验Bstar纠正/致错方向及`P(pred|GT)`是否随外观分组改变；检查预测分布是否能在真实划分条件下迁移。Target只作无标注特征比较，不把预测当GT。
4. **选择具体一个小实验**：仅在出现可靠、能与官方线上差异一致的证据时，才比较修正的BBSE/EM、真正同构的两模型参数插值或监督门控变化。Source变化不自动代表线上收益。
5. **保护正式提交链**：H75已评分ZIP、模型SHA、Docker及报告优先，不为未确认可行性的研究动用提交额度。

## 6. 来源

上传包本地相对路径：`01_复赛两模型融合/{FUSION_PROBE.json,PAIR_COMPAT.json,融合可行性结论.md}`，`02_H75错误机制/{R0_A3_BSTAR_EFFECT.json,R1_REVERIFY.json,H75错误机制研究.md}`，`04_最终报告/下一阶段深度研究报告.md`；独立数值复现脚本：`verify_h75_next_package.py`。

论文官方链接：
- Model Soups，ICML 2022 https://proceedings.mlr.press/v162/wortsman22a.html
- Label Shift / BBSE，ICML 2018 https://proceedings.mlr.press/v80/lipton18a.html
- MIC，CVPR 2023 https://openaccess.thecvf.com/content/CVPR2023/html/Hoyer_MIC_Masked_Image_Consistency_for_Context-Enhanced_Domain_Adaptation_CVPR_2023_paper.html
- HRDA，ECCV 2022 https://arxiv.org/abs/2204.13132
- DACS，WACV 2021 https://openaccess.thecvf.com/content/WACV2021/html/Tranheden_DACS_Domain_Adaptation_via_Cross-Domain_Mixed_Sampling_WACV_2021_paper.html