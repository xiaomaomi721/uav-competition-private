# H75 目标域类别bias校准｜2026-10-29实验记录

> **只记录已经执行的证据；截至此时本页全部新候选没有官方评分。** 以下计算由Claude Code/队友在既有H75缓存上完成，GPU使用为0（本轮EM、出现率、单轴和ZIP回放）；文件与SHA来自执行方回报，不代表本GitHub提交重新打开服务器复现。

## 1. 固定模型与推理协议

模型：单checkpoint **SegFormer-B3 H75**，SHA256 `b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a`。使用RGB/ImageNet normalize → D4-TTA8逆几何对齐 → 在**原生8×256×256 FP32 logits**上求平均 → 加统一固定类别bias → bilinear upsample到1024×1024（align_corners=False）→ argmax(内部0..7)+1 → L uint8 PNG。无多模型推理融合，无重新训练H75。

类别顺序：
`[Background, Building, Road, Water, Barren, Vegetation, Agricultural, Vehicle]`

| 偏置 | Background | Building | Road | Water | Barren | Vegetation | Agricultural | Vehicle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原Bstar | -0.2 | 0.1 | 0.0709 | 0.4 | 3.6 | 0.4 | 2.1215 | -0.5 |
| **PRESENCE_V1** | -0.2 | 0.1 | 0.0709 | **0.2** | **2.6** | 0.4 | **2.8215** | **-0.3** |
| **BARREN_2P6** | -0.2 | 0.1 | 0.0709 | 0.4 | **2.6** | 0.4 | 2.1215 | -0.5 |
| **BARREN_3P4（队友）** | -0.2 | 0.1 | 0.0709 | 0.4 | **3.4** | 0.4 | 2.1215 | -0.5 |
| **BARREN_3P8（队友）** | -0.2 | 0.1 | 0.0709 | 0.4 | **3.8** | 0.4 | 2.1215 | -0.5 |

## 2. 公开“出现图数”与使用许可

官网PDF公开各类出现图数，**不是像素先验**。Source6996真实GT按“图中至少1个该类像素”统计，含Ignore的9类数值与官网train列完全一致，确认了官网计数定义。官网Target3594 **没有逐像素GT**，只公开聚合出现图数。

| 类别 | 官方Target出现图数 | Bstar模型预测 | PRESENCE_V1预测 | BARREN_2P6预测 |
|---|---:|---:|---:|---:|
| Water | 1754 | 1825 | 1816 | 未提供 |
| Barren | 721 | 840 | **723** | **746** |
| Agricultural | 1281 | 1093 | 1139 | 1096 |
| Vehicle | 752 | 744 | 749 | 未提供 |

BARREN_2P6只改Barren，其他类的图像级预测出现张数也可能因像素改判受到影响；原始报告仅明确其荒地746/农田1096，**未提供Water/Vehicle重新统计值**，故不填。

> 重要：出现图数匹配不能确认模型选对了哪些图片，不能当作Target IoU。用户曾明确咨询组委会“能否使用官网聚合出现统计直接优化固定bias”，执行方收到回复“可以”；建议连同**完整提问原文和老师回复截图**保留，以便复现技术报告说明。旧问答文档中的第11题仅存问题原文，实际答复来自10月29日用户确认。

## 3. PRESENCE_V1实验设计

- Source1399固定按图拆成**FIT699 / VAL700**（seed20261029）；Source约束和各类别出现检测阈值只在FIT优化，VAL只报告。
- GT是否出现定义为该类在1024²图中至少1像素；模型出现判断为原生网格预测面积≥每类阈值τ，τ以FIT上的Zero bias预测出现F1选择。**模型的面积阈值是预测代理，不等于GT的“≥1像素”。**
- 只调整Water、Barren、Agricultural、Vehicle，`Δ ∈ [-1.5, +1.5]`（原始logit单位，步长0.1），从Bstar起步按荒地→农田→水体→车辆两轮坐标调整。
- 目标为4类预测出现数和官网目标数对数比的平方和；约束包括Target总改判≤3%、改判>20%的图≤1%、Source FIT整体mIoU不低于Bstar-1pp、逐类不低于-3pp、4类出现precision不低于Bstar-5pp。
- 查表加速替代逐候选完整argmax，小样本一致性复核通过；搜索耗时约13分钟CPU。
- 结果：荒地预测出现图数840→723，农业1093→1139（仍低于官方1281，进一步增大农业bias触发Source约束），水体1825→1816，车辆744→749。

## 4. 同协议Source VAL700结果

| 候选 | mIoU% | 相对Bstar | Barren IoU变化 | Agricultural IoU变化 | Target全像素改判 |
|---|---:|---:|---:|---:|---:|
| Bstar | 76.957 | — | — | — | — |
| PRESENCE_V1 | **77.306** | +0.349pp | **+4.88pp** | **-2.82pp** | **1.4353%** |
| BARREN_2P6 | **77.649** | +0.692pp | **+4.66pp** | **-0.22pp** | **0.393%** |
| 队友BARREN_3P4 | 77.098 | +0.141pp | 未提供 | 未提供 | 约0.082% |
| 队友BARREN_3P8 | 76.820 | -0.137pp | 未提供 | 未提供 | 约0.084% |

PRESENCE_V1的Background IoU +0.58pp；BARREN_2P6的Background +1.04pp，其他七类除明确列出的变动很小。**VAL700只是Source子集，不能与Source1399 mIoU直接比较，更不能外推Target3594。**

## 5. Target3594预测变化与结果包

全Target共`3594×1024×1024=3,768,582,144`个像素：

- **PRESENCE_V1** 相对已评分H75+Bstar ZIP：**54,090,300像素（1.4353%）**变化。主要流向Background→Agricultural 0.679%、Barren→Background 0.224%、Barren→Agricultural 0.196%；Target荒地预测面积约1.95%→1.49%，农田12.02%→13.09%。ZIP 30,802,765 bytes；SHA `f509a2e56b6840a09f19c726806373d3f936ecce1ed63e4ea4ced9cb24fcbdff`。Windows文件名：`H75_PRESENCE_V1_20261029.zip`。
- **BARREN_2P6** 相对已评分H75+Bstar ZIP：**14,816,205像素（约0.393%）**变化，1746/3594张完全不变；全部由Barren转为其他类：Barren→Background 0.248%、→Agricultural 0.097%、→Water 0.031%。ZIP 30,994,944 bytes；SHA `d797ebf8ab48d4aa738fe03b84ccf73b0754d77512f16427e03dc79c694df79b`。Windows文件名：`H75_BARREN_2P6_20261029.zip`。
- **队友BARREN_3P4**：约3,085,852像素（0.081884%）从Barren转向其他类；ZIP SHA `dd2e5a5001b9af8315975184fbbf2b6ac4d5d0d15f7860aa775befb9bf40e9d2`。
- **队友BARREN_3P8**：约3,170,767像素（0.084137%）由其他类转为Barren；ZIP SHA `1975606926af0d899d0c4b9cbd65fd383d6c8f515e1dc21f97c7938c052f7d0d`。

以上四个ZIP均执行方报告**3594 PNG、根目录平铺、文件名集合一致、1024²、灰度L、1..8、CRC/重复检查 PASS**；前两份已复制到用户Windows并校验本地/服务器SHA一致，**四份均尚未线上提交**。

队友在同一缓存回放原Bstar与已评分原ZIP逐像素比较时发现3130个像素不一致，占约8.31×10^-7，属于需如实保留的微小回放差异。因此各候选预测改判统计应写明参照是**原正式ZIP还是同缓存重建的Bstar**；本节列出的两份主候选比较采用已评分H75+Bstar ZIP。

## 6. 未通过的EM相对校准（不同分支，未出包）

- 旧EM先验代码审计发现：Source像素先验取自按类配额的像素reservoir（并非自然GT像素分布）、温度到原始logit的换算遗漏、Source/Target取样口径不一致。旧结果作废。
- 修复后温度`T=1.446`、固定采样、原始logit单位，构造`Bstar + T·[log π_T(3594)−log π_T(test1)]`候选（对比是描述性，不以test1恢复Bstar方向作为硬门槛）。
- 模拟测试A：Source无偏移最大误差0.172，**通过**；测试B：已知先验漂移6场景中barren_x0.5误差**0.236 > 0.20**，**失败**，其余5个在0.12–0.18；相对差分形式单独误差0.05，但整体B未通过，按预设规则NO-GO。
- EM候选Source1399：Bstar77.53→候选77.98（+0.45pp）；Target0.97%像素改判，主要转入Background；没有用于正式ZIP/官网提交。
- EM的Target Barren像素估计0.14%与官方Target Barren图像出现率20.1%**并不数学矛盾**，因为一个是像素份额、一个是出现图比例。可能存在模型漏检等解释，但尚未实证。

## 7. 实验归因与未来更新

- PRESENCE_V1与BARREN_2P6不仅差Agricultural，还同时差Water和Vehicle。**两者官方差分只能归因为三类联合附加bias的净效应，不能单独说“农田贡献了全部涨跌”。**
- Bstar的老test1调参背景与当前Target不同；更贴合官方聚合出现率不保证mIoU提高。
- 所有结果截至10月29日均**未在赛事平台评分**；一旦用户提供成绩，应更新成绩台账与本页状态，并关联具体提交包/时间/配置。
