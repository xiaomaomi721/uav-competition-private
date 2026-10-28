# HISTORICAL_CALIBRATION_FACTORY_DESIGN_V2

**设计记录，未部署、未启动。** 仅在Inventory明确存在RUN_NOW后实施；不修改已冻结的Paramwise协议。

## 复用边界

现有calibration.py只能复用经过核验的流式bias/argmax、哈希、等价去重等底层逻辑。历史factory需要独立入口、独立依赖冻结和独立结果目录，因为现有入口绑定Paramwise best/52/56、缺整图fold统计且敏感类别角色是固定的。

## 搜索阶段

1. zero / LOCKED8D / EXP41_STYLE三中心。
2. 8轴正负扰动；计划中的偏移和向量上限须在实际开跑前冻结，不能观察结果后放宽。
3. 实际高敏感轴局部联合。
4. 多种子完整邻域稳定性评估。
5. peak / 两个稳定盆地 / conservative / alternative / 两个实际敏感类别组合角色。

不做5^8穷举。读写预算需包括每个自适应阶段重新顺序读取已有缓存的成本，而不是声称CPU扫描没有成本。

## image-level稳定性

全模型共用固定seed42、按SHA256(seed,stem)排序并轮流分配的5fold manifest。每候选保存每image的8×8 int64 confusion。由整图confusion求和计算总体与fold指标，不平均单图mIoU、不随机分像素。

每个候选报告总体/各fold mIoU、8类IoU、预测占比、相对zero/LOCKED/EXP41各自delta、正delta fold数、最差fold与中位delta、邻域覆盖与变化。缺失类别明确报告支持度。

如启用固定seed paired image bootstrap，候选与基线共用相同抽样image索引；所有confidence均标描述性/选择后估计。fold与bootstrap不能宣称解决了同一验证集自适应搜索的选择偏差。

## 缓存信任

缓存必须绑定真实checkpoint SHA、规范1399 split、GT编码、producer source、预处理、D4/inverse/aggregation、原始计算精度、各文件清单及哈希。存储dtype=float32不证明原始forward为FP32。

可信历史FP16不一刀切排除，但必须有相应精度/用途闭环，且历史只测固定三个bias的精度检查不能自动证明所有新bias的FP32等价性。已明确FAIL或禁止下游使用的缓存不投入扫描。

## 资源与存储

- CUDA隐藏；nice19、ionice idle、串行一个模型、1–2线程。
- 流式逐图，禁止缓存所有image×candidate logits。
- 若133候选/1399image，每image int64 confusion原始体积约91MiB；实际向量预算决定最终量。
- JSON/CSV存摘要，数值使用压缩数组或可追踪分片；不在巨大JSON中反复改写全部矩阵。
- 不复制已有logits，不生成每候选PNG。
- 完整新1399 FP32 logits若以后允许GPU补算，原始需43.71875GiB/core；500 test需15.625GiB/core。当前10GiB级空间不够。
- test仅复用可信现有cache；没有则QUEUE_TTA。正式出包独立空间门禁、逐包验证、预测内容去重。

这份设计不等于已执行。RUN_NOW=0时输出明确“没有开始扫描”，不为占CPU而改用不可信缓存。
