# CPU 形态审计资料

仅分析已有预测，不运行模型推理。

- `CPU形态审计报告.md`：数值汇总与统计定义。
- `GT_B_ERROR_MORPHOLOGY.json`：Stage23 B 全部1399张验证图的真实错误统计。
- `TEST_ERROR_MORPHOLOGY.json`：B/s62/s58 三组500张test1零偏置预测分歧。
- `AUDIT_RECEIPT.json`：模型/验证集历史身份、脚本和结果SHA、历史指标复核。
- `决策解读.md`：本轮形态证据对后续方法优先级的含义与局限。

GT预测为单视图无偏置；测试预测为TTA8无偏置。测试分歧不代表真实错误。s62/s58全量验证预测未确认，本轮不包含它们的GT错误审计。

CPU执行：显式 `CUDA_VISIBLE_DEVICES=''`，线程限制为1，任务nice=15。没有修改正在运行的KD、冻结trainer或队列。

旧morph_v1结果不作为决策依据：GT匹配为0；其逐图平均连通域分位数/像素份额与新版全局池化定义不同，旧有向类对计数也不精确。
