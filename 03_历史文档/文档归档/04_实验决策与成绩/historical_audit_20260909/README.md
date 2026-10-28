# Historical Method Interaction Audit (2026-09-09)

冠军诊断完成后的历史实验复审+方法重分类+组合关系分析。纯文档审计，未训练。

## 结论速览
- 最可靠正收益: 校准(+2.5但FRAGILE)/no-decay(+0.673)/CWFA(+0.632)/MixStyle(+0.49)/旋转TTA8(+0.47)/伪标签贴测试(+0.68)
- 训练期天花板≈+0.67, 无1+1>1
- 3组已知negative interaction: CWFA+no-decay, 边界+CWFA, MaxEnt×伪标签
- 被过早判死: TTA(复杂vs简单)/伪标签(原版vs贴测试)
- Implementation failure: 复杂TTA; Dose failure: MixStyle/蒸馏/CWFAε;
- 校准=CONFIRMED_TARGET_SPECIFIC_GAIN + TRANSFER_FRAGILITY

## 文件
- master_experiment_ledger.csv — 29代表实验总账(校准百行扫描见线上成绩表)
- method_effect_ledger.csv — 27方法按9状态重分类
- local_online_mismatch.csv — 14条本地/线上不一致
- method_interaction_ledger.csv — 10组合
- interaction_failure_analysis.csv — 7组合失败机制分析
- revisit_candidate_methods.csv — 15个重审候选
- method_mechanism_matrix.csv — 16方法×13机制
- future_combination_candidate_matrix.csv — 14组合候选
- reliable_positive_methods.csv — 10可靠正收益排序
- small_gain_methods.csv — 6小收益需复验
- historical_method_audit_report.md — 12问完整回答

## 权威源
线上成绩记录表.md(549行online分) + 失败方案归因/根因复盘/方法集成/赛题本质复盘 + 跨会话记忆判决(成绩表未覆盖的09近期实验)
