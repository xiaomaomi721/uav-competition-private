# Quick 实验轻量报告

本目录是被 Git 忽略的 outputs/quick_7step/ 的仓库安全摘要，只保留可审查的 CSV、JSON 和 Markdown。

- experiment_summary.csv：11 组实验的扁平汇总
- experiment_summary.json：机器可读明细和逐类指标
- quick_7step_report.md：quick pilot 性质、结果、显存、速度与限制

生成方式：

    python scripts/summarize_experiments.py --output-dir outputs/quick_7step --report-dir reports/experiments

生成过程只读取已有实验结果，不训练模型。摘要中的权重位置是仓库相对引用；权重、训练日志、混淆矩阵二进制文件和可视化不会复制到本目录。
