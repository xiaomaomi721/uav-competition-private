# HISTORICAL_CALIBRATION_RECOVERY_RESULTS

**状态：BLOCKED_NO_RUN_NOW — 未启动扫描，不是扫描完成报告。**

- 可信完整val logits达到RUN_NOW条件的模型：0。
- 新CPU八维扫描：0；新GPU forward：0；新ZIP：0；线上提交：0。
- 缺logits只登记QUEUE_TTA；Pseudo继续RULE_BLOCKED；不扫描已明确FAIL缓存。

| candidate | checkpoint | zero-bias val | LOCKED8D val | EXP41 val | model-specific peak | stable basin score | fold stability | selected bias vectors | test logits availability | package readiness |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| Stage23_Golden_build_p100 | best_miou.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | NOT_IDENTIFIED_IN_INSPECTED_PATHS; PNG/ZIP are not logits | BLOCKED_MISSING_TRUSTED_LOGITS |
| Exp47_standalone_SEMEDA | best_miou.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | NOT_IDENTIFIED_IN_INSPECTED_PATHS | BLOCKED_MISSING_TRUSTED_LOGITS |
| Stage66_best51800 | best_miou.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | NOT_IDENTIFIED_IN_INSPECTED_PATHS; PNG/ZIP are not logits | BLOCKED_MISSING_TRUSTED_LOGITS |
| Stage66_checkpoint52000_C | checkpoint_step52000.pt | 0.8131277295176083 | 0.7875682049879659 | 0.7876702342447008 | NOT_RUN | NOT_RUN | NOT_RUN | [] | NOT_VERIFIED | BLOCKED_MISSING_TRUSTED_LOGITS |
| Stage66_checkpoint56000 | checkpoint_step56000.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | NOT_IDENTIFIED_IN_INSPECTED_PATHS; PNG/ZIP are not logits | BLOCKED_MISSING_TRUSTED_LOGITS |
| Exp45 | best_miou.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | UNKNOWN | BLOCKED_MISSING_TRUSTED_LOGITS |
| Exp34 | best_miou.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | UNKNOWN | BLOCKED_MISSING_TRUSTED_LOGITS |
| Exp31 | best_miou_clean.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | UNKNOWN | BLOCKED_MISSING_TRUSTED_LOGITS |
| Exp41_core | best_miou.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | UNKNOWN | BLOCKED_MISSING_TRUSTED_LOGITS |
| Exp49 | best_miou.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | UNKNOWN | BLOCKED_MISSING_TRUSTED_LOGITS |
| Stage55 | best_miou.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | UNKNOWN | RULE_BLOCKED |
| Stage55b | best_miou.pt | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | UNKNOWN / NOT_EVALUATED | NOT_RUN | NOT_RUN | NOT_RUN | [] | UNKNOWN | RULE_BLOCKED |

## 已有指标的严格边界

Stage66 step52000表中的三个分数来自旧TTA_ANATOMY.json记录的1399图pre-quantization真FP32 reference，不是本轮CPU重扫结果。现存落盘FP16缓存精度FAIL且禁止下游复用。旧记录可整理，不代表能从它恢复任意新bias预测。

其他模型不把single-view local、线上分或旧PNG结果填进新matched TTA8分数。没有峰值、稳定盆地、fold结果或新选中向量。

## 未执行的后续

统一factory的imagefold、敏感轴与邻域稳定性方案见HISTORICAL_CALIBRATION_FACTORY_DESIGN_V2.md。没有RUN_NOW就不部署扫描任务。未来获得可认证完整缓存后才能按用户授权运行；新GPU补TTA仍需满足本项目GPU许可和磁盘条件。

## 产物身份

完整checkpoint路径、SHA、证据来源和分类见HISTORICAL_CALIBRATION_RECOVERY_INVENTORY.md/json。现有历史固定预测包保留，但不能冒称本次model-specific候选包。
