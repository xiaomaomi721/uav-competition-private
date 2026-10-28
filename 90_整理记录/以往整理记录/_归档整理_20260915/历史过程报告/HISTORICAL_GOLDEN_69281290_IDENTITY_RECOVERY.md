# Golden 69.281290 历史身份追溯

## 结论分层

- **HISTORICALLY_RECORDED_ASSOCIATION_CLOSED**：本地历史成绩账本的build_p100条目，与服务器真实执行日志、生成配置、checkpoint SHA、ZIP SHA及500张预测文件形成对应链。
- **PLATFORM_INDEPENDENT_SCORE_VERIFICATION_UNAVAILABLE**：本次没有取得独立比赛平台原始评分回执。69.281290仍属于历史账本记录的线上分，不冒称平台重新验证。
- 本次发现补充了先前窄范围审计的UNKNOWN结论，但不修改冻结历史报告，也不把历史AMP推理等同于新FP32 matched-control。

## 历史权重与向量

Checkpoint：
`/home/jhy/deploy_exp18/exp18_b3_1024_full_ce_20260825T093709Z_bbec9e37a8f1/stage23_runs/stage23_w1_exp23_b3_1024_cwfa_ce_formal_seed42/best_miou.pt`

实际文件SHA256：`d3740b4ef73fbb3ac57ddeedc3547ce4ce1ba31442dc7c70e1fe96b1fab3e228`

向量（Background, Building, Road, Water, Barren, Vegetation, Agricultural, Vehicle）：

`[-0.2,0.1,0.0709,0.7,3.6,0.4,3.3215,-0.5]`

这与本轮称为FIXED8D_EXP41_STYLE的向量数值相同。但向量名称不能用来断定Exp41是独立训练模型或其checkpoint身份。

## 来源闭环

1. 本地`文档归档/04_实验决策与成绩/线上成绩记录表.md:255`：build_p100 / building+0.10 / 69.281290；第258行给出完整最终向量。
2. `/home/jhy/run_calib_road_build_scan.sh`：build_p100显式传完整向量。SHA `545e85824480ba29c7aeb039d2b5eef13fd2be342ed1e33aff97b5c8778fea72`。
3. `/home/jhy/calib_road_build.log:71255`：实际执行记录完整bias；71777–71779、81433–81434：包路径、DONE及SHA。
4. `/home/jhy/calib_build_p100/run_config.yaml`：绑定准确checkpoint路径、SHA及D4八视图。SHA `51058e97ca123bcec1fc9544d0eeb1235aa2aeb98461c330463c2398f43d6de3`。此文件没有保存bias，bias由脚本及实际执行日志补足。
5. `/home/jhy/calib_build_p100.zip`：本次重新哈希 `6be4e8b0b15aaa4a002bb99d1b1fe0677ec9fb67fe1ee32ac4e0fda5d763ee04`；根目录500个PNG与服务器同次prediction目录逐文件名、逐字节一致。
6. `/home/jhy/calib_build_p100/check_out/submission_check.json`：历史官方格式检查PASS；SHA `80c93584b00958207950e93a2cb6586de40a4103408c7b62b02c3ec4ec7bab3d`。格式PASS不是线上得分回执。

## 推理协议边界

对应`scripts/rot_tta.py`第138–142行说明显式prior-bias覆盖默认prior_scale，无隐藏×3；第175行按配置AMP，第185–188行逆变换对齐后mean logits，加一次bias，argmax后+1映射官方ID。

因此这份历史身份关联不等于新协议“无autocast真FP32 TTA8”的数值验证结果，不能据此填新matched val分数。

## 缓存结论

现有Golden `/home/jhy/gpu2_day_queue/tta_golden/mean_logits_fp16`只找到1182张，缺217张。历史16样本精度审计不等于1399验证覆盖。身份恢复不补出缺失logits；该core仍为QUEUE_TTA，不是RUN_NOW。
