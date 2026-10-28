# RAW EVIDENCE REPORT

> 纯事实收集。2026-09-09。来源=服务器run config/metrics/validation_history/sha256 + 线上成绩记录表.md + run_*.sh注释(线上分唯一载体) + 跨会话记忆判决。
> 不做方法分析。

## 找到什么（有正式来源）

### 实验config/metrics（服务器run目录实读，62实验入master ledger）
- **Exp01-14** (deploy_stage05/5be45d9)：config+metrics+best_miou sha256 全读。来源=服务器 run config.yaml+metrics.json。
- **Exp15-47** (deploy_exp16/17/18/24/25/26/27 + 主力仓)：config精确参数(cwfa/mixstyle/distill/copy_paste/semeda/edge/lora/ema)+best_miou+sha256。来源=服务器。
- **Exp48-65** (主力仓 stage48-65)：含MaxEnt三档/6796全数据/锐化/pseudo/region/wd扫描/P1。来源=run_manifest.json+config。
- **统一超参确认**(全config验证)：lr6e-05(Exp35=2e-05), wd0.01(Exp63=0.05,Exp64=0.005), AdamW, poly1.0, bs2×ga2, seed42, rare_crop p0.5 ids[7,4,6], crop1024(Exp01=512,早期=768), max_steps56000(Exp36/37扫描=14000,exp51/52=67960,stage65=84000)。

### 绝对线上分（来源层级）
- **成绩记录表.md**(153行含绝对分,最权威online) → raw_online_submission_ledger.csv
- **run_*.sh注释**(线上分回填载体,如run_calibv2.sh锁定向量,run_exp23_submit.sh 66.739/67.371/68.686) → 冠军链
- **stage06_handoff ONLINE_SCORES**(Exp05=64.283137/Exp06=66.212776/Exp07=65.484737,唯一ONLINE_SCORES文件)
- **记忆判决**(成绩表未覆盖的09近期): exp49=67.77, stage55b=68.0480, Exp28=66.195521等

### 冠军69.281290分解链（绝对分）
| 级 | 绝对分 | 来源 |
|---|---|---|
| BASE B3纯CE(Exp18) | 66.739544 | 成绩表+run_exp18注释 |
| +CWFA(Exp23标准) | 67.371405 | 成绩表+run_exp23注释 |
| +旋转TTA8(无校准) | 67.209674 | 成绩表(Exp18+rotTTA8) |
| +rotTTA8+V1e校准 | 68.032596(Exp18) / 68.686044(Exp23) | 成绩表 |
| **锁定8维(B3+CWFA+TTA8+校准)** | **69.281290** | 成绩表2.11(build_p100) |
- flip TTA独立提交在Exp18验证(非Exp23)；冠军推理=Exp23 ckpt(d3740b4e)+rotta8+8维校准。**无flip TTA在Exp23上的独立提交记录**。

### 8维校准锁定向量（源码级证据）
- run_calibv2.sh 注释: `-0.2000 -0.0384 0.0709 0.7000 3.6000 0.4000 3.3215 -0.5000` 标注"锁定八维(B3+CWFA最优)"。类别顺序=背景/建筑/道路/水体/荒地/植被/农田/车辆(注释)。
- 应用: logits加法(rot_tta.py prior-shift), TTA后, prior_scale=3.0。来源=rot_tta.py+run_calib*.sh。
- **搜索方式/总次数**: 无独立文件记录总提交数; 成绩表逐行可数(≥111校准行)。**UNKNOWN字段见gaps**。

### 提交zip（服务器实存）
- stage18_submission_standard/prior(.zip), exp23_submission_standard/rotta8_calib(.zip,含V1e=0,-0.0384,0.0709,1.0692,1.1048,0.1932,3.3215,-0.6960), flip/rotTTA目录族, exp23calib_* 73个校准zip。来源=/home/jhy ls -la。

## 哪些数字仍UNKNOWN
1. 大量早期实验(Exp01-22)线上绝对分：成绩表未收录,无ONLINE_SCORES文件(除Exp05-07) → **UNKNOWN**(Exp01-04/08-22线上分无记录)
2. 冠军8维校准"总提交/评估次数"、每步搜索方法严格日志：无独立文件
3. 每个zip的严格提交→分数一一对应：线上分在.sh注释维护,非结构化文件
4. 各checkpoint sha256：config_sha256有,但best_miou.pt完整sha仅部分实验读了
5. heldout指标：无记录
6. 早期实验(Exp15-22)在线提交目录：部分有zip但分数未回填
7. Exp19: 确认不存在; Exp20: 仅smoke。→ 无线上分

## 每条数字来源文件索引
| 数据 | 文件 |
|---|---|
| 绝对线上分 | 文档归档/04_实验决策与成绩/线上成绩记录表.md(153行) |
| 锁定向量 | 服务器 /home/jhy/run_calibv2.sh, run_calib_exp25v2.sh, run_veg_scan.sh 注释 |
| 冠军提交zip | /home/jhy/stage18_submission*, exp23_submission*, exp23calib_*(73), fliptta_b3_*, rotta_b3_*, hflip_b3_* |
| 训练config/metrics/sha | /home/jhy/deploy_*/stage*_runs/*/config.yaml+metrics.json+sha256sum |
| Exp48-65 | /home/jhy/deploy_exp18/.../stage48-65_runs/*/run_manifest.json+config |
| Exp01-14 | /home/jhy/deploy_stage05/5be45d9/stage05-14_runs/ |
| 记忆判决 | C:/Users/.../memory/*.md |
| 提交log(无线上分) | /home/jhy/submit*.log (仅自检,不含分数) |
