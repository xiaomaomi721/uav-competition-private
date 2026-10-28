# 全部实验方案与线上成绩汇总

**整理日期：2026-09-14；最新正式训练状态快照18:56:07。**

本包整理的是可追溯记录，不把实验编号数当实际独立训练数量。core表含未找到编号占位及短跑/checkpoint变体；推理表与core会共享对照行，不能相加当提交次数。

## 先明确：八维在Stage23上搜索，其他模型经常迁移复用

**“校准后成绩”不等于“对该模型重新完整扫描八维后的最好成绩”。** 每行calibration_origin/model_specific_scan_verified和推理参数用于区分。Stage23冠军向量、旧LOCKED及中间版本不同；有证据的本模型局部扫描也明确记录，不把所有方案一概称未扫。

## A. 训练方案总表

| 编号 | 模型/尺寸 | 方法 | 训练总步数；best step | local mIoU(fraction) | 标准/关联线上分(%) | TTA/校准关联线上分(%) | 无分说明/状态 |
|---|---|---|---|---|---|---|---|
| Exp01 | B0/512 | CE / rare_crop(disabled) | 56000; best=53200 | 0.710122 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| Exp02 | B0/768 | CE / - | 56000; best=54600 | 0.743154 | 62.279098 | 暂无可确认绝对分 | NOT_APPLICABLE_PHASE2_EXACT_ONLINE_SUPPLEMENTS_RAW_LEDGER |
| Exp03 | B0/768 | CE / domain_aug(p0.5) | 56000; best=56000 | 0.741288 | 61.807964 | 暂无可确认绝对分 | NOT_APPLICABLE_PHASE2_EXACT_ONLINE_SUPPLEMENTS_RAW_LEDGER |
| Exp04 | B0/768 | CE / rare_crop(p0.5) | 56000; best=56000 | 0.749709 | 62.471861 | 暂无可确认绝对分 | NOT_APPLICABLE_PHASE2_EXACT_ONLINE_SUPPLEMENTS_RAW_LEDGER |
| Exp05 | B1/768 | CE / rare_crop | 56000; best=53200 | 0.769940 | 64.283137 | 暂无可确认绝对分 | NOT_APPLICABLE_PHASE2_EXACT_ONLINE_SUPPLEMENTS_RAW_LEDGER |
| Exp06 | B2/768 | CE / rare_crop | 56000; best=56000 | 0.787114 | 66.212776 | 暂无可确认绝对分 | NOT_APPLICABLE_PHASE2_EXACT_ONLINE_SUPPLEMENTS_RAW_LEDGER |
| Exp07 | B3/768 | CE / rare_crop | 56000; best=56000 | 0.796894 | 65.484737 | 暂无可确认绝对分 | NOT_APPLICABLE_PHASE2_EXACT_ONLINE_SUPPLEMENTS_RAW_LEDGER |
| Exp08 | B2/768 | weighted_CE / weights clip.5-5.0 | 56000; best=54600 | 0.778700 | 65.837945 | 暂无可确认绝对分 | NOT_APPLICABLE_PHASE2_EXACT_ONLINE_SUPPLEMENTS_RAW_LEDGER |
| Exp09 | B2/768 | CE / light_blur(p0.2,r.4-.8) | 56000; best=54600 | 0.787219 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| Exp10 | B2/768 | CE / ema(decay.999) | 56000; best=56000 | 0.787047 | 暂无可确认绝对分 | 暂无可确认绝对分 | 只记录相对跌幅，基线和绝对分不闭合；禁止反推 |
| Exp11 | B2/768 | ce_lovasz / ce.8/lovasz.2 | 56000; best=54600 | 0.789225 | 65.797297 | 暂无可确认绝对分 | NOT_APPLICABLE_PHASE2_EXACT_ONLINE_SUPPLEMENTS_RAW_LEDGER |
| Exp12 | B2/768 | CE / light_blur(p0.2,r.3-.8) | 56000; best=49000 | 0.783942 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| Exp13 | B2/768 | CE / light_blur(p0.1,r.3-.8) | 56000; best=50400 | 0.782152 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| Exp14 | B2/1024 | CE / rare_crop(cwfa off) | 56000; best=51800 | 0.804564 | 66.443 | 暂无可确认绝对分 | NOT_APPLICABLE_DOCUMENT_PRECISION_ONLY |
| Exp15 | B2/1024 | CE / rare_crop | 56000; best=56000 | 0.798495 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| Exp16 | B2/1024 | CE / rare_crop+upernet头 | 56000; best=54600 | 0.806580 | 65.9078 | 暂无可确认绝对分 | NOT_APPLICABLE_LATER_RESULT_SUPPLEMENTS_OLD_UNKNOWN |
| Exp17 | B2/1024 | CE / rare+cwfa(p0.3,eps4,28001) | 56000; best=56000 | 0.803976 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| Exp18 | B3/1024 | CE / rare_crop | 56000; best=47600 | 0.810525 | 66.739544 | 67.209674 / 68.032596 / 68.039691 | NOT_APPLICABLE |
| Exp19 | -/- | - / - | -; best=- | - | 暂无可确认绝对分 | 暂无可确认绝对分 | 服务器find无*exp19*；历史账本标为无独立线上提交，不代表当前全局状态 |
| Exp20 | B2/1024 | CE / copy_paste(smoke only) | -; best=- | UNKNOWN | 暂无可确认绝对分 | 暂无可确认绝对分 | 仅smoke无正式run；历史账本标为无独立线上提交，不代表当前全局状态 |
| Exp21a | B2/1024 | CE / rare+cwfa(eps6.0) | 56000; best=47600 | 0.803610 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| Exp22 | B2/1024 | CE / rare+pseudo | 56000; best=53200 | 0.799759 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| Exp23 | B3/1024 | CE / rare+cwfa(p0.3,eps4,28001) | 56000; best=56000 | 0.810986 | 67.371405 | 68.686044 / 69.28129 | NOT_APPLICABLE |
| Exp24 | B3/1024 | CE / rare+pseudo(500) | 56000; best=53200 | 0.813576 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| Exp25 | B3/1024 | CE / rare+no_decay | 56000; best=56000 | 0.810192 | 67.412 | 69.2638 | NOT_APPLICABLE |
| Exp26 | B4/1024 | CE / rare_crop | 56000; best=53200 | 0.817376 | 66.951058 | 68.030036 / 68.759823 | NOT_APPLICABLE |
| Exp27 | B5/1024 | CE / rare_crop | 56000; best=53200 | 0.816936 | 66.405239 | 67.200599 / 67.679726 | NOT_APPLICABLE |
| Exp28 | B3/1024 | CE / rare+cwfa+no_decay | 56000; best=56000 | 0.812250 | 66.195521 / 66.196 | 67.691 | NOT_APPLICABLE |
| Exp29 | B4/1024 | CE / rare+cwfa | 56000; best=53200 | 0.815372 | 66.001109 | 67.459865 | NOT_APPLICABLE |
| Exp30 | B4/1024 | CE / rare+no_decay | 56000; best=53200 | 0.816488 | 66.584083 | 7.926311 | NOT_APPLICABLE |
| Exp31 | B3/1024 | CE+distill / distill(λ1.0,start1,dinov2-large) | 56000; best=47600 | 0.812413 | 66.991 | 暂无可确认绝对分 | NOT_APPLICABLE |
| Exp32 | B5/1024 | CE+lora / lora(r8,α16,drop.1) | 56000; best=54600 | 0.726849 | 63.68103 | 63.159382 | NOT_APPLICABLE |
| Exp33 | B3/1024 | CE+copy_paste / copy(p0.5,ids[4,6,7],max3) | 56000; best=56000 | 0.815971 | 66.427993 | 68.775101 | NOT_APPLICABLE |
| Exp34 | B3/1024 | CE+mixstyle / mixstyle(p0.5,α0.1) | 56000; best=56000 | 0.809144 | 67.231 | 暂无可确认绝对分 | NOT_APPLICABLE |
| Exp35 | B4/1024 | CE / dp0.2, lr2e-5 | 56000; best=53200 | 0.794389 | 66.832587 | 67.972872 | NOT_APPLICABLE |
| Exp36-λ0.1 | B3/1024 | CE+distill / λ0.1, 14000步 | 14000; best=14000 | 0.772570 | 66.225255 | 66.736261 | NOT_APPLICABLE |
| Exp36-λ0.3 | B3/1024 | CE+distill / λ0.3, 14000步 | 14000; best=14000 | 0.773206 | 66.043135 | 66.922685 | NOT_APPLICABLE |
| Exp37-α0.05 | B3/1024 | CE+mixstyle / α0.05, 14000步 | 14000; best=14000 | 0.762026 | 66.243589 | 66.634921 | NOT_APPLICABLE |
| Exp37-α0.2 | B3/1024 | CE+mixstyle / α0.2, 14000步 | 14000; best=12600 | 0.764494 | 65.841038 | 66.522947 | NOT_APPLICABLE |
| Exp38_UNRESOLVED | UNKNOWN/UNKNOWN | UNKNOWN | UNKNOWN; best=UNKNOWN | UNKNOWN | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩 |
| Exp39 | B3/1024 | CE+distill+mixstyle / λ0.1+α0.05, 56000步 | 56000; best=51800 | 0.811321(vh@51800) | 66.915682 | 68.725136 | NOT_APPLICABLE |
| Exp40 | B3/1024 | CE / decode_head=attention_fusion | 56000; best=53200 | 0.805751 | 65.978999 | 67.40267 | NOT_APPLICABLE |
| Exp41 | B3/1024 | CE+edge_BCE / edge(r5,w0.2) | 56000; best=54600 | 0.811840 | 67.053 | 67.480463 / 68.79692 / 68.812 | NOT_APPLICABLE |
| Exp42 | B3/1024 | weighted_CE / clip.5-2.5,weights=[.5,.5196,.7483,.8871,1.6717,.5,.8316,2.5] | 56000; best=56000 | 0.810649 | 66.6902 | 68.303 | NOT_APPLICABLE |
| Exp43 | B3/1024 | focal / γ1.0 | 56000; best=54600 | 0.810164 | 66.397 | 67.9802 | NOT_APPLICABLE |
| Exp44 | B3/1024 | ce_dice / ce.7/dice.3 | 56000; best=53200 | 0.813167 | 66.5635 | 68.4128 | NOT_APPLICABLE |
| Exp45 | B3/1024 | CE+mixup / mixup(α.1,p.5,start1) | 56000; best=51800 | 0.811281 | 67.0098 | 69.1399 | NOT_APPLICABLE |
| Exp46 | B3/1024 | CE+edge+cwfa / edge(r5,w.2)+cwfa | 56000; best=53200 | 0.812938 | 66.2893 | 68.1674 | NOT_APPLICABLE |
| Exp47 | B3/1024 | CE+semeda / semeda(λ.5,fs256,noise.1) | 56000; best=56000 | 0.809569 | 67.796 | 69.0776 | NOT_APPLICABLE |
| stage48 | B3/1024 | CE+MaxEnt / λ0.1 | 56000; best=UNKNOWN | 0.811948 | 67.1308 | 暂无可确认绝对分 | NOT_APPLICABLE |
| stage49 | B3/1024 | CE+MaxEnt / λ1.0 | 56000; best=UNKNOWN | 0.809339 | 67.7711 | 暂无可确认绝对分 | NOT_APPLICABLE |
| stage50 | B3/1024 | CE+MaxEnt / λ5.0 | 56000; best=UNKNOWN | 0.809813 | 65.3961 | 暂无可确认绝对分 | NOT_APPLICABLE |
| exp51 | B3/1024 | CE / 全数据6796,67960步 | 67960; best=UNKNOWN | 0.818171 | 66.61 | 暂无可确认绝对分 | NOT_APPLICABLE |
| exp52 | B3/1024 | CE+MaxEntλ1 / 6796全数据 | 67960; best=UNKNOWN | 0.815931 | 66.2421 | 暂无可确认绝对分 | NOT_APPLICABLE |
| Exp53_UNRESOLVED | UNKNOWN/UNKNOWN | UNKNOWN | UNKNOWN; best=UNKNOWN | UNKNOWN | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩 |
| exp54 | B3/1024 | CE(锐化resume exp49) / 10000步,cwfa start5000 | 10000; best=10000 | 0.803651 | 66.8485 | 暂无可确认绝对分 | NOT_APPLICABLE |
| Stage55 | B3/1024 | Pseudo short14k | 14000; best=13592 | 0.7700222181356076 | 暂无可确认绝对分 | 暂无可确认绝对分 | 较新产物/身份审计已确认实验存在；所查资料未发现该正式版本线上成绩，不等同未提交或失败。 |
| stage55b | B3/1024 | CE / 5597真+500伪标签 | 56000; best=UNKNOWN | 0.809790 | 68.048 | 暂无可确认绝对分 | NOT_APPLICABLE |
| exp57 | B3/1024 | CE+RegionContrast / modeB w0.1 | 56000; best=UNKNOWN | 0.810375 | 67.5182 | 暂无可确认绝对分 | NOT_APPLICABLE |
| exp58 | B3/1024 | CE+RegionContrast / modeC pairs[[6,0],[1,0]] w0.1 | 56000; best=UNKNOWN | 0.811458 | 67.6376 | 暂无可确认绝对分 | NOT_APPLICABLE |
| Stage59 | B3/1024 | B3 targeted Region Contrast v2；pairs[[4,0],[6,0]] | 56000; best=UNKNOWN | UNKNOWN | 暂无可确认绝对分 | 暂无可确认绝对分 | 较新产物/身份审计已确认实验存在；所查资料未发现该正式版本线上成绩，不等同未提交或失败。 |
| Exp60_UNRESOLVED | UNKNOWN/UNKNOWN | UNKNOWN | UNKNOWN; best=UNKNOWN | UNKNOWN | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩 |
| exp61 | B3/1024 | CE+MaxEntλ1 / 5597真+500伪 | 56000; best=UNKNOWN | 0.805786 | 67.78 | 暂无可确认绝对分 | NOT_APPLICABLE |
| stage62 | B3/1024 | CE(续训) / resume stage55b last.pt→70000步 | 70000; best=68600 | 0.810394 | 暂无可确认绝对分 | 暂无可确认绝对分 | 指定账本未提供绝对线上分；未用相对涨跌反推绝对分 |
| stage63 | B3/1024 | CE / wd0.05 | 56000; best=56000 | 0.812205 | 67.52 | 暂无可确认绝对分 | NOT_APPLICABLE |
| stage64 | B3/1024 | CE / wd0.005 | 56000; best=53200 | 0.810903 | 67.61 | 暂无可确认绝对分 | NOT_APPLICABLE |
| stage65 | B3/1024 | CE / P1第二轮3教师,84000步 | 84000; best=- | None(训练中) | 暂无可确认绝对分 | 暂无可确认绝对分 | 历史账本2026-09-09记训练中、尚未提交；不是2026-09-14实时状态 |
| Stage66 | B3/1024 | SEMEDA0.5+RareCrop, delayed CWFA p0.3 eps4 start28001 | 56000; best=51800 | 0.8125200681968625 | 暂无可确认绝对分 | 暂无可确认绝对分 | 较新产物/身份审计已确认实验存在；所查资料未发现该正式版本线上成绩，不等同未提交或失败。 |
| LATE_CWFA_STAGED_C7 | UNKNOWN/UNKNOWN | 成熟Exp47 SEMEDA best初始化→Late-CWFA staged fine-tune | 8000计划总步数；本行checkpoint step7000; best=UNKNOWN | UNKNOWN | 暂无可确认绝对分 | 67.9246 / 69.0699 | 指定账本没有staged无TTA分或local分；不能从TTA结果推断 |
| LATE_CWFA_STAGED_C8 | UNKNOWN/UNKNOWN | 成熟Exp47 SEMEDA best初始化→Late-CWFA staged fine-tune | 8000计划总步数；本行checkpoint step8000; best=UNKNOWN | UNKNOWN | 暂无可确认绝对分 | 67.8534 / 69.121 | 指定账本没有staged无TTA分或local分；不能从TTA结果推断 |

## B. 最近GPT提出/执行的方案状态（无成绩可后补）

| 方案 | 类别 | 方法 | 状态 | 当前结果 | 线上成绩/批注 |
|---|---|---|---|---|---|
| Full6996 Stage23 Golden recipe全量refit | formal_training_and_packages | 公开预训练重新初始化；6996全数据；69997 outer steps；CWFA start35000；不是Stage23续训 | TRAINING_COMPLETE_SIX_PRIORITY_PACKAGES_READY | updates69971/skips26；last与69997权重内容一致；默认六包为65000/69997×TTA8/冠军向量/LOCKED | 未找到/尚未回填；不等于失败；尚未看到用户线上回填；55k/60k后处理未完成，不影响已完成六包事实 |
| Stage59 targeted Region Contrast v2 | formal_training_and_packages | 实际类对[[4,0],[6,0]]；B3/1024/CE/CWFA框架中的targeted Region Contrast | FORMAL_AND_HISTORICAL_PACKAGES_EXIST | 本轮汇总未找到可安全引用的最终local mIoU；本地20原包中含Stage59，UPLOAD_ORDER含去重候选 | 未找到/尚未回填；不等于失败；未找到该正式版本线上回填；不是失败结论 |
| Stage66 SEMEDA0.5 + delayed CWFA | formal_training_and_packages | SEMEDA λ0.5+RareCrop+CWFA p0.3 eps4 start28001；best51800/52000/56000独立权重 | FORMAL_COMPLETE_PACKAGES_EXIST | best训练记录0.812520；per-checkpoint 51800=0.812530,52000=0.812550,56000=0.812467；旧52000 FP32 D4 zero=0.813128 | 未找到/尚未回填；不等于失败；无明确线上回填；FP16 cache精度门禁FAIL是缓存工程失败，不是算法失败 |
| RareCrop 1024实际行为审计 | audit_only | 1000对实际样本检查crop1024空间行为 | AUDIT_PASS_RARECROP_SPATIALLY_INERT | 1000对bbox均为整图、pair tensors一致；触发515但accepted crop264仍不改变1024空间裁剪 | 未找到/尚未回填；不等于失败；审计，不是独立提交方案 |
| RareExposure v2 bounded exposure sampler | formal_training_running | Stage23其余不变；每epoch4197 uniform+700 Barren+700 Agricultural；无放回子流、每图上限3、seed42+epoch | RUNNING_AT_2026_09_14_18_56 | PID283448，step55111/56000；formal配置SHA未变；只保存best和last，不存在52k | 未找到/尚未回填；不等于失败；尚未完成/尚未生成matched三包/尚未提交；线上资格UNKNOWN |
| Stage23 Golden + canonical_paramwise_optimizer (Paramwise01) | formal_training_running | 原采样/增强/CWFA不变；backbone LR6e-5、head6e-4；实际norm WD0，其余WD0.01；641 trainable tensors完整覆盖 | RUNNING_AT_2026_09_14_18_56 | PID2115724，step8630/56000；formal配置SHA未变；40/44/48/52/56保存策略 | 未找到/尚未回填；不等于失败；尚未完成/尚未出包/尚未提交；不是winner |
| Stage23/Rare/Param matched-control协议 | postprocessing_protocol | 同FP32 D4 mean logits；ZERO/CHAMPION_8D_69_281290/LOCKED8D；checkpoint权重内容去重 | STREAMING_SCHEDULERS_DEPLOYED_WAITING_FORMALS | 容量CPU preflight PASS；测试ZIP适配官方checker已验证；生产GPU出包尚未运行 | 未找到/尚未回填；不等于失败；工程协议，不是比赛成绩；不会自动提交 |
| Paramwise persistent full-resolution FP32 cache protocol v2 | postprocessing_engineering | 三个val全缓存+selected test缓存+CPU8D scan；完整来源SHA门禁 | BLOCKED_STORAGE_SUPERSEDED_BY_STREAMING | required215605739520B≈200.80GiB；当时free≈10.34GiB；GPU_started=false,StageA NOT_STARTED | 未找到/尚未回填；不等于失败；容量架构阻塞，不是Paramwise算法失败 |
| STORAGE_AWARE_CALIBRATION_PROTOCOL_V1 | postprocessing_engineering | 每图FP32 TTA8 logits驻留内存，同时应用多个bias并ZipFile.writestr，随后释放；不保存完整logits/PNG目录 | DEPLOYED_FOR_RARE_AND_PARAM_MATCHED_PACKAGES | 动态联合门禁约5.51GiB，当前约10GiB PASS；低分辨率cache NOT_TESTED/NOT_ENABLED | 未找到/尚未回填；不等于失败；正式包需训练完成后才生成；当前无线上成绩 |
| LOW_RES_EXACT_CACHE审计提案 | optional_optimization_not_run | 审计canonical低分辨率logits中add bias/resize/argmax与现有全分辨率路径是否像素完全等价 | NOT_TESTED_NOT_ENABLED | 静态数学线性不足以证明FP32归约和argmax逐像素相等；未占用训练GPU跑32图审计 | 未找到/尚未回填；不等于失败；不是已执行方案；当前仍保留全分辨率FP32数学 |
| Historical Positive Calibration Recovery V2 inventory | audit_only | 审计Golden/Exp47/Stage66/MixUp/MixStyle/MaxEnt/DINO/Exp41及历史cache | INVENTORY_COMPLETE_RUN_NOW_0 | QUEUE_TTA10,RULE_BLOCKED2,SKIP3；Golden cache1182/1399；Stage66 52000 FP16完整但precision FAIL | 未找到/尚未回填；不等于失败；未启动新8D扫描；不把历史PNG/汇总mIoU当logits |
| POST_RARE two-pass Streaming Calibration Factory | local_engineering_not_deployed | PassA固定320整图/63向量；PassB全1399最多26向量+per-image confusion；最终test一遍多bias ZIP | LOCAL_FIXTURE_ONLY_NOT_REMOTE_DEPLOYED | factory synthetic CLI通过，10角色；生产preflight/strict model load/GPU未验证；queue静态复核发现2个缺陷已记录，生成本报告时尚未确认修复 | 未找到/尚未回填；不等于失败；不是完成的模型扫描或比赛成绩；远程部署被Auto权限拦截 |
| Post-Rare历史GPU2回收队列 | planned_not_deployed | Rare matched三包后：Golden→Exp47 standalone→Stage66-52000→Exp45；全部Streaming FP32 | LOCAL_QUEUE_ONLY_REMOTE_NOT_DEPLOYED | 真实1399 GT histogram与320 subset已冻结到本地；不执行Stage66其他权重/第二批/Pseudo | 未找到/尚未回填；不等于失败；计划未在服务器运行；Rare专属扫描必须等待真实线上matched结果 |
| Stage23 CHAMPION_8D_69_281290身份恢复 | historical_evidence_recovery | 闭合build_p100执行日志、Stage23 checkpoint、ZIP SHA和历史成绩账本 | HISTORICALLY_RECORDED_ASSOCIATION_CLOSED | vector[-0.2,0.1,0.0709,0.7,3.6,0.4,3.3215,-0.5]；ZIP SHA6be4e8... | 69.28129；历史项目账本分；没有独立平台原始评分回执。该八维在Stage23搜索，其他core多为迁移复用 |
| 磁盘安全清理计划 | operations_audit | 334510文件只读盘点；只删safe_to_delete=yes | STOP_CLEANUP_SAFE_ITEMS_0 | 删除0、回收0；保护所有formal/checkpoint/cache/log/manifest；约10GiB剩余 | 未找到/尚未回填；不等于失败；运维任务，不是算法方案 |
| Pseudo/Target Adaptation规则审计 | rules_audit | Stage55/Stage55b checkpoint和历史证据整理；检查test/transductive训练许可闭环 | RULE_BLOCKED | Stage55b历史core分68.048，但官方规则材料未明确允许test参与训练 | 68.048；历史分不解除规则阻塞；禁止自动生成/提交新包 |

## C. 近期实验、诊断与无线上成绩项目

| 方案 | 已做到什么 | 线上成绩 | 注意事项 |
|---|---|---|---|
| Phase1 HDBSCAN/Louvain group discovery | 不训练新分割模型；6996图分组诊断 | 未找到线上成绩记录；不填0 | 无独立比赛预测ZIP；产物为诊断分组与验收回执 |
| Phase2 group-aware reweighting historical calibration | 复用8个历史模型及其既有线上成绩；不训练新模型 | 未找到线上成绩记录；不填0 | 原1399固定val上的分组重加权诊断；无独立新线上提交或新比赛预测包 |
| Stage59 RegionC pairs [[4,0],[6,0]] (barren/agri -> background) | 历史last与s56000包路径已记录；本次未找到独立完成回执，不以文件名单独认证完整训练 | 未找到线上成绩记录；不填0 | 本地提交包/stage66_stage59_20260911/stage59_submission及autopack_stage59中10个ZIP实际存在（本次Glob确认）；协调方提供既往SHA核验完成信息，本轮未重hash。CANDIDATES的SHA null只属于旧盘点字段，不否定已下载 |
| Stage66 SEMEDA0.5 + RareCrop + delayed CWFA | S1审计已读取metrics.json/completed.json，存在独立best51800、52000、56000权重身份 | 未找到线上成绩记录；不填0 | 本地提交包/stage66_stage59_20260911/latecwfa_submission_stage66及autopack_stage66中10个ZIP实际存在（本次Glob确认）；协调方提供既往SHA核验完成信息，本轮未重hash |
| Full6996 refit | 69997终态包与last->69997别名已记录；65000优先包完成，不把69997改写成70000 | 未找到线上成绩记录；不填0 | 6个ZIP实际存在本地，本次逐个SHA256匹配原PASS报告；官方检查PASS来自历史回执，本次未重跑官方检查 |
| RareExposure v2 formal | 最后记录仍训练中，未见更晚本地formal PASS回执 | 未找到线上成绩记录；不填0 | 快照packages_started=false；尚无新matched包完成证据 |
| Paramwise01 formal UUID-fixed | 最后记录仍训练中，未见更晚本地formal PASS回执 | 未找到线上成绩记录；不填0 | 快照packages_started=false；尚无新matched包完成证据 |
| RareExposure smoke1/smoke2工程验收 | smoke1产物要求不一致；本地准备脚本记录错误at least 8 prediction visualizations，配置只产2。smoke2只把max_val_samples/prediction_visualization_count从2改8，后续formal运行快照证明已越过smoke入口 | 未找到线上成绩记录；不填0 | smoke是工程验收，不是正式500图候选包 |
| Paramwise protocol_v2全量缓存校准 | 非训练实验；容量预检不改变formal | 未找到线上成绩记录；不填0 | 该缓存方案未产生新生产包；后续由storage-aware streaming scheduler替代，旧控制器快照为T |
| 历史可信缓存恢复审计V2 | 不启动训练 | 未找到线上成绩记录；不填0 | scans_started=0,new_gpu_forwards=0,new_packages=0；不是恢复完成 |
| POST_RARE两遍验证/一遍test历史factory与observer | 不启动新训练；历史首批Golden->Exp47(CWFA off)->Stage66_52000->Exp45 | 未找到线上成绩记录；不填0 | CPU合成fixture为17候选10角色，非真实生产1399/500图结果；新生产ZIP未确认 |

## D. 推理/校准明细

逐行保留 197 条推理/校准/证据缺口记录，其中原始成绩表有分变体 174 条（部分行有标准版与校准版两列）。见03_推理校准全部成绩.csv以及证据/推理校准详细审计.md。

最重要的历史链：Exp18标准66.739544 → Exp18 D4 TTA-only67.209674；Exp23 CWFA标准67.371405 → TTA+迁移早期校准68.686044 → Stage23逐维搜索最终69.281290。这些不是全部严格同一因素阶梯，不能把差值盲目相加。

## E. 未有线上分与待补信息

- **Exp01**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **Exp09**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **Exp10**：只记录相对跌幅，基线和绝对分不闭合；禁止反推；TTA/校准关联：暂无可确认绝对分。
- **Exp12**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **Exp13**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **Exp15**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **Exp17**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **Exp19**：服务器find无*exp19*；历史账本标为无独立线上提交，不代表当前全局状态；TTA/校准关联：暂无可确认绝对分。
- **Exp20**：仅smoke无正式run；历史账本标为无独立线上提交，不代表当前全局状态；TTA/校准关联：暂无可确认绝对分。
- **Exp21a**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **Exp22**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **Exp24**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **Exp38_UNRESOLVED**：指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩；TTA/校准关联：暂无可确认绝对分。
- **Exp53_UNRESOLVED**：指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩；TTA/校准关联：暂无可确认绝对分。
- **Stage55**：较新产物/身份审计已确认实验存在；所查资料未发现该正式版本线上成绩，不等同未提交或失败。；TTA/校准关联：暂无可确认绝对分。
- **Stage59**：较新产物/身份审计已确认实验存在；所查资料未发现该正式版本线上成绩，不等同未提交或失败。；TTA/校准关联：暂无可确认绝对分。
- **Exp60_UNRESOLVED**：指定本地账本没有可独立绑定该编号的训练core行；不得凭编号或记忆补方法/成绩；TTA/校准关联：暂无可确认绝对分。
- **stage62**：指定账本未提供绝对线上分；未用相对涨跌反推绝对分；TTA/校准关联：暂无可确认绝对分。
- **stage65**：历史账本2026-09-09记训练中、尚未提交；不是2026-09-14实时状态；TTA/校准关联：暂无可确认绝对分。
- **Stage66**：较新产物/身份审计已确认实验存在；所查资料未发现该正式版本线上成绩，不等同未提交或失败。；TTA/校准关联：暂无可确认绝对分。
- **LATE_CWFA_STAGED_C7**：指定账本没有staged无TTA分或local分；不能从TTA结果推断；TTA/校准关联：67.9246 / 69.0699。
- **LATE_CWFA_STAGED_C8**：指定账本没有staged无TTA分或local分；不能从TTA结果推断；TTA/校准关联：67.8534 / 69.121。

- Stage59/Stage66/Full6996的固定包已存在但本材料未找到线上回填；可以由用户补充包名+分数，不能推定失败。
- RareExposure/Paramwise在最新快照仍训练中，online资格为UNKNOWN。
- stage65的“训练中”是旧账本状态，不是当前服务器实时核验。
- 编号Exp38/53/60以及Exp19缺独立记录不等于我们知道其实际做法；不要补造。

## F. 为分析AI保留的关键例外

- Exp16原账本UNKNOWN由2026-09-09较新报告补为65.9078；原文后半仍有待提交模板，不能让模板覆盖顶部回填。
- Exp55/59/66不在早期raw master表，但后续审计有身份/资产，已在本合并表补入，不再全局标NOT_FOUND。
- Exp31真实线上权重是clean checkpoint。Exp41旧standard包实际上TTA；仅true_standard_v2是无TTA标准分。
- Phase2早期score表虽然名calibration，其任务是验证指标重加权，不代表这些线上预测一定做过8D后处理；原始raw core协议也随包保留。
- 已发现raw派生calibration CSV漏掉加粗高分行；本次推理明细以原Markdown表重建，含build_p100、veg_p400等，不使用漏项表当全量真相。
- online最高来自历史项目账本，没有独立平台原始回执；这次未自动提交或向另一AI上传。

## G. 供另一AI建议的边界

请先读00_给另一个AI_先读说明.md的问题清单。不要建议立即停止/重启当前formal、混用online与local、把FP16精度失败当算法失败、把迁移Stage23偏置的失败当独立core否定、或无条件叠加多个正效益方法。
