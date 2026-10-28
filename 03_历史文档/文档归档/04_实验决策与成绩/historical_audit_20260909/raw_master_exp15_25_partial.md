# RAW partial: Exp15-25 服务器config精确证据（收集代理af1dfe79af710ba2f 2026-09-09）

> 全部读取自服务器 run config.yaml + metrics.json + sha256sum(best_miou.pt)。不解释。
> 统一超参(所有行config验证)：lr=6e-05, wd=0.01, AdamW, poly power1.0, bs2×accum2, amp, planned_epochs=40, steps/epoch1400, max_steps=56000, seed42, crop1024, rare_crop p0.5 ids[7,4,6], domain_aug disabled, flip/jitter p0.5。预训练=/home/jhy/models/segformer-bX-ade512。

## Exp15-25 正式run
| exp | 目录 | model | res | loss | 方法 | local_best_miou@step | sha256 |
|---|---|---|---|---|---|---|---|
| 15 | deploy_stage05/stage15 exp15_b2_mixed1024_768rare | B2 | 1024 | CE | rare_crop | 0.7984945996175357@56000 | a54a4656af41a76a585d3fb05174af1ed9570bd0d52933da2ab4322700b935f0 |
| 16 | deploy_exp16 exp16_b2_1024_upernet | B2 | 1024 | CE | rare_crop p0.5[7,4,6] | 0.80657990922721@54600 | f3cd64e5d7435540ed75f6d631212633090d7880c626ddd6f7436dfd8787bc20 |
| 17 | deploy_exp17 exp17_b2_1024_late_cwfa_a | B2 | 1024 | CE | rare_crop+cwfa(p0.3,eps4,28001) | 0.8039760133131967@56000 | 01b13bceebf4f9de2a54ae6ae3cc9c91fabe63560a407c676cd1ce1a71b89fcb |
| 18 | deploy_exp18 stage18 exp18_b3_1024_full_ce | **B3** | 1024 | CE | rare_crop | **0.8105253202207898@47600** | 66667268d824d559a68b4b80872fe64d729bed7c6361d1a9939d6a3f92de88fd |
| 21a | deploy_p1p2 exp21a_b2_1024_cwfa_eps6 | B2 | 1024 | CE | rare_crop+cwfa(eps6.0) | 0.8036103018862699@47600 | 2172cf55c7f9a4e5717088cdfe7ef465609dd0b5e7a5a0466b45d933af46c2ca |
| 22 | deploy_p1p2 exp22_b2_1024_pseudo | B2 | 1024 | CE | rare_crop+pseudo | 0.7997586929397491@53200 | bea38471ee2b73ed3de13e304a3ae96b4654d4d7cca1a0274027737aaefa5d64 |
| 23 | deploy_exp18 stage23 exp23_b3_1024_cwfa | **B3** | 1024 | CE | rare_crop+cwfa(p0.3,eps4,28001) | **0.8109861457370224@56000** | d3740b4ef73fbb3ac57ddeedc3547ce4ce1ba31442dc7c70e1fe96b1fab3e228 |
| 24 | deploy_exp24 exp24_b3_1024_pseudo | B3 | 1024 | CE | rare_crop+pseudo(500) | 0.8135763744309306@53200 | 4b935e0961bfd47a8d12c376ef60a0af3cb6bc517df91fb1aaf09a7727ea66dc |
| 25 | deploy_exp25 exp25_b3_1024_norecay | B3 | 1024 | CE | rare_crop+no_decay | 0.8101924634092424@56000 | e76f1fb254627483894f50da9f414ec299b6a3e6856f05c520443f3c9104e140 |

## NOT_FOUND
- **Exp19**: /home/jhy 下无任何 *exp19* 文件 → 不存在
- **Exp20**: 仅 smoke(stage20_w1_exp20_b2_1024_copypaste_smoke40), 无正式run

## 5仓库内额外formal run（Exp28起主力仓）
| exp | 目录 | model | loss | 方法 | local_best@step | sha[:16] |
|---|---|---|---|---|---|---|
| 28 | exp28_b3_1024_cwfa_norecay | B3 | CE | rare+cwfa+no_decay | 0.8122497900835162@56000 | 43ca85b3522c3780 |
| 30 | exp30_b4_1024_norecay | B4 | CE | rare+no_decay | 0.8164878854502982@53200 | da94fdfb733d88dc |
| 31 | exp31_b3_1024_dino2distill | B3 | CE | rare+distill(λ1.0,start1,dinov2-large) | 0.8124134723397496@47600 | best 54812df1699afef3; clean bad30e00e7dd0b0e |
| 33 | exp33_b3_1024_copypaste | B3 | CE | rare+copy_paste(p0.5,rare_ids[4,6,7]) | 0.8159709266196946@56000 | dd60d47791555654 |
| 34 | exp34_b3_1024_mixstyle | B3 | CE | rare+mixstyle(p0.5,α0.1) | 0.8091444104209777@56000 | d19752b8f0cfdb7d |
| 40 | exp40_b3_attnfusion | B3 | CE | rare(attn头) | 0.805751213206212@53200 | 0fac0092d7df01f3 |
| 41 | exp41_b3_edge_branch_bce | B3 | CE+BCE | rare | 0.8118399199741564@54600 | 98cb41c0f40e75df |
| 42 | exp42_b3_weighted_ce | B3 | weighted(weights [0.5,0.5196,0.7483,0.8871,1.6717,0.5,0.8316,2.5]) | rare | (无metrics.json) | a0b95e73f81767ea |
| 42v2 | ...resume_v2 | B3 | weighted | rare | 0.8106487356408772@56000 | 096ea38f7ad1ef52 |
| 43 | exp43_b3_focal | B3 | focal(γ1.0) | rare | 0.810164138561673@54600 | 35686d12a6afc485 |
| 44 | exp44_b3_cedice | B3 | ce_dice(ce0.7,dice0.3) | rare | 0.8131669204493128@53200 | c7b89b1b3d2b723a |
| 45 | exp45_b3_mixup | B3 | CE | rare | 0.8112808706598829@51800 | 172d8512ae140662 |
| 46v2 | exp46_b3_edge_cwfa_v2 | B3 | CE | rare+cwfa | 0.8129378851031946@53200 | 1b6d49e46fadbcd7 |
| 47 | exp47_b3_semeda | B3 | semeda | rare | 0.809568813669471@56000 | 27e1441c008ab1b9 |

## 冠军链关键事实（Exp18 vs Exp23）
- Exp18 与 Exp23 同基座(B3/1024/CE/lr6e-5/wd0.01/56000/seed42/rare p0.5[7,4,6])。Exp23 仅加 cwfa(enabled,p0.3,eps4.0,start28001)。
- Exp18 local best 0.8105253202207898@47600; Exp23 0.8109861457370224@56000 → local delta CWFA=+0.00046
- Exp18 submission确认 step=47600 best=0.8105253 sha 66667268...88fd
- Exp23 submission确认 step=56000 best=0.8109861 sha d3740b4e...e228

## TTA脚本/提交档案
- flip_tta.py: 4视图(orig/hflip/vflip/hflip+vflip), 含 PRIOR_LOG_BIAS=[0.0119,-0.0128,0.0197,0.2970,0.2455,-0.0356,0.7381,-0.2320]
- rot_tta.py: 8视图(4rot×2mirror=D4); rot_tta.py.before_edge=旧变体
- exp23_submission_rotta8_calib: Exp23 ckpt(d3740b4e), n_views=8, prior_shift=true prior_scale3.0, 500预测
- exp23_submission_standard: 同ckpt, TTA:false
- stage18_submission_exp18_b3_standard/prior: Exp18 ckpt(66667268)
- **flip-TTA输出均来自Exp18(非Exp23)**: fliptta_b3_hv_standard/prior3.0, fliptta_b3_prior2.5/3.5/4.0, hflip_b3_standard/prior3.0 (各run_config记 checkpoint_sha256=66667268...88fd)
- **冠军推理=Exp23 ckpt + rotta8 + prior-shift/校准**; flip-TTA仅在Exp18验证
- 其他 exp*_submission_rotta8calib 覆盖 Exp24/25/28/29/30/31/33/34/35/39/40
- exp23calib_*/run_calib_* = 8维类bias校准扫描 on Exp23 ckpt
