# RAW partial: Exp01-47 服务器config精确证据（收集代理×3, 2026-09-09）

> 全部读取自服务器 run config.yaml + metrics.json + validation_history + sha256sum。不解释。
> 统一超参(全部config验证)：lr=6e-05(Exp35为2e-05), wd=0.01, AdamW, poly power1.0, bs2×ga2, amp, crop1024(Exp01为512,Exp02-13为768), seed42, rare_crop enabled prob0.5 priority[7,4,6] max_offset0.25。预训练=/home/jhy/models/segformer-bX-ade512。

## 关键纠正发现（vs 之前审计推断）
1. **Exp19: 不存在**(/home/jhy无任何*exp19*)；**Exp20: 仅smoke**(copypaste_smoke40), 无正式run
2. **Exp28 目录名=exp28_b3_1024_cwfa_norecay**(两方法组合CWFA+no_decay), 非"三方法"; 无单独三合一run
3. **Exp42/46 有失败attempt1**(BrokenPipe) + resume_v2才完成——之前只记最终
4. **Exp39 训练完成但final-eval崩**(DataLoader worker Terminated), best来自validation_history 0.811321@51800; 有best_miou_clean.pt
5. **Exp36/37超参扫描=max_steps 14000(10ep)** 非56000——绝对分低因步数
6. Exp42加权CE weights精确: clip[0.5,2.5], resolved=[0.5,0.5196,0.7483,0.8871,1.6717,0.5,0.8316,2.5]
7. Exp44 CE+Dice: ce_weight0.7 dice_weight0.3; Exp43 focal γ1.0; Exp41 edge_branch radius5 weight0.2; Exp47 semeda λ0.5 featsize256 gtnoise0.1

## Exp01-14 (deploy_stage05) — local_best_miou / sha
| exp | model | res | loss | 方法 | best_miou@step | sha前8 |
|---|---|---|---|---|---|---|
| 01 | B0 | 512 | CE | rare_crop(disabled) | 0.710122@53200 | 58713d19 |
| 02 | B0 | 768 | CE | rare disabled | 0.743154@54600 | c29df23d |
| 03 | B0 | 768 | CE | domain_aug(on,.5) | 0.741288@56000 | 62b7e9b6 |
| 04 | B0 | 768 | CE | rare_crop(on,.5) | 0.749709@56000 | 57c21675 |
| 05 | B1 | 768 | CE | rare | 0.769940@53200 | 4ee8748c |
| 06 | B2 | 768 | CE | rare | 0.787114@56000 | da942b75 |
| 07 | B3 | 768 | CE | rare | 0.796894@56000 | 707f336b |
| 08 | B2 | 768 | weighted CE(weights如上,clip.5-5.0) | rare | 0.778700@54600 | cd924ebd |
| 09 | B2 | 768 | CE | rare+light_blur(.2,.4-.8) | 0.787219@54600 | aa75de44 |
| 10 | B2 | 768 | CE | rare+ema(.999) | 0.787047@56000(raw>ema0.786823) | e7cbf65b |
| 11 | B2 | 768 | ce_lovasz(.8/.2) | rare | 0.789225@54600 | fb6e04d3 |
| 12 | B2 | 768 | CE | rare+light_blur(.2,.3-.8) | 0.783942@49000 | bf0d235d |
| 13 | B2 | 768 | CE | rare+light_blur(.1,.3-.8) | 0.782152@50400 | 09d7f62e |
| 14 | B2 | 1024 | CE | rare(+cwfa enabled=False) | 0.804564@51800 | 5e4bc1c2 |

## Exp15-47 补充(主力仓+deploy_exp16/17/24/25/26/27)
见 raw_master_exp15_25_partial.md + Exp26-47关键行:
| exp | model | 方法 | best_miou@step | sha |
|---|---|---|---|---|
| 26 B4纯CE | B4 | rare | 0.817376@53200 | 3f4fed23...bce |
| 27 B5纯CE | B5 | rare | 0.816936@53200 | 5807d763...561 |
| 28 CWFA+no_decay | B3 | rare+cwfa(p0.3,eps4,28001)+no_decay | 0.812250@56000 | 43ca85b3...dd1 |
| 29 B4+CWFA | B4 | rare+cwfa | 0.815372@53200 | 8010e02d...06a |
| 30 B4+no_decay | B4 | rare+no_decay | 0.816488@53200 | da94fdfb...e92 |
| 31 蒸馏 | B3 | rare+distill(λ1.0,start1,dinov2-large) | 0.812413@47600 | 54812df1...c2a |
| 32 B5+LoRA | B5 | rare+lora(r8,α16,drop0.1) | 0.726849@54600 | 6153c85c...555 |
| 33 CopyPaste | B3 | rare+copy(p0.5,ids[4,6,7],max3) | 0.815971@56000 | dd60d477...ff2 |
| 34 MixStyle | B3 | rare+mixstyle(p0.5,α0.1) | 0.809144@56000 | d19752b8...2d0 |
| 35 B4低lr+dp0.2 | B4 | rare+dp0.2, lr2e-5 | 0.794389@53200 | 9ba3f88f...da8 |
| 36 蒸馏扫描 | B3 | λ0.1/0.3/0.5, **14000步** | .77257/.77321/.77120@14000 | c829/ef8b/e5a3 |
| 37 MixStyle扫描 | B3 | α0.05/0.2/0.5, **14000步** | .76203/.76449/.76108 | 8565/e16c/2768 |
| 39 蒸馏+MixStyle | B3 | mixstyleα0.05+distillλ0.1, **56000步** | 0.811321@51800(vh;final-eval崩) | 651b5e67...dcb |
| 40 注意力头 | B3 | decode_head=attention_fusion | 0.805751@53200 | 0fac0092...15c |
| 41 边界分支 | B3 | edge_branch(r5,w0.2) | 0.811840@54600 | 98cb41c0...b3b |
| 42 weightedCE final | B3 | weighted(clip.5-2.5) | 0.810649@56000(v2) | 096ea38f...827 |
| 43 Focal | B3 | focal(γ1.0) | 0.810164@54600 | 35686d12...db2 |
| 44 CE+Dice | B3 | ce0.7+dice0.3 | 0.813167@53200 | c7b89b1b...df1 |
| 45 MixUp | B3 | mixup(α0.1,p0.5,start1) | 0.811281@51800 | 172d8512...34d |
| 46 边界+CWFA v2 | B3 | edge(r5,w0.2)+cwfa | 0.812938@53200 | 1b6d49e4...aa | 
| 47 SEMEDA | B3 | semeda(λ0.5,fs256,noise0.1) | 0.809569@56000 | 27e1441c...13e |

## 注: local_best = metrics.json best_miou(=mIoU=best ckpt全验证集评估); @N=best_step
