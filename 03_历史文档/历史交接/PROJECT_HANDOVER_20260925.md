# AIC2026 UAV语义分割 — 项目完整交接 (2026-09-25)

## 1. 比赛信息
- 8类语义分割，1024x1024输入，输出PNG标签1-8
- 类别: Background(0), Building(1), Road(2), Water(3), Barren(4), Vegetation(5), Agricultural(6), Vehicle(7)
- SegFormer-B3 backbone
- 推理协议: D4 TTA8 -> FP32 view mean at 256x256 -> +Bstar -> bilinear upsample 1024 -> argmax -> +1 -> PNG
- Bstar: [-0.2, 0.1, 0.0709, 0.4, 3.6, 0.4, 2.1215, -0.5]
- 每天2个提交槽，今天还剩1个，约4小时

## 2. 线上成绩（全部已提交）

| 方案 | mIoU | vs theta278 | 结果 |
|------|------|-------------|------|
| theta261+D4 | 69.639339 | -0.930pp | 基线 |
| theta278+D4 | 70.569757 | 0 | 旧冠军 |
| f_s2p100_s3p125 | 70.586921 | +0.017pp | 微调冠军 |
| GTRUST_T Geometry | 70.080740 | -0.506pp | 失败 |
| C2 SWA | 70.488526 | -0.098pp | 失败 |
| D1_2k (alpha=1.0) | 70.736134 | +0.149pp | 新冠军 |
| **D1_alpha050** | **70.763755** | **+0.194pp** | **当前冠军** |

## 3. 当前冠军包
- 文件: GTRUST_71_20260923/D1_alpha050_test2.zip
- SHA256: 821e6d6e6fe40de8a2245541241835972e3eefb85470cc1196840e770195f0c3

## 4. D1训练配置（已验证+0.149pp）
- 起点: theta278 (f_s2p100_s3p125)
- 数据: test2无标签1300张 + source GT 5597张
- hard pseudo-label: 261==278 agreement with confidence>=0.98
- Loss: source CE + ramp(500步)*0.5*hard_ce
- 可训练: segformer.stages.2.+stages.3.+decode_head.
- LR=1e-5, AdamW, WD=0.01, HEAD_MULT=0.05
- MAX_STEPS=2000, 实际639秒
- 强增强: strong_photo + D4 random view
- 脚本: /home/jhy/rare_diag_20260915/round2_20260921/scripts/_r2_dart_train.py

## 5. alpha插值方法
- theta_alpha = theta278 + alpha * (D1_2k - theta278)
- H300/R1099评价:
  - alpha=0.0: H=0.64222 R=0.77144
  - alpha=0.5: H=0.64133 R=0.77238 (peak on R)
  - alpha=0.75: H=0.64074 R=0.77237
  - alpha=1.0: H=0.63985 R=0.77177
  - alpha=1.25: H=0.63886 R=0.77071
  - alpha=1.5: H=0.63743 R=0.76948
- alpha=0.5线上=70.763755（最优）

## 6. 已失败方向（不要重复）

| 方向 | 结果 | 原因 |
|------|------|------|
| GTRUST_T Geometry | 线上-0.506pp | 本地holdout与test2不一致 |
| C2 SWA | 线上-0.098pp | 无新target信息，只是轨迹平均 |
| Ensemble | R1099无提升 | 模型互补性不足 |
| Stage-wise interp | 已peak | a2=0.5/a3=0.5最优 |
| Round2 self-training | pseudo-label不变 | alpha050 vs 278 flip仅0.612% |
| Head calibration | head变化0% | 不是head问题 |
| Prototype alignment | 无feature塌缩 | target类间距离>=source |
| UW1 uncertainty-weighted | H300-0.49pp | (1-conf)加权噪声区损害source |

## 7. 关键分析结论
- D1参数变化: stage2=77.3%, stage3=22.7%, head=0%
- D1 flip: 1.93%全test2, 82.6%在theta278低置信区域
- pseudo-label已干净: 83.8%像素confidence>0.95
- D1方向与ATDA正交: cos(Delta_ATDA, Delta_D1)~0
- 本地holdout与线上弱相关: GTRUST本地+0.87pp线上-0.51pp
- flip空间分布: 35.1%离边界>10px（不是纯边界问题）

## 8. 刚完成的实验
Multi-scale TTA: alpha050 x scales(0.75,1.0,1.25) x D4 TTA8
- 融合logits at 256x256
- 包: GTRUST_71_20260923/MSTTA_alpha050_test2.zip
- SHA256: b4b6915d9ba1924e5d247b750139ad77f9c8bbe862edcdc86ff2a8609c2644fb
- 尚未做行为审计，尚未决定是否提交

## 9. 服务器
- SSH: teacher-4090 (jhy@10.8.131.63:6816)
- GPU2可用
- venv: /home/jhy/.venvs/uavseg/bin/python3
- 项目根: /home/jhy/rare_diag_20260915
- test2 images: /home/jhy/datasets/AIC2026_UAV_semifinal/test_2/images/
- source split: SOURCE_AUDIT_SPLIT.json (H300=300, R1099=1099)

## 10. 给Claude的问题
当前冠军70.763755，剩余4小时，1个提交槽。
D1已证明target adaptation有效。UW1(CBD-D1变体)已失败。
已排除: pseudo-label weighting, prototype, head calibration, ensemble, SWA, self-training round2.

1. AC-FDA（训练stage0/1做Fourier amplitude风格适配，source GT监督，不碰test2 pseudo-label）是否值得做？D1只动了stage2/3。具体怎么实现？
2. Multi-scale TTA(0.75/1.0/1.25xD4)值不值得作为最后提交槽？
3. 4小时内还有什么可实现的新方向？
