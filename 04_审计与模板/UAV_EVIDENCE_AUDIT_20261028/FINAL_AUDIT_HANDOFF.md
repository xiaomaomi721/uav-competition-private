# FINAL_AUDIT_HANDOFF｜供 GPT 独立源码级审核

采集：`teacher-4090` / `jhy` / 2026-10-28 20:11–20:35 CST。
性质：只读取证。**未启动训练、未提交平台、未中断他人任务。**

本文件严格区分 **可据原始文件证实** 与 **仍只是报告自述**。

---

## A. 可据原始文件证实（VERIFIED_NOW）

### A1 冠军与提交包身份

| 对象 | 路径 | 字节 | SHA256 |
|---|---|---:|---|
| H75 checkpoint | `/home/jhy/aic2026_h76_20261026/models/H75.pt` | 189,091,023 | `b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a` |
| **官方已评分 ZIP** | `/home/jhy/aic2026_ctrl3594_teacher_20261027/reference/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip` | 31,099,544 | `c012bae1237160a0c0d5fbccd168f58facf62c03bf60ee2cc82877cb21454d30` |
| Claude C1 候选 | `/home/jhy/aic2026_h76_20261026/deliver/C1_3594/C1_3594_BSTAR.zip` | 31,156,937 | `90a55b7de2d0df11d2543d2c7d0b5581b94338db8394c11efe064a7d346ce3c1` |
| 3594 manifest | `/home/jhy/aic2026_h76_20261026/manifests/semifinal_3594.csv` | 460,173 | `76463ff99a495afe82085bb59530032f13f9efbff2cc1936f09178285e6cdd19` |
| M2F donor iter_30000 | `/home/jhy/aic2026_h76_20261026/experiments/T3_FULL_ST_20261027_import/teachers/Mask2Former_iter_30000.pth` | 2,610,605,869 | `dea40595287980777eae5751e76f5e86811071da9edc12b989b3c2ed0556c68c` |

> **结论**：70.998665 对应的提交包身份，本次已在服务器上独立核实，与历史记录一致。这是本包最硬的一条锚点。

### A2 当前运行状态（取自 nvidia-smi / systemd / state.json）

```
时间 2026-10-28 20:11:30 CST
GPU0 GPU-bc304d79-c807-968a-eaad-d2c2ecc5a984  2713 MiB  50%   ← H75_STPP_LITE_V2_CLASSAWARE
GPU1 GPU-c6eee6d9-9fa7-69d8-0ebc-a649e3528778    18 MiB   0%   ← 空闲
GPU2 GPU-cacb7500-f2e9-c482-d85c-53c7f8d1def9 14195 MiB  82%   ← 第二GPU M2F 60k 长训
GPU3 GPU-32853ccf-4c4f-1328-8d75-04efe9d5298f    15 MiB   0%   ← 空闲
```

- `h76-train.service` = **failed**
- `h76-watchdog.service` / `ziponly-watchdog.service` / `secondgpu-m2f-watchdog.service` = **active (running)**

### A3 第二 GPU M2F 长训的真实状态

`/home/jhy/aic2026_secondgpu_m2f_20261028/run_m2f_clean_secondgpu/long_m2f_clean_train/state.json`：

```
state = TRAINING
effective_step = 4796 / 60000
optimizer_updates = 4796
ema_updates = 4796, ema_momentum = 1e-4
accum = 8, runner_iter = 38367
median_s_per_effective_step = 3.026
eta_cst = 2026-10-30 18:36:08
pid = 2431347
```

`run_manifest.json`：`config_sha256=287974cfa8e071b7c202aac5b9037f2956ebd0096586c4943ae4328f6a7dcba8`，
`started_at=2026-10-28T17:33:16+0800`，`max_iters=480000`，`torch=2.1.0+cu121`。

**2000 步固定节点复核**（`review_20k/SCIENTIFIC_REVIEW_2000.md`，统一 768 协议、FP32、no TTA）：

| 指标 | 值 | 状态 |
|---|---:|---|
| EMA mIoU | **0.7380%** | `SCIENTIFIC_REVIEW_WARNING`，塌缩类 6 个 |
| raw mIoU | **22.8262%** | 警告，Barren 塌缩 |

对照参考：Exp00R 79.381034 / E02 B1 79.565390（**报告自述协议未互核**）。

### A4 Claude reliable_kd 的真实产物

| 文件 | 内容 |
|---|---|
| `train/C0/SUMMARY.json` | `completed_steps=3594`, `runtime_sec=1650.4`, `ess_frac_mean=1.0`, `peak_mem_mib=6020.1` |
| `train/C1/SUMMARY.json` | `completed_steps=3594`, `runtime_sec=1339.4`, `ess_frac_mean=0.94957`, `ess_frac_min=0.73499`, `peak_mem_mib=6020.6` |
| `teacher_diag/TEACHER_DIAG.json` | `d4_cal mIoU=77.531`（= 历史 77.5311488），`d4_raw=80.442`（= 历史 80.4417713） |
| `deliver/C1_3594/PACKAGE_REPORT.json` | `count=3594`, `bad_count=0`, ZIP SHA 见 A1 |

### A5 Codex TargetSSL 的真实产物

| 文件 | 内容 |
|---|---|
| `pilot_C1/PILOT_GRAD.json` | `grad_src=0.23953`, `grad_aux=0.13011`, `ratio=0.5432`, `λ_for_0.25=0.46023` |
| `pilot_C2/PILOT_GRAD.json` | `grad_src=0.23953`, `grad_aux=1.12789`, `ratio=4.7088`, `λ_for_0.25=0.05309` |
| `train_C*/RUN_CONFIG.json` | 三臂 `aux_head_params`：C0=0、**C1=24,768**、C2=0；`trainable_params=2,892,416` |
| `train_C*/SUMMARY.json` | 三臂 `frozen_diff_after_training = []`（冻结层零变化）|

### A6 DINOv3 V2 离线审计原始数据

`experiments/T3_ST_DINOV3_V2_OFFLINE_AUDIT/TARGET_A1_20261028_cmMv0d/TARGET_SUPERVISION_DELTA.json`：

```
selected_pixels = 37001
dino_mask_top1_agreement_fraction_on_selected = 1.0
dino_mask_tv_gt_0.1_fraction = 0.87452
dino_mask_tv_gt_0.2_fraction = 0.49553
all_KL_region_gradient_delta_l2_ratio = 0.20456
no_target_correctness_claim = true
```

> 注意：`agreement_fraction = 1.0` 说明**入选像素的定义就要求 DINO 与 mask 一致**，因此该数字是选择规则的产物，不能当作"独立正确性证据"。

---

## B. 仅属历史报告 / 未独立核验（REPORTED_ONLY）

| 说法 | 状态 |
|---|---|
| 线上 H75+Bstar = 70.998665；Zero = 67.979564；CTRL@3594 = 70.318575；T3-FULL-ST = 70.990138 | **REPORTED_ONLY**（服务器上未见平台回执截图/原始响应文件） |
| Exp00R 79.381034 / E02 B1 79.565390 / Exp04 TREAT 79.6019 | **REPORTED_ONLY**（评测日志与混淆矩阵来源未在本包闭环；Exp04 内部记录本身标 CONFLICT） |
| V1 原始 813 像素 / 3 图 vs 最终入选 0 | **REPORTED_ONLY**（未见 V1 原始 CSV） |
| V2 Source 净纠错 +16,660 像素 | **REPORTED_ONLY**（`V2_INDEPENDENT_AUDIT.csv` 已收，但未复算） |
| DINO-B 教师 2500+2500 步、Source1399 = 60.0826 | **REPORTED_ONLY**（未见该训练日志与权重 SHA） |
| EM/BBSE 是否真实运行 | **部分**：`a09a2026_ziponly_20261028/code/em_bbse.py` 与 `logs/em_bbse_*.log` 存在，但未核验其输出与实验结论 |

---

## C. 本任务明确未做（NOT_RUN）

- 未启动任何训练 / 全量推理 / 打包
- 未提交比赛平台
- 未重算逐视图 logits 分歧（需授权 GPU 且缓存未持久化）
- 未重跑 source1399 评估
- 未构建 Docker、未整理最终交付材料
- 未核验 GPU0 上 `H75_STPP_LITE_V2_CLASSAWARE` 实验的科学性（不在本任务范围，仅记录其在运行）

---

## D. 给出 GPT 的 5 个审核问题

1. **M2F 60k 长训是否应当立即停止？**
   证据：`effective_step=4796`，但 2000 步复核 EMA mIoU=0.74%、raw=22.83%，6 类塌缩。请判断这是"早期正常现象"还是"应当止损"。证据路径见 A3。

2. **`dino_mask_top1_agreement_fraction_on_selected = 1.0` 是否意味着 V2 的入选规则存在循环论证？**
   若入选条件本身要求 DINO 与 mask 一致，则"一致率 100%"不构成独立验证。请判断 V2 的 Source 净纠错 +16,660 像素结论是否仍然成立。

3. **Claude `votes` 统计的分母口径是否构成实质错误？**
   `votes` 在 native 256 生成，而"6065/1,445,188,329"的分母是 1024 上采样后的有效像素。这是否会推翻"`a` 项退化"的结论？见 `02_reliable_kd/D4_VOTES_AUDIT.md`。

4. **在 `h76-train.service=failed`、GPU3 空闲的情况下，`ziponly-watchdog` 仍 active，是否会造成误导性状态？**
   请判断是否需要清理（本任务未做任何变更）。

5. **最终交付材料（11-01 ZIP / 11-02 Docker+报告）当前无人负责，是否是本次审计中最严重的风险？**
   相比继续做方法实验，是否应把资源转向交付材料？

---

## E. 证据包结构

```
UAV_EVIDENCE_AUDIT_20261028/
  README_FIRST.md
  FINAL_AUDIT_HANDOFF.md        ← 本文件
  MISSING_AND_BLOCKED.md
  EVIDENCE_INDEX.csv
  SHA256SUMS.txt
  00_current_status/            ACTIVE_RUNTIME.md, TASK_REALITY_MATRIX.csv, CHAMPION_AND_SUBMISSION_ANCHORS.csv
  01_m2f_lineage/               M2F_IDENTITY_TABLE.csv, M2F_CURRENT_PIPELINE.md, 原始 config/state/review
  02_reliable_kd/               src/ (6 脚本) + artifacts/ (9 JSON) + D4_VOTES_AUDIT.md + C0_C1_MATCHED_CHECK.md
  03_h75_ssl/                   src/ (3 脚本) + artifacts/ (11 文件) + SSL_STATUS_CORRECTION.md
  04_dino_v2_evidence/          V2_FROZEN_RULE.json, V2_INDEPENDENT_AUDIT.csv, TARGET_A1_COVERAGE.json, TARGET_SUPERVISION_DELTA.json, REPORT_CORRECTION.json, SOURCE_GATE_SPLIT.json
  05_prior_and_delivery/        em_bbse.py, em_bbse_*.log, 02_em_bbse 产物
```
