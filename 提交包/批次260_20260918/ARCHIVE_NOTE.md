# ARCHIVE — Axis S 停止收尾（2026-09-18 16:16）

**状态：Axis S（SEMEDA stage2 surgical）已按指令终止，不再占用 GPU2。本目录及以下资产原样保留，未删除。**

## 1. 停止记录

| 项 | 值 |
|---|---|
| 训练 | `p1_runs/p1_semeda_s2_init209_seed42`（init=N9_e050_dm010，只训 segformer.stages.2） |
| SIGKILL 时 step | **7060** / 16000 |
| 终止方式 | SIGINT（进程 + 进程组）**未生效**，改用 SIGKILL |
| 后果 | 未写出 `last.pt`；**step 4000–7060 不可恢复** |

## 2. 最近一个完整可 resume 的 checkpoint

| 项 | 值 |
|---|---|
| 路径 | `p1_runs/p1_semeda_s2_init209_seed42/checkpoint_step4000.pt` |
| 大小 | 431,021,277 bytes |
| **sha256** | **dc7c134b1a587f271c93e89da9aa003686d273f0a3715770b756edeb58e84a26** |
| step | 4000（optimizer_update_count 4000） |
| 含 | model(644) / optimizer / scheduler / scaler(scale=262144.0) / step / config |
| 不含 | RNG / dataloader state（该 trainer 本就不保存） |

同时保留：`checkpoint_step1000.pt`、`checkpoint_step2000.pt`、`best_miou.pt`。

## 3. 保留未动（未删除、未修改）

```
phase0_eval_20260918/           Phase 0 全部产出与日志（10/10，PHASE0_EVAL_DONE 13:59:32）
axisS_cands_step1000/           1k 候选构造 + 已打出的 1 个包 + pack.log
p1_runs/                        Axis S 训练全部 checkpoint 与 run_init209.log
```

## 4. 提交批次（本地 `提交包/批次260_20260918/`）

| 编号 | 状态 |
|---|---|
| 260–270 | **未提交**，原样保留 |
| 271 | **坏包 HOLD**（flip 23.27% vs 正常 3.7%；成因：把 1000 步训练方向放大 5.31× 以凑 R_old） |

## 5. 本指令下明确未做

未打包 Axis S；未跑 8k/16k；未生成更多 Phase 0 候选；未跑 calibration；未跑 cosine/SVD；未修改 Axis T-lite；未写新研究方案。
