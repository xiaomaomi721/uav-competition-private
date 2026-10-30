# 当前运行状态（只读快照）

- 采集时间：**2026-10-28 20:11:30 CST (+0800)**
- 主机：`ubuntu-Super-Server`（别名 teacher-4090，`10.8.131.63:6816`）
- 登录用户：`jhy`
- 项目工作根：`/home/jhy/aic2026_h76_20261026`

---

## GPU（nvidia-smi 只读）

| 物理号 | UUID | 显存占用 | 利用率 | 归属进程 |
|---:|---|---:|---:|---|
| 0 | `GPU-bc304d79-c807-968a-eaad-d2c2ecc5a984` | 2,713 / 24,564 MiB | 50% | `2774419 jhy` → `H75_STPP_LITE_V2_CLASSAWARE_20261028T112548Z/score_full.py` |
| 1 | `GPU-c6eee6d9-9fa7-69d8-0ebc-a649e3528778` | 18 MiB | 0% | **无** |
| 2 | `GPU-cacb7500-f2e9-c482-d85c-53c7f8d1def9` | 14,195 / 24,564 MiB | 82% | `2431347 jhy` → 第二GPU M2F 60k 长训；另有 `2803029` review_20k |
| 3 | `GPU-32853ccf-4c4f-1328-8d75-04efe9d5298f` | 15 MiB | 0% | **无（主线 GPU3 空闲）** |

---

## systemd 用户服务（只读查询，未做任何变更）

| 服务 | 状态 | 说明 |
|---|---|---|
| `h76-train.service` | **failed** | H76 CaSC 队列，历史已停止 |
| `h76-watchdog.service` | **active (running)** | 心跳存在但无训练负载 |
| `ziponly-watchdog.service` | **active (running)** | 半决赛 ZIP-only 单卡任务看门狗 |
| `secondgpu-m2f-watchdog.service` | **active (running)** | 第二 GPU M2F 看门狗（物理 GPU2 only） |

> ⚠️ 注意：`ziponly-watchdog` 与 `secondgpu-m2f-watchdog` 均 active，而 `h76-train` 为 failed。看门狗存活 ≠ 有训练在跑。

---

## 相关进程（jhy 可见）

| PID | 运行时长 | 命令摘要 |
|---|---|---|
| 2431347 / 2431692 / 2431729 | 2:38:09 | `uav_secondgpu.train ... m2f_clean_60k.py --run-id M2F-CLEAN-60000-20261028` |
| 2803029 | 0:07:59 | `uav_secondgpu.review_20k ... 2000 both`（2000 步复核） |
| 2774419 | 0:20:54 | `H75_STPP_LITE_V2_CLASSAWARE/score_full.py`（GPU0） |
| 2774405 | 0:20:54 | 同实验 `orchestrate.py` |
| 2627051 | 1:19:48 | `ops/watchdog.py` |
| 2805766 | 0:06:59 | `aic2026_secondgpu_m2f_20261028/ops/watchdog.py` |

---

## 最近心跳

`/home/jhy/aic2026_secondgpu_m2f_20261028/run_m2f_clean_secondgpu/long_m2f_clean_train/state.json`：

```json
{
  "state": "TRAINING",
  "effective_step": 4796,
  "optimizer_updates": 4796,
  "ema_updates": 4796,
  "ema_momentum": 0.0001,
  "accum": 8,
  "locked_target_steps": 60000,
  "median_s_per_effective_step": 3.0259,
  "eta_cst": "2026-10-30 18:36:08",
  "timestamp": "2026-10-28T20:12:06+0800"
}
```

---

## 本次采集行为声明

- 仅使用 `nvidia-smi`、`systemctl --user status/list-units`、`ps`、`stat`、`sha256sum`、文件读取
- **未执行** restart / stop / start / kill / 修改任务队列
- **未读取**其他用户（`daiqun` 等）的进程细节（仅有 nvidia-smi 显示其显存占用）
