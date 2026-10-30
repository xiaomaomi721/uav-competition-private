# 历史重复核查（H75 × T3 深层融合是否已做过）

检查时间：2026-10-28。范围：`/home/jhy` 授权目录。

## 1. 历史权重平均产物清单（VERIFIED_NOW）

`/home/jhy/aic2026_h76_20261026/experiments/soup/`

| 文件 | SHA256 | 成员构成 |
|---|---|---|
| `soup_3way.pt` | `e538bd04dfada5ea2fc77777d04929219db499d4497aca65ee51825da1c7a747` | H75 + CTRL@5400 + BR-Control（1:1:1） |
| `soup_3seed.pt` | `a3837769293fc6c945e6871c398de62bd92da0bf51b86d52ccca0d158e248ba4` | CTRL@5400 + ctrl_s2 + ctrl_s3 |
| `soup_h75_ctrl.pt` | `033b3d6e6d237d91ccd51fd3ad7ff4147d6564122cdfd6ea4926d6255426857c` | H75 + CTRL@5400 |
| `soup_4mem.pt` | `0af6bd5407609f2671ef6ae39f74dc49b664a32d86ac8dde1c4153312a384abe` | CTRL@5400 + s2 + s3 + BR-Control |
| `soup_ctrl_br.pt` | `473f79228f2e96ef78536f580b29fb6dbe7a53ee13a9febbdaf64470fc097e4e` | CTRL@5400 + BR-Control |
| `soup_w211.pt` | `c3be02fc06cc269f0b2432392bdcc7e2885dfc563ea187e20f8281398e5793df` | CTRL 2 : BR 1 : H75 1 |

## 2. 结论：**无重复**

- 上述六个 soup **全部不含 T3**（`checkpoint_update3594.pt`，SHA `cb958870…e8e8`）。
- 搜索 `*DEEP25*`、`*H75_T3*`、`*interp*` 在授权目录内**均无结果**。
- 未见任何以 α=0.25 对 `segformer.stages.2/3 + decode_head` 做 delta 融合的历史产物。
- 历史 ZIP 中的 `soupC_*` 系列属于 2026-09 的早期 farmland/water bias 扫描，与本次无关。

**判定：本任务为全新工作，不存在可复用的既有产物。**

## 3. 与历史 soup 的机制差异（为什么这不是简单重复）

| 维度 | 历史 soup（soup_4mem 等） | 本次 Deep25 |
|---|---|---|
| 融合对象 | 全部**训练后**同配方 checkpoint（CTRL 系） | H75（冠军）与 **T3-FULL-ST**（用全部 5594 源 + 3594 目标自训练） |
| 权重点 | 等权 1:1:1:1 或加权 | **固定 α=0.25**，单点，不扫描 |
| 深度选择性 | 全量平均 | **只融合 stage2/3/head，stage0/1 严格 bitwise 复制 H75** |
| 目的 | 降方差 | 把 T3 学到的目标域增量以 25% 注入冠军，同时保留 H75 的浅层 |

T3 与 H75 的 stage0/stage1 **逐位相同**（166 张量 max_abs_delta=0），因此"只融合深层"在数学上等价于"全量融合"——浅层无差异可融合。这一点已在 `ASSET_SHA_AND_LAYER_DELTA.csv` 中逐张量记录。
