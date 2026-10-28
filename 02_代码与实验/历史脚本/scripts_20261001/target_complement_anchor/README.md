# Target Complement Anchor (TCA v4)

本目录是 2026-10-02 冻结的单候选执行包。它不修改历史代码，不访问 test2 GT，不自动提交平台。

## 为什么做

- Arm A（纯 target）70.957249，Arm B（source=0.1）71.045315：需要防漂移护栏；
- source-only CONTROL 相对 A05 约 −0.503pp：source CE 本身存在域税；
- RAS-KD TREAT 线上 71.042244，较 Arm B −0.003071pp：当前 mask 内 D4-JS 类内重加权没有增益；
- BLACK_CORE_DROP 线上 71.029651，较 Arm B −0.015664pp：该复合包没有增益，而且黑区删除不应混入 TCA 的唯一变量。

RAS 与 BLACK 两包相对历史 Arm B 都带有共同 `xlogy` 变化，因此各自净效应仍需 RAS CONTROL 分数隔离；但这不改变 TCA 的预注册配方。

TCA v4 保留 Arm B 的 A05 teacher、阈值、soft KL、增强、训练范围和 endpoint，只做：

```text
source CE -> 完全移除
selected mask -> 主 KL
完整网格的 mask complement -> 梯度比一次冻结的弱 KL anchor
teacher KL -> zero-safe xlogy
```

TCA v4 **不排除 black core**，selected 与 complement 严格划分完整 256×256 teacher 网格。这样不混入 BLACK_CORE_DROP，保持唯一科学问题。

## 文件

- `tca_common.py`：共享契约和数学实现；
- `calibrate_tca_mu.py`：无参数更新地冻结唯一 mu；
- `train_tca.py`：4-loop smoke / 2000-loop formal；
- `package_tca.py`：单 checkpoint 标准出包；
- `compare_submission_packages.py`：与 Arm B 做无 GT behavior 比较；
- `test_tca_contracts.py`：纯 CPU 合约测试；
- `执行手册_交队友.md`：唯一授权执行顺序。

## 版本与过期规则

唯一有效公式版本：

```text
tca-v4-xlogy-zero-safe-all-pixels-selected-plus-gradient-calibrated-complement
```

所有 `tca-v3-*` 代码包、calibration receipt、checkpoint 和旧桌面 ZIP 均为 **obsolete**，不得与 v4 混用。v4 calibration receipt 会绑定新的 `tca_common.py` SHA，必须从 calibration 重新开始。

## 本地验证状态

发布前本地验证已完成：

- Python compile：PASS；
- Ruff：PASS；
- 11/11 unit tests：PASS；
- teacher KL 的 `torch.xlogy(q,q)-q*logp` 零概率测试：PASS；
- calibration/train/package 静态检查不含 black exclusion、source loader/CE、test2 GT、网络或上传：PASS；
- CLI import/help：PASS；
- Arm B ZIP self-compare：0 changed pixels，1300/1300 identical。

尚未完成、必须由队友在新服务器执行：

- GPU identity/preflight；
- 1300 teacher cache 全量验证；
- v4 mu calibration；
- smoke/formal；
- 正式出包和 server-side SHA。

## 今日提交纪律

- 槽 1：人工提交既有 RAS CONTROL，补完昨天 RAS/BLACK 的共同 `xlogy` 控制；
- 槽 2：仅在完整证据门通过后人工提交 TCA v4；
- 不生成第二个 mu，不生成 TCA+RAS/TCA+BLACK，不使用 step1000；
- 若 TCA v4 未及时通过，第二槽空置；任何脚本均不包含平台上传功能。
