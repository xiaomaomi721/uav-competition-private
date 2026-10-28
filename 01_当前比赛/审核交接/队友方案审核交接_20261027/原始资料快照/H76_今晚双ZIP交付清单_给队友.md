# H76 半决赛双 ZIP 交付清单（队友版）

**日期：** 2026-10-26\
**状态：** 两个预测 ZIP 已在服务器生成并通过报告中的
QA；本文件生成时，尚未确认已下载到本地或提交平台。

## 1. 本轮交付的两个 ZIP

### A. H75_BSTAR（基线候选）

-   完整服务器路径：\
    `/data/uav/reproduction/H75_713294_REPRO/11_H76_ACDD_3594_20261026/p0b_runs/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip`
-   报告中的 SHA256 缩写：`c012bae1…454d30`
-   报告 QA：`FORMAT_QA_PASS=True`
-   3594 张 PNG；根目录平铺；单通道 `L`、1024×1024、`uint8`、标签值
    1--8。
-   配置：H75 + D4 TTA8 + 原始 Bstar bias。
-   定位：当前基线候选，建议作为第一份对照提交。

### B. H75_ZERO（Zero Bias 对照）

-   完整服务器路径：\
    `/data/uav/reproduction/H75_713294_REPRO/11_H76_ACDD_3594_20261026/p0b_runs/H75_ZERO_20261026/H75_ZERO_semifinal.zip`
-   报告中的 SHA256 缩写：`9280d809…592cff`
-   报告 QA：`FORMAT_QA_PASS=True`
-   3594 张 PNG；根目录平铺；单通道 `L`、1024×1024、`uint8`、标签值
    1--8。
-   配置：与 H75_BSTAR 相同，仅固定类别 bias 改为全零。
-   定位：单变量对照，用于验证固定 bias 的实际线上影响。

**哈希注意：**
上述是报告中的缩写，不能用于最终完整性证明。下载前需读取服务器完整
SHA256；下载后在 Windows 计算完整 SHA256，并逐字核对。

## 2. 两个候选共同的来源

-   Checkpoint：`/data/uav/runs/ASTRA_H75_X110_20261005/ckpt/H75.pt`
-   Checkpoint
    SHA256：`b5eea2795d8a6d90c803a15b2ca66cf0abcf62322ef1aacd0d4d0ec82098a28a`
-   输入 `test_3.zip`
    SHA256：`77b3d396a8f503a0379983642a9cedbab8ded833ab5a16f2b92d832bb5c5b9b4`
-   输入总数 3594 = test1 500 + test2 1300 + 新增 1794。test1/test2
    已包含在合并集中，不可重复拼接成 5394 张。
-   两候选共用 checkpoint、manifest、D4 TTA8、预处理、logits
    对齐/融合、上采样和标签转换；仅 bias 不同。

## 3. Claude 报告中的 QA

  检查项                       报告结果
  ---------------------------- ----------
  每份 ZIP 含 3594 张          PASS
  文件名与清单对应             PASS
  PNG 单通道 `L`               PASS
  1024×1024 / uint8            PASS
  标签值 1--8                  PASS
  缺失/重复检查                PASS
  两候选 only-bias-diff        PASS
  历史 H75 checkpoint 未修改   PASS
  Docker 镜像构建              未执行

报告称两候选约 5.56% 像素不同、涉及 3539 张图像。该结果证明 bias
改变预测，不证明哪个候选目标域分数更高。

## 4. 下载到本地

目标目录：`E:\A人工智能大赛\A半决赛\zip`

下载后应保留文件名： - `H75_BSTAR_semifinal.zip` -
`H75_ZERO_semifinal.zip`

Windows PowerShell 本地核验：

``` powershell
$dir = 'E:\A人工智能大赛\A半决赛\zip'
Get-ChildItem $dir -Filter '*semifinal.zip' |
    Select-Object Name, Length, FullName

Get-FileHash "$dir\H75_BSTAR_semifinal.zip" -Algorithm SHA256
Get-FileHash "$dir\H75_ZERO_semifinal.zip" -Algorithm SHA256
```

将本地完整哈希与服务器完整哈希逐字核对。若同名文件已存在且哈希不同，不覆盖；停止并报告差异。

## 5. 发给 Claude Code 的下载指令

``` text
只执行两个预测 ZIP 的服务器到本地下载；不训练、不推理、不构建 Docker、不提交平台。

目标目录：E:\A人工智能大赛\A半决赛\zip

文件 1：
/data/uav/reproduction/H75_713294_REPRO/11_H76_ACDD_3594_20261026/p0b_runs/H75_BSTAR_20261026/H75_BSTAR_semifinal.zip

文件 2：
/data/uav/reproduction/H75_713294_REPRO/11_H76_ACDD_3594_20261026/p0b_runs/H75_ZERO_20261026/H75_ZERO_semifinal.zip

1. 使用现有 SSH 配置/已验证服务器别名，不猜 IP 或用户名，不输出凭据。
2. 下载前读取服务器两文件的完整 SHA256 和字节数。
3. 只复制以上两个 ZIP，不下载整个 P0-C 目录。
4. 若同名文件已存在且哈希不同，不覆盖；停止并报告。
5. 下载后计算本地完整 SHA256，与服务器值比较；并对 ZIP 做完整性测试。
6. 最终只报告文件路径、字节数、服务器 SHA256、本地 SHA256、校验 PASS/FAIL。
若连接或路径无法确认，停止并报告阻塞，不猜测、不反复重试。
```

## 6. 分数与提交的解释边界

-   source1399 H75+Bstar mIoU：`0.7753114880`。
-   source1399 H75+Zero Bias mIoU：`0.8044177132`。
-   这些不是半决赛测试集 mIoU，不能据此断言 Zero Bias 线上更优。
-   建议在额度可用、本地完整哈希和 QA 通过、负责人接受格式风险后，按
    BSTAR → ZERO 顺序提交。每次记录时间、文件名、提交
    ID、分数、回执截图和完整 SHA256。

## 7. 仍需说明的风险

1.  官方公开文字未明确 ZIP 必须根目录平铺还是放入子目录；当前 flat ZIP
    是历史 H75 打包习惯，不是官方独立确认。
2.  当前 H75 输出值为 1--8；官方文本记录值域为 0--8，但逐类 ID
    映射仍应以官方文件为准。
3.  P0-C 复现目录已完成静态校验，但 Docker 未实际构建。
4.  H75 参数平均的书面许可仍待确认；本轮 ZIP 不涉及参数平均。
5.  最新教师审计对当前 ACDD-3594 recipe 的结论是
    NO-GO，今晚不启动该训练。

## 8. 关联材料

-   P0B 报告：\
    `/data/uav/reproduction/H75_713294_REPRO/11_H76_ACDD_3594_20261026/p0b_runs/P0B_BOTH_PASS_REPORT.md`
-   P0C 复现交付目录：\
    `/data/uav/reproduction/H75_713294_REPRO/11_H76_ACDD_3594_20261026/p0c_delivery_20261026/`
-   交付验证：`DELIVERY_VERIFY.json`
-   完整哈希清单：`SHA256SUMS.txt`

P0-C 是复现材料目录，不是应直接上传排行榜的预测 ZIP。

**交付原则：**
先下载并完成完整哈希核验，再由负责人决定是否提交。没有平台回执时，只能说"预测
ZIP 已生成并通过报告中的 QA"，不能说"已提交"或"线上成绩已确认"。
