# STORAGE_AWARE_CALIBRATION_PREFLIGHT

协议：STORAGE_AWARE_CALIBRATION_PROTOCOL_V1。

**容量与CPU接口预检：PASS。GPU生产推理尚未执行，不能标为GPU端到端PASS。**

## 1–3. 训练与空间快照

2026-09-14 17:09:06服务器时间：
- RareExposure v2：GPU2，formal PID/SID283448，step49817/56000。
- Paramwise01：GPU3，formal PID/SID2115724，step3174/56000。
- 初始可用10,891,079,680字节，约10.14GiB；真实新CLI预检时可用10,872,913,920字节，约10.13GiB。

两条正式训练都不停止、不resume、不改参数、不换卡。本轮只替换post-training scheduler路径。

## 4. Scratch只读审计

已只读执行df/lsblk并检查/data、/mnt、/scratch及已知scratch位置。没有确认属于本项目、可写且>220GiB的scratch。

- /data、/scratch不存在。
- /mnt为root拥有、当前用户不可写，仍属根文件系统。
- /ssd、/hdd空闲大，但当前用户无根目录写权限；明确项目子目录不存在。
- 不进入其他用户目录，不创建scratch试探权限，不移动formal或checkpoint。

**选择：全分辨率FP32 streaming，不使用其他用户scratch。**

## 5. 原200.80GiB门槛来源

原方案属于 **persistent full-resolution FP32 cache architecture**，不是算法必需容量。

| 项目 | 字节 | 说明 |
|---|---:|---|
| 三个1399×8×1024×1024 FP32 val cache | 140827951104 | 每个43.71875GiB，三个131.15625GiB |
| 500×8×1024×1024 FP32 test cache | 16777216000 | 15.625GiB |
| NPZ容器开销 | 153911296 | 不计压缩收益 |
| PNG/ZIP多角色保守上界 | 52477952000 | 旧方案重复保留多个输出形态 |
| 旧安全余量 | 5368709120 | 5GiB |
| 合计 | 215605739520 | 约200.80GiB |

新实现不落盘完整val/test logits，也不保留prediction目录再复制ZIP。

## 6. LOW_RES_EXACT_CACHE

**NOT_TESTED / NOT_ENABLED。**

当前代码每view逆变换→bilinear上采样(align_corners=False)→FP32八view mean。低分辨率先聚合后resize、或bias/resize交换虽然有数学线性依据，但FP32舍入与argmax相等不能仅凭推导认定。

没有占用GPU2/GPU3运行32张等价审计；默认fallback直接使用原始FP32全分辨率推理。保留原8-view张量mean的归约顺序，不改成逐view累加。不使用FP16/BF16/INT8/有损压缩。

## 7–10. Streaming峰值磁盘与是否可出包

实际31个历史ZIP（全部保留）：
- min：3,853,993字节。
- median：4,504,849字节，约4.30MiB。
- max：4,715,891字节，约4.50MiB。
- sum：137,282,345字节。

新冻结门禁采用**4×历史max/包**，不是未经证实的实际最终大小：
- 单包估计18,863,564字节。
- 三包56,590,692字节，约53.97MiB。
- 十包188,635,640字节，约179.89MiB。
- Rare最多两个不同权重×3角色：113,181,384字节。
- Paramwise最多三个不同权重×3角色：169,772,076字节。

门禁：`estimated_output + working_temp + safety_reserve + formal_growth_reserve + other_queue_output_reserve`

| 项目 | Rare | Paramwise |
|---|---:|---:|
| 本队列全部预计ZIP | 113181384 | 169772076 |
| 工作临时预算 | 268435456 | 268435456 |
| 系统安全余量 | 3221225472 | 3221225472 |
| formal未来写入保护 | 2147483648 | 2147483648 |
| 另一队列预计输出保护 | 169772076 | 113181384 |
| 所需可用空间合计 | **5920098036** | **5920098036** |

**两队列联合保留条件约5.51GiB，当前约10.13GiB，均PASS。Rare和Paramwise均具备流式生成单checkpoint三包的容量条件。** 这不是包已生成或一定通过官方检查的保证。

formal增长保护依据：实际best单文件约189MB；Param剩余5milestone+last约1.06GiB，Rare last约0.18GiB，另留日志/可视化等余量，总体预留2GiB，独立于3GiB系统余量。

每图还会检查剩余空间、下一批PNG与ZIP结尾开销；压缩量超4×max预算就停止后处理并保留未认证产物，不侵占保护余量。两队列后处理持共享flock串行写出，但两条训练不受该锁限制。

## Checkpoint事实

- Paramwise：best / checkpoint_step52000 / checkpoint_step56000，按模型tensor内容去重。
- RareExposure：当前冻结trainer只保存best和terminal last，无52k周期保存。完成后认证last.step=56000，将其视作56k别名；best/last权重相同则去重。**不修改训练补保存，不伪造52k。**

## 真正执行路径

`one checkpoint TTA8 test forward -> FP32 mean -> three fixed biases -> PNG BytesIO -> ZipFile.writestr -> discard image logits`

不保存完整logits、不生成prediction目录。官方checker逻辑不改，使用只读ZIP虚拟路径适配器执行原main；逐条再核验PNG/尺寸/单通道/ID/CRC及全部500图内容SHA。仅本次新建且全部检查PASS的完全等价ZIP可按明确alias/授权审计去重，历史文件和未完成ZIP不删除。

## 线上门禁与后续8D

匹配三包完成后状态：**WAITING_USER_ONLINE_SELECTION**。不自动线上提交；没有真实线上结果就没有线上DeltaCore/DeltaFixedCal，也不会用local替代或自动开启扫描。

后续获胜checkpoint的新协议：PassA固定320张whole-image/class-coverage subset进行63向量粗扫；PassB最多26向量完整1399验证、保存每image8×8 confusion（约17.76MiB上界）、整图fold与明确范围的邻域稳定性；最终test一遍forward同时写最终多个biasZIP。匹配阶段test与胜者最终test是不同阶段各一次，不冒称全流程只发生一次。

当前已完成独立policy-only planner及CPU fixture验证；**真实PassA/PassB GPU执行和最终角色选择入口尚未实现/启用**，待真实线上winner后完成其生产接入。不能把planner宣称为完整扫描器。

## 实际接口验证

- Rare真实driver --preflight：PASS，500输入图，torch_imported=false，GPU_started=false。
- Paramwise真实driver --preflight：PASS，同上。
- 两个supervisor真实--validate-only --before-handoff：PASS，核验实际formal argv/start/SID/旧锁与依赖SHA，无进程信号。
- 实际两图PNG→streamingZIP→原官方checker：PASS，仅fixture，不是正式包。
- 现有Golden完整500图ZIP经新适配器：官方PASS，0 failures / 1 warning；不解压、不改历史ZIP。原SHA仍6be4e8b0b15aaa4a002bb99d1b1fe0677ec9fb67fe1ee32ac4e0fda5d763ee04。
- 错误policy SHA与重复输出目录：实际CLI均拒绝，旧回执保留。
- GPU推理阶段没有为了预检抢占正在训练的GPU。其数学内核复用已冻结v2代码；本次不称GPU全流程已验证。

## 11. Formal配置SHA

- RareExposure：`fc6005a5e8a59e03f134a17b832e620e2764a36697e789b5dd5c60d9a181fe09`。
- Paramwise：`b13dabef0c44ac802bff16443d8fc7c1d81606540c6dde2c3cc5b22321f4068d`。

**两者与启动时完全一致。**

## 部署状态

新代码与policy已在`/home/jhy/storage_aware_calibration_v1`独立冻结，CPU预检PASS。此预检文件生成时尚未暂停旧调度器；实际交接结果另写STORAGE_AWARE_HANDOFF_REPORT.json，避免把待执行说成已完成。旧训练/旧冻结代码不热改，formal不接收信号。
