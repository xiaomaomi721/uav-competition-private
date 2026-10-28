# POST_RAREEXPOSURE_STREAMING_CALIBRATION_ACK

实测时间：2026-09-14 18:39:09（服务器时间）。

1. RareExposure：GPU2，formal PID/SID283448，step54297/56000，继续运行。
2. Paramwise：GPU3，formal PID/SID2115724，step7790/56000，继续运行。
3. 正式配置SHA均与启动时相同：
   - Rare：fc6005a5e8a59e03f134a17b832e620e2764a36697e789b5dd5c60d9a181fe09
   - Param：b13dabef0c44ac802bff16443d8fc7c1d81606540c6dde2c3cc5b22321f4068d
4. 可用空间：10,869,813,248字节，约10.12GiB。
5. 冠军向量正式命名已写本地CANONICAL_CALIBRATION_IDENTITIES_V1.json：CHAMPION_8D_69_281290=[-0.2,0.1,0.0709,0.7,3.6,0.4,3.3215,-0.5]。保留“69.281290来源于历史项目账本，未取得独立平台评分回执。”主对照为冠军向量，LOCKED8D为第二参考；Exp41_EDGE_BCE_CORE单独标识。旧冻结文件不改名/覆盖；服务器当前运行的新流式调度器仍使用旧标签，尚未更新到本次命名。
6. Streaming Calibration Factory：**完整生产factory尚未完成/冻结**。已冻结运行的是固定三包streaming路径；两阶段目前只有policy planner，不冒称PassA/PassB及角色选择已实现。
7. 用户最高优先级修正后的自动顺序：formal audit -> matched three-package generation -> MATCHED_PACKAGES_COMPLETE_WAITING_ONLINE_RESULTS。真实线上matched结果决定DeltaCore/DeltaChampionCal/DeltaLockedCal资格；没有用户提供的线上结果，不标winner，不运行Rare PassA/PassB。等待期间可在GPU2空闲后进行历史回收，未来确认Rare资格只在当前完整推理任务结束后的安全边界让位。当前已部署streaming supervisor保持有效；本次历史factory扩展尚未部署。
8. 历史第一批：Golden -> Exp47 -> Stage66-52000 -> Exp45；第二批MixStyle/MaxEnt/DINO/Exp41保持QUEUE_TTA_SECONDARY；Stage66其他两权重需符合后续触发条件；Pseudo RULE_BLOCKED。
9. 不自动开启第三条56k训练。
10. 不自动线上提交。

## 已知checkpoint限制

Rare当前冻结trainer只保存best和terminal last，未设置52k周期保存；不能停止/重启/改训练补造52k。完成后审计真实文件，last.step56000可作56k别名，记录52k=NOT_SAVED。Paramwise保留真实best/52k/56k。

## 已解决的口径与剩余部署确认

用户补充已明确：资格只以真实线上matched结果为准，不用local/val替代；Rare仅best+last/56000，Paramwise仍best/52k/56k。当前已部署streaming supervisor保持有效，不恢复旧scheduler。

用户已明确授权teacher-4090部署，但真实传输随后仍被Auto模式拒绝，要求切换为非Auto逐次审核。本轮新远程部署尚未执行，不绕过拦截；现有训练与已部署流式scheduler继续有效。详见POST_RARE_RECOVERY_DEPLOYMENT_BLOCKER.md。

本轮目前仅只读状态检查和新增本地文档；没有发进程信号、没有修改服务器训练或活跃策略。
