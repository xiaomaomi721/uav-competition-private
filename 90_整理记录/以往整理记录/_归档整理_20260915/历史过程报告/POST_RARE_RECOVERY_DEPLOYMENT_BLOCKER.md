# POST_RARE_RECOVERY — 部署权限阻塞

用户已通过明确选项授权向 teacher-4090（jhy@10.8.131.63:6816）/home/jhy 项目目录部署新增代码与报告。

但后续真实SCP工具调用再次被Auto模式拒绝：

> The SCP uploads project reports and calibration metadata to teacher-4090, a non-repository remote host not listed as a trusted destination; run this step outside auto mode so the user can review the transfer directly.

因此：
- 不把用户授权误报为工具已经允许。
- 不更换工具、编码或stdin脚本绕过传输阻止。
- 被拒绝的复合命令没有执行，新的post_rare_recovery_v1远程目录/报告/代码不视为存在。
- 当前已部署storage-aware supervisor与两个formal继续运行，不发信号、不恢复旧scheduler。
- 仅继续新增本地factory/statistics/planner/queue实现与允许的只读调查。
- 远程部署需要用户切换为可逐次确认的非Auto权限模式后直接审核传输操作。

当前命名注册文件CANONICAL_CALIBRATION_IDENTITIES_V1.json与ACK仅本地完成；新历史factory尚未部署/冻结/运行。不要用“handoff complete”描述本次扩展。
