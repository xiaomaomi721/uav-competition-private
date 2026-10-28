# 新历史流式factory部署前核对项

## 不改变当前服务器执行

当前Rare/Param streaming supervisor继续工作。本次新observer不发送信号、不替换活跃scheduler、不修改其source/policy，先只读等成功终态。线上结果未知时不启动Rareownscan。

## 必须完成后才冻结/部署

- canonical registry固定类顺序/向量，冠军分数来源边界保留。
- actual1399GT histogram SHA、原split SHA、320整图subset/stems SHA；不把fixture混进production。
- FP32原D4 inverse/resize/stack.mean原顺序，无精度降低，无长期logits。
- PassA完整subset一次forward评估全部向量；PassB完整1399一次forward评估所有候选；不逐bias重复forward。
- perimage int64confusion与fullGT有效像素一致、全图predictionhash、每fold与两个历史bias和zero的delta。
- 实际评估了的邻域才计稳定性；最多26候选的有限邻域不能声称完整8D盆地。
- 角色不足或等价须alias，不强造8–12个不同包。
- 最终test一遍多bias直ZIP，按500图实际prediction全内容去重。
- alias_of必须为实际canonical ZIP绝对路径，与queue outputs_sha256兼容；alias_role另存。
- 官方check包括500全量PNG与类别规则；原ZIP未动；新重复ZIP只有新建/完整SHA/官方PASS/审计记录后可删除。
- 所有source/config/checkpoint/plan/stats/ZIP须有SHA绑定；report不能仅statusPASS。
- 共享后处理锁串行，GPU2uuid/PCI/CVD空闲核验，禁止占用GPU3。
- 3GiB安全+2GiB训练增长+working+多core输出预算，每图/每轮检查；不足只停止后处理。
- 长任务异常保留锁直到已知child自然终止；不重试、不恢复、不偷偷换checkpoint。
- observer在安全完整core边界看userwinner通知；出现文件只可保守停队，真实资格仍需核验，不自动Rarelaunch。
- Stage66只52000，其他两步/第二批/规则阻塞pseudo不自动扩展。

## 验证声明

真实GPUfactory端到端尚未执行。CPU CLI/fixture只证明对应接口路径，不代表生产GPU验证。若权限仍阻止上传，就保持本地准备状态，不声称已部署或自动回收在跑。
