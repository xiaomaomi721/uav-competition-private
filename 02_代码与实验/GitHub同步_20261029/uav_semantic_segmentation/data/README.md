# 本地数据目录说明

官方比赛数据不进入 Git 仓库。每位成员必须通过比赛官方或其他获得授权的渠道自行获取数据，并对本地数据承担保密责任。

## 推荐本地结构

数据可以放在仓库外部；如果确需放在项目 data 目录下，建议采用以下结构：

    data/
    ├── README.md
    └── official/
        └── <DATASET_ROOT_CONTENTS>/
            ├── Label.txt
            ├── images/
            └── train/
                ├── images/
                └── masks/

data 目录中的 README.md 会保留在 Git 中，其他内容均由 .gitignore 排除。推荐优先把数据放在仓库外，以减少误提交风险。

## 运行只读检查

使用命令行显式传入本地数据集根目录：

    python scripts/inspect_dataset.py --dataset-root "<DATASET_ROOT>" --output-dir reports --sample-count 20 --seed 42

<DATASET_ROOT> 应指向同时包含 Label.txt、train/images、train/masks 和测试 images 的官方解压目录。

## 安全规则

- 不复制、上传或提交官方图像、标签和压缩包。
- 不修改、插值、归一化、覆盖或自动修复原始标签。
- 不在官方数据目录内部生成报告、缓存或派生数据。
- 训练时的标签映射必须在数据加载流程中动态完成，不能改写原始 PNG。
- 禁止将数据上传到 GitHub、网盘或未经授权的外部服务。
- 本目录在 Git 版本中只保留本说明文件。
