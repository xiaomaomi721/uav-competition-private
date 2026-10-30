# 已验证运行环境

以下是 Stage 02 quick pilot 和第一次提交推理实际验证过的环境快照，不要求队友盲目升级到这些版本。

| 项目 | 已验证值 |
|---|---|
| 操作系统 | Windows 10/11 64-bit |
| Python | 3.11.15 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA build | 12.8 |
| CUDA 可用 | 是 |
| transformers | 5.13.0 |
| NumPy | 2.4.4 |
| Pillow | 12.2.0 |
| PyYAML | 6.0.3 |
| 本地 GPU | NVIDIA GeForce RTX 4060 Laptop GPU，约 8 GB |

## 安装原则

- PyTorch/CUDA 应按显卡驱动和官方安装说明单独安装，不使用普通 requirements 命令覆盖已经验证的 CUDA 构建。
- 其余直接依赖记录在 requirements-segmentation.txt。
- 不要求安装未被项目导入的 torchvision、OpenCV、pandas、matplotlib 或 tqdm。
- Transformers 的传递依赖由其安装过程管理。

## Windows 建议

- num_workers=0 作为默认安全配置，避免多进程 DataLoader 在 Windows 下重复启动或卡死。
- 先运行 scripts/check_env.py 和三个专项回归测试，再启动任何训练。
- 路径通过 pathlib 和命令行参数传入，不在代码中写死本机目录。

## 算力分工

- RTX 4060：本地环境验证、单元测试、冒烟训练、配置排错和小规模实验。
- RTX 4090：Stage 03 完整训练、高分辨率实验和较大模型实验。

本文件不记录用户名、Python 可执行文件路径、数据绝对路径或模型缓存位置。
