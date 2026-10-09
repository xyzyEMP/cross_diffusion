# 开发与验证

文档与实现应保持当前收敛范围：两个模型入口、完整 Diffusion Planner、缓存/路线/normalizer 数据路径，以及独立 pair mining。新增实验方法或改变 split、数据、模型前向、损失、seed、指标时，应先确认研究语义和复现证据，再更新对应入口与文档。

## 包边界

- `scripts/train.py`、`scripts/evaluate.py` 是模型运行入口。
- `engine/training_utils.py` 提供 checkpoint load/publish、diffusion loss 与 validation。
- `datasets/` 保存缓存 Dataset、preprocessing、route、TartanGround pose/features 和独立 pair 工具。
- `models/` 保存未修改的 Diffusion Planner 网络。
- `evaluation/` 定义固定目标导航、碰撞几何及指标。

当前不保留 score wrapper/loss、method factory、guidance、preflight、proxy 或 reference-baseline 流程。环境和模型依赖以 [`requirements_torch.txt`](../requirements_torch.txt)、[`setup.py`](../setup.py) 与运行环境现状为准；本轮文档更新未运行安装、测试、训练或评估。

## 测试说明

测试源码仍保留在 `tests/`，覆盖核心特征/模型回归、导航指标、route 构造和 replanning 等路径。pair mining 的输入数据位于外部，文档更新不需要生成 pair 数据来验证链接或协议说明。仅改写文档不会运行项目测试；需要检查实现时，按对应测试文件及当前环境选择命令，并单独记录是否访问外部数据或 GPU。
