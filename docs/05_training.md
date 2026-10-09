# 训练流程

唯一模型训练入口为 [`scripts/train.py`](../scripts/train.py)，执行全量微调复现使用的完整 `Diffusion_Planner` fine-tuning。该入口读取已有 train/validation cache、导航验证 manifest、source args 与 checkpoint；不会构建 split 或 cache。

## 协议

训练使用原始 diffusion loss，scheduler 依据 validation loss 更新，checkpoint 依据 navigation validation 选择。通用数据和输出路径及 CLI 命令见[实验与资源](07_experiments.md)。实验特有的参数、固定输入核对、历史结果与运行状态记录在[1_全量微调复现记录](experiments/1_全量微调复现.md)，避免在通用流程页维护特定实验数据。

## 训练与 pair 输入的关系

训练路径不调用 `datasets.pair`。该工具独立生成候选 pose 窗口；当前没有 pair 数据训练器、identity classifier 或跨具身辅助损失，不能从 pair mining 工具存在推断配对训练已经实现。pair 输入生成方式见 [03 数据流程](03_data_pipeline.md#独立-pair-mining)。

训练共享函数位于 [`engine/training_utils.py`](../engine/training_utils.py)，范围限于 checkpoint load/publish、diffusion loss 和 validation。模型结构见 [04 模型](04_model.md)，评估方式见 [06 评估](06_evaluation.md)。
