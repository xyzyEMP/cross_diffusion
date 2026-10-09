# 项目概述

当前工作树提供 Diffusion Planner 全量微调和固定目标导航评估流程，以及一套可独立运行的 TartanGround pose 配对窗口挖掘工具。模型运行入口只有 [`scripts/train.py`](../scripts/train.py) 与 [`scripts/evaluate.py`](../scripts/evaluate.py)。通用流程可用于后续实验；目前核验的模型实验是 ANYmal 的全量微调复现。pair 工具只生成候选 pair 数据，不接入当前训练过程。

## 当前范围

- 训练直接使用 `Diffusion_Planner`，全模型 fine-tuning，训练目标为原始 diffusion loss。
- 数据运行路径使用已物化的 target cache、现有 validation/test manifests、路线构造、feature 和 normalizer。
- 模型评估执行固定 goal 导航并输出任务级与 episode-macro 指标；它不是物理仿真或真实机器人安全评估。
- `datasets.pair` 从 TartanGround pose 中独立生成跨 embodiment 的局部轨迹窗口候选，可用于后续数据检查。当前没有 paired-training、identity classifier 或跨具身辅助 loss 训练流程。

## 边界

本仓库不再提供 score wrapper、method factory、辅助 loss、guidance、preflight、proxy workflow、legacy dataset builders 或 reference baseline workflow。模型网络本身在这次精简中没有改变。通用数据、checkpoint 和 runs 路径见[实验与资源](07_experiments.md)；已核对的固定输入、参数、历史结果和当前重跑状态见 [1_全量微调复现记录](experiments/1_全量微调复现.md)。

模型、调用结构、数据、训练、评估和开发说明分别见 [04 模型](04_model.md)、[02 架构](02_architecture.md)、[03 数据](03_data_pipeline.md)、[05 训练](05_training.md)、[06 评估](06_evaluation.md)和[08 开发](08_development.md)。
