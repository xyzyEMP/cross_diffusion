# 文档导航

当前文档说明 Diffusion Planner 模型训练/导航评估流程，以及独立的 TartanGround pair mining 数据工具。通用流程可用于后续实验；目前核验的模型实验是 ANYmal 的全量微调复现。通用路径与运行命令见[实验与资源](07_experiments.md)，实验特有的输入核对、历史结果和运行状态记录在各实验页。

## 当前代码

| 文档 | 内容 |
|---|---|
| [01 项目概述](01_overview.md) | 当前能力与研究边界 |
| [02 整体架构](02_architecture.md) | 目录职责、模型入口和 pair 工具边界 |
| [03 数据流程](03_data_pipeline.md) | 固定 cache/manifest 输入与可选 pair mining |
| [04 模型结构](04_model.md) | Diffusion Planner 主干与调用 |
| [05 训练流程](05_training.md) | 训练目标、验证及 checkpoint 选择 |
| [06 评估与推理](06_evaluation.md) | 固定目标导航 rollout 和指标 |
| [07 实验与资源](07_experiments.md) | 当前实验入口与输入状态 |
| [08 开发与验证](08_development.md) | 包边界和验证说明 |

## 实验记录

- [1_全量微调复现](experiments/1_全量微调复现.md) | 已核对的固定输入、历史结果与当前重跑状态

## 历史资料

以下原文保留其历史内容与归档提示，不代表其中命令、代码路径或实验结论仍适用于当前工作树。

- [旧项目状态](archive/project_status.md)
- [旧 research-score 说明](archive/research_score_readme.md)
- [旧 TartanGround 说明](archive/tartanground_readme.md)
- [旧 pair 说明](archive/pair_readme.md)
- [旧 guidance 说明](archive/guidance_tutorial.md)
- [旧根 README](archive/root_readme.md)
- [第二轮结构整理记录](archive/second_round_cleanup.md)
