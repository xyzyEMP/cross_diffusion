# 模型结构

当前模型入口直接构造 `Diffusion_Planner`：训练从 source checkpoint 初始化整个网络并 fine-tune；评估将导航 checkpoint 严格加载到同一模型类。主干文件、参数化、前向路径和 state-dict 架构在精简中没有更改。

## 组成

[`models/diffusion/planner.py`](../models/diffusion/planner.py) 连接场景 encoder 与 diffusion decoder。encoder 处理 ego 当前状态、邻车历史、静态物体、lane 和 route 特征；decoder 通过 DiT 在场景与路线条件下预测/去噪轨迹，采样实现位于 `models/diffusion/sampling.py` 与 SDE/solver 模块。

模型构造参数与 state/observation normalizer 由 CLI 传入的 args JSON 提供；source checkpoint 路径见[实验与资源](07_experiments.md)。Experiment 1 已核实的具体权重文件及加载证据见[实验记录](experiments/experiment1.md)。

## 运行边界

训练使用原始 diffusion loss 对完整 `Diffusion_Planner` 更新参数。导航评估直接调用模型生成预测轨迹，再交由路线控制循环和几何指标计算。当前代码不在模型外包 score decomposition、adapter、guidance 或 proxy encoder；pair mining 仅生成候选输入记录，不改变模型前向。

模型前向细节及网络配置不在本轮文档变更中修改。张量字段与 route/feature 来源见 [03 数据](03_data_pipeline.md)，训练选择规则见 [05 训练](05_training.md)，评估过程见 [06 评估](06_evaluation.md)。
