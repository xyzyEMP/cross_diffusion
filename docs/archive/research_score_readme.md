> 历史资料：正文保留归档前内容，路径、命令、指标和完成状态不作为当前保证。当前说明见[文档索引](../README.md)。

# Research-score 代码导航
这个子包包含跨具身研究的新增逻辑，不修改原始 Diffusion-Planner 主干的基础结构。

- `configs/experiment/research_score/`：冻结的数据分割、轨迹表示和路线参数。
- `datasets/research_score/`：8 m/80点重采样、路线生成、表示桥接和分组分割。
- `models/research_score/`：`EmbodimentEncoder` 和 `ZeroResidualAdapter`，由 `ScoreDecompositionPlanner` 包装主干。
- `datasets/research_score/` and `engine/research_score/`：预先物化的 ANYmal 与 Car 特征缓存数据集。
- `evaluation/research_score/`：固定目标路线、闭环控制、碰撞、SPL 与终止条件。
- `scripts/research_score/` and `experiments/research_score/`：名字按功能划分：`build_*`、`materialize_*`、`train_*`、`evaluate_*`、`summarize_*`、`run_stage07_*`。
- `tests/research_score/`：分割、无未来信息、路线、碰撞和模型包装的测试。
正式闭环主入口是 `experiments/research_score/run_stage07_nonoverlap_eval.sh`；它只评价四种保留方法的收敛 checkpoint，并在同一批63个非重叠8 m 任务上比较。
