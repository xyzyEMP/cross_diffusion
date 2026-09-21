# Research-score 代码导航
这个子包包含跨具身研究的新增逻辑，不修改原始 Diffusion-Planner 主干的基础结构。

- `configs/`：冻结的数据分割、轨迹表示和路线参数。
- `data/`：8 m/80点重采样、路线生成、表示桥接和分组分割。
- `model/`：`EmbodimentEncoder` 和 `ZeroResidualAdapter`，由 `ScoreDecompositionPlanner` 包装主干。
- `training/`：预先物化的 ANYmal 与 Car 特征缓存数据集。
- `evaluation/`：固定目标路线、闭环控制、碰撞、SPL 与终止条件。
- `scripts/`：名字按功能划分：`build_*`、`materialize_*`、`train_*`、`evaluate_*`、`summarize_*`、`run_stage07_*`。
- `tests/`：分割、无未来信息、路线、碰撞和模型包装的测试。
正式闭环主入口是 `scripts/run_stage07_nonoverlap_eval.sh`；它只评价四种保留方法的收敛 checkpoint，并在同一批63个非重叠8 m 任务上比较。
