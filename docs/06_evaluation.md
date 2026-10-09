# 评估与推理

唯一模型评估入口为 [`scripts/evaluate.py`](../scripts/evaluate.py)。它从 test manifest 读取 fixed-goal navigation tasks，加载 `Diffusion_Planner` checkpoint，运行基于 occupancy 和路线重建的闭环 rollout，并写出 `per_segment.csv` 与 `summary.json`。当前不保留 reference-baseline workflow。

## 任务与路径

每条 manifest 记录对应一个独立任务。评估按当前状态和固定 goal 生成 route features，调用模型预测轨迹并逐段推进；碰撞、成功、停滞或路线失败由几何规则判定。它不运行机器人动力学、物理仿真或关节控制。通用路径和 CLI 参数见[实验与资源](07_experiments.md)；全量微调复现的测试 manifest 和输入核对见[实验记录](experiments/1_全量微调复现.md)。

每次运行使用新输出目录，实际命令见[实验与资源](07_experiments.md)。

## 指标边界

`evaluation/metrics_navigation.py` 聚合 SR、CR、SPL、goal progress 等任务指标；逐 segment 与按 episode 等权的 episode-macro 汇总应按各自定义报告。地图无效任务的分母处理和任务结果以评估实现与输出为准。几何指标用于该固定目标 benchmark，不是现实机器人导航、安全或动力学能力保证。
