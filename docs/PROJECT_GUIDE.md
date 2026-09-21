# Cross-Diffusion 项目总览：模型、数据、结果与下一步

> 读者对象：希望不打开每个代码文件，仍能理解这个项目在做什么、结果能说明什么、以及接下来如何继续的人。

## 1. 一句话目标

本项目研究：**nuPlan 自动驾驶汽车的扩散轨迹规划知识，能否迁移到 TartanGround 中的 ANYmal 导航任务。**
模型的输出是机体级 SE(2) 轨迹（x、y、yaw），不涉及足端、关节或电机控制。闭环价值是静态地图上的简化运动学 rollout，不等价于真实机器人部署性能。

## 2. 从输入到输出的数据流

```text
nuPlan Car checkpoint
        │
        ├── 加载为 Diffusion-Planner 主干
        │
ANYmal 当前位姿 + 当前占据地图 + 固定目标
        │
        ├── 构建最多6条 route candidates（route_set）
        │
        ├── 转为 route_lanes 和其他主干输入
        │
        ├── 扩散去噪网络输出共享轨迹
        │                         └── 可选的具身残差修正
        │
        └── 80点 SE(2) 路径 → 简化控制器 → 闭环指标
```

### 2.1 轨迹表示

原始 Car 预训练使用 8 s/80点的时间表示。为了让 Car 与 ANYmal 在同一研究空间比较，两者都被转为**前8 m的固定弧长路径，再重采样为80点**。
对于不足8 m的未来轨迹，`valid_mask` 标出真实可用的点；不用填充部分伪造监督信号。关键函数是 `tartan/research_score/data/core.py:resample_fixed_arc` 。

### 2.2 无未来信息输入

每次重规划只读取：

- 当前 ANYmal 位姿；
- 当前占据地图；
- 离线冻结的固定目标。

未来真实轨迹只用于离线定义目标和训练标签，不在预测时读取。`route_set` 从当前地图和固定目标重建，最多6条候选路线。

## 3. 数据划分和预算

全24条 ANYmal episode 先按完整 episode 完成固定切分：

| 集合 | episode | 密集窗口 | 用途 |
|---|---:|---:|---|
| Train | 16 | 2,048 | 模型训练 |
| Validation | 3 | 384 | checkpoint 选择 |
| Test | 5 | 639个原始滑窗 | 构建闭环任务 |

三个集合的 episode 交集为0。训练时每1秒生成一个滑窗，这样能充分利用少量目标域数据；它们不被当作独立统计样本。
预算仅指 ANYmal **训练集**的名义预算：

| 名义预算 | 实际训练 episode | 实际窗口 |
|---|---:|---:|
| 1% | 1 | 128 |
| 10% | 2 | 256 |
| 100% | 16 | 2,048 |

这些比例之所以不等于精确的1%和10%，是因为完整 episode 不能拆分。验证集和测试集对所有预算都不变。

## 4. 四种比较方法

| 方法 | 主干 | 训练 batch | 具身模块 |
|---|---|---|---|
| `pretrain_finetune` | 加载 Car checkpoint后全参数更新 | ANYmal | 无 |
| `pretrain_adapter` | 冻结且保持 eval | ANYmal | 只训练 embodiment encoder + residual adapter |
| `joint_train` | 全参数更新 | ANYmal update 后加一次 Car update | 无 |
| `emb_cond_diffusion` | 全参数更新 | 同上 | Car/ANYmal ID + 能力向量 + residual correction |

### 具身条件位于哪里

`ScoreDecompositionPlanner` 包装原始主干：

```text
shared = DiffusionPlanner(...).score
z_emb = EmbodimentEncoder(platform_id, ability)
delta = ZeroResidualAdapter(shared, z_emb)
total = shared + delta
```

`EmbodimentEncoder` 将离散 platform ID embedding 与10维能力向量编码相加，产生64维条件。`ZeroResidualAdapter` 将每个4维轨迹状态与该条件合并，输出4维残差。它的最后一层零初始化，所以初始时不会破坏预训练主干。
训练阶段的残差与主干输出相加后计算 loss；推理阶段它通过 `score_correction_fn` 注入采样器，因此参与每一步去噪，不是生成结束后才修正轨迹。

## 5. 训练协议

- batch size 64；
- 最少5,000次、最多10,000次 ANYmal optimizer updates；
- 每250次在384个 validation windows 上评估；
- validation 处于 `eval()` 模式，因此 checkpoint 选择不受 dropout/DropPath 干扰；
- 连续10次验证无改善时早停，使用 ReduceLROnPlateau；
- 联合方法每个 ANYmal update 后追加一个 Car update；Car 从固定4096条特征缓存采样，不计入 ANYmal 预算。

## 6. 正式闭环评价

旧测试会将每条 episode 压缩为一个最远目标，只产生5个闭环结果。v1.2.4 改为在5条 held-out episode 内按累计弧长连续分段：

- 每段的目标为前8 m；
- 段与段不重叠；下一段从上一段终点之后的首个10-frame地图锚点开始；
- 共得到63个任务（P2001: 9, P2003: 14, P2012: 16, P2015: 11, P2018: 13）；
- 每段独立计算 SR、CR、SPL、goal progress、route failure和推理时延；
- 主表报告段级均值，另报告5条 episode 等权宏平均。

段不重叠使指标的分辨率从原来的1/5=20%提高到1/63≈1.6%，但它们仍来自5条 episode和同一地图；因此不把63段声称为63个完全独立场景。

`astar_disconnected` 任务按 D024 统一安全停车，记为 `route_failure`，保留在 SR/CR/SPL 分母中且不计碰撞。

## 7. seed 11 v3 结果

| 方法 | 预算 | 训练窗口 | SR | CR | SPL | episode-macro SR |
|---|---:|---:|---:|---:|---:|---:|
| pretrain_finetune | 1% | 128 | 0.5556 | 0.1429 | 0.5001 | 0.5556 |
| pretrain_finetune | 10% | 256 | 0.4921 | 0.1905 | 0.4705 | 0.4823 |
| pretrain_finetune | 100% | 2048 | **0.6349** | 0.1746 | **0.5966** | **0.6297** |
| pretrain_adapter | 1% | 128 | 0.3492 | 0.2222 | 0.3492 | 0.3626 |
| pretrain_adapter | 10% | 256 | 0.2222 | 0.3016 | 0.2222 | 0.2317 |
| pretrain_adapter | 100% | 2048 | 0.2698 | 0.2540 | 0.2698 | 0.2846 |
| joint_train | 1% | 128 | 0.4603 | 0.1746 | 0.4417 | 0.4624 |
| joint_train | 10% | 256 | 0.4762 | 0.1746 | 0.4413 | 0.4669 |
| joint_train | 100% | 2048 | 0.3810 | 0.2222 | 0.3662 | 0.3934 |
| emb_cond_diffusion | 1% | 128 | 0.5079 | 0.1587 | 0.4804 | 0.5096 |
| emb_cond_diffusion | 10% | 256 | 0.1270 | 0.2381 | 0.0961 | 0.1258 |
| emb_cond_diffusion | 100% | 2048 | 0.5714 | 0.1587 | 0.5147 | 0.5573 |

四种方法的 `route_failure_rate` 都是 10/63 = 0.1587，这是任务图与路线可达性的共同边界，不是方法间性能差异。

### 如何解读

1. **全量微调是当前最稳定的方法。** 它在1%、10%和100%下都有可用的闭环结果，100%预算最好。
2. **具身条件扩散不是单调收益。** 1%和100%有竞争力，10%明显不稳定，现阶段只能说明条件机制可运行且值得继续诊断。
3. **无条件联合训练在目标域数据充足时出现退化。** 这表明交替 Car update 可能与 ANYmal 适应冲突，不能自动等价为更好的迁移。
4. **冻结 adapter 不足。** 它的 validation loss 可收敛，但闭环不如全量微调，说明目标域差异不能只靠小型输出残差修正解决。

## 8. 还不能声称什么

- 当前只有 seed 11，不能报告训练随机性的稳定性结论。
- 测试来自同一地图中的5条 episode，不支持 unseen-map 泛化声明。
- 63个段不重叠，但仍有 episode/map 聚类相关性。
- 闭环是简化运动学价值，不代表真实机器人部署。

## 9. 下一步计划

1. 以 `pretrain_finetune` 作为稳定参照，分析 `joint_train` 在100%下退化的来源（Car/ANYmal 更新比例、学习率、梯度冲突）。
2. 针对 `emb_cond_diffusion` 的10%运行诊断，检查 checkpoint 选择和分 episode 闭环失败模式。
3. 扩充真正独立的 ANYmal episode，优先增加新地图；这比仅增加同一数据集的 seed 更能增强外部有效性。
4. 在增加数据后，使用不变的非重叠8 m 闭环测试流程重复评价，以便新结果可与当前结果直接比较。

## 10. 关键文件

| 目的 | 文件 |
|---|---|
| 正式训练器 | `tartan/research_score/scripts/train_method_formal.py` |
| 具身模型包装 | `tartan/research_score/model/score_decomposition.py` |
| 具身编码器 | `tartan/research_score/model/embodiment.py` |
| 残差 adapter | `tartan/research_score/model/adapters.py` |
| 非重叠段构建 | `tartan/research_score/scripts/build_nonoverlap_segments.py` |
| 非重叠闭环评价 | `tartan/research_score/scripts/evaluate_nonoverlap_segments.py` |
| 闭环指标 | `tartan/research_score/evaluation/metrics_navigation.py` |
| 正式协议 | `experiment_execution_plan/07_experiment_transfer.md` |
| 复现命令 | `run.md` |
