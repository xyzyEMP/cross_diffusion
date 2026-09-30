# Proxy Training Protocol

本文档定义两个跨 embodiment 实验。两者均使用 Omni 和 Diff 训练、ANYmal
作为完全未见的测试 embodiment；实验 B 在实验 A 的基础上加入 Diff–Omni 配对数据
和因果模块。

当前状态：**proposed**。仓库尚未接通对应的数据 manifest、训练入口和完整测试流程。

## 1. 统一数据划分

| Embodiment | Train | Validation | Test |
|---|---:|---:|---:|
| Omni | 80% | 20% | 0 |
| Diff | 80% | 20% | 0 |
| ANYmal | 0 | 0 | 100% |

Omni 和 Diff 分别以完整 trajectory 为单位做确定性 80/20 划分，再合并各自的
train 和 validation。划分 seed 固定为 `20260911`。

```text
Omni-train + Diff-train -> train
Omni-val   + Diff-val   -> validation
全部 ANYmal             -> unseen-embodiment test
```

禁止按 frame、局部窗口或 segment 随机切分。必须先冻结 trajectory manifest，再从
各 split 内生成窗口或配对；同一 trajectory 的所有派生样本只能属于一个 split。

每个 embodiment 的取整规则为：

```text
n_val = max(1, round(0.2 * n_trajectories))
n_train = n_trajectories - n_val
```

现有文档记录 Diff 5 条、Omni 6 条 trajectory；若实际数据不变，对应为 Diff 4/1、
Omni 5/1。正式实验前必须按实际目录重新核验。

## 2. 实验设计

| 设置 | A：Baseline | B：Causal Pair |
|---|---|---|
| 训练/验证划分 | Omni + Diff trajectory split | 与 A 完全相同 |
| ANYmal test | 全部 trajectory | 与 A 完全相同 |
| 模型 | 普通 `Diffusion_Planner` | `ScoreDecompositionPlanner` |
| 训练样本 | 非配对窗口 | 相同基础样本 + Diff–Omni train pairs |
| 损失 | `L_diff` | `L_diff + λ_inv L_inv + λ_swap L_swap + λ_sep L_sep` |
| Embodiment 输入 | 无 | ID + ability vector |

### 2.1 实验 A：Diffusion Planner Baseline

使用普通 [`Diffusion_Planner`](../diffusion_planner/model/diffusion_planner.py)，合并
Omni-train 和 Diff-train 进行训练。模型不接收 embodiment ID，不使用配对数据或因果
辅助损失。

验证集为 Omni-val 与 Diff-val 的合并集。checkpoint 按合并验证集最低 masked
denoising loss 选择，同时分别报告 Omni 和 Diff 验证损失。

该实验衡量普通 Diffusion Planner 从 Omni+Diff 到 ANYmal 的跨 embodiment baseline。

### 2.2 实验 B：Diff–Omni Pair + Causal Module

实验 B 使用与 A 完全相同的基础训练样本，并加入仅由训练 trajectory 生成的
Diff–Omni 配对数据。模型分解为：

```text
total = shared + embodiment_residual
```

其中 `shared` 来自 Diffusion Planner backbone；residual 由 embodiment ID 和 ability
vector 条件化，并以零初始化 adapter 加到模型输出。

损失为：

```text
L = L_diff + λ_inv L_inv + λ_swap L_swap + λ_sep L_sep
```

- `L_diff`：两侧轨迹的 masked denoising loss；
- `L_inv`：配对样本 shared 输出的一致性；
- `L_swap`：交换 Diff/Omni residual 后的预测一致性；
- `L_sep`：形态分离项，包括外部 adversarial loss 与 residual 正则化。

实现接口位于
[`score_decomposition.py`](../tartan/research_score/model/score_decomposition.py) 和
[`losses.py`](../tartan/research_score/training/losses.py)。当前 adversarial classifier、
真实配对训练入口及损失权重尚未完成，因此 B 仍是待实现协议。

## 3. Diff–Omni 配对约束

- 只允许从 `Diff-train × Omni-train` 构造训练 pair；
- validation pair 的两侧必须都来自各自 validation trajectory，且不得用于训练；
- 不允许任何 pair 跨越 train/validation；
- 每个 pair 必须保存两侧 `trajectory_key`、segment ID 和匹配指标；
- 重叠窗口不能被视为独立 trajectory 样本；
- 配对数据只给实验 B 使用，实验 A 不读取 pair manifest。

Diff–Omni pair 由几何相似轨迹构成，属于 observational matching，不等价于真实
intervention 或 counterfactual pair。

## 4. 共同训练设置

两项实验必须保持以下设置一致：

- 相同 trajectory manifest、基础窗口和 ANYmal test manifest；
- 相同 8 m / 80 点轨迹表示、坐标系、上下文特征和 normalizer；
- 相同 nuPlan checkpoint 初始化；
- 相同 batch size、优化器、学习率、更新次数和随机种子；
- 相同验证规则和 checkpoint 选择标准；
- 相同基础 denoising sample 顺序。

默认沿用 [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)：batch size 64、AdamW、
learning rate `1e-4`、最多 10,000 次 target updates、每 250 次验证、至少训练
5,000 次，并在连续 10 次验证无改善后早停。

实验 B 必须单独报告 pair 数量、辅助损失权重和额外计算量。

## 5. ANYmal Unseen-Embodiment Test

全部 ANYmal trajectory 仅用于最终测试，不参与训练、验证、normalizer 拟合、
checkpoint 选择或阈值调整。

离线测试至少报告：

- masked denoising loss；
- 有效点 ADE/FDE；
- trajectory-macro 主指标；
- 全窗口 micro 指标和逐 trajectory 结果。

若已生成兼容的冻结闭环任务，再报告 SR、Collision Rate、SPL、Goal Progress、
Stuck Rate 和 Route-Failure Rate。实验 A 与 B 必须使用完全相同的 ANYmal test
manifest。

## 6. 结果解释

A/B 对比可以检验：加入 Diff–Omni 配对和因果模块后，ANYmal unseen-embodiment
测试是否优于普通 Diffusion Planner baseline。

仅凭该实验不能声称：

- 已识别真实 embodiment 因果效应；
- 已实现物理 intervention 或真实反事实预测；
- 已验证 unseen-map 或真实机器人泛化。

当前还需实现 Omni/Diff/ANYmal 统一 loader、trajectory manifest builder、Diff–Omni
pair 数据接口、实验 B 的真实配对训练入口，以及全量 ANYmal 冻结测试 manifest。
