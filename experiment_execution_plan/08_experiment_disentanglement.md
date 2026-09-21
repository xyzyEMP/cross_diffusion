---
stage: 08
plan_version: "1.2.3"
status: NOT_STARTED
protocol_version: score-decomp-transfer-v1.2.3
depends_on: [stage_07]
profiles: [proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 08：实验二——共享意图解耦与消融

> v1.2.2 边界：当前只实现并测试接口；真实解耦实验必须等待 Strict 配对数据，不能用现有 Diff↔ANYmal 替代。

> 论文问题：共享分支是否保留任务信息，具身分支是否承担平台差异？  
> 前置：[实验一](./07_experiment_transfer.md) 已完成并选定统一主 checkpoint 规则。

## 1. 三类必做验证

### 1.1 跨具身稳定性

固定场景 $S$、目标 $G$ 和同一个 noisy query $q_t$，只替换 A/B 观测或具身标识。主指标在 x0 空间计算：

$$
D_{emb}^{x0}=\mathbb E\left[\|\hat x_{0,I}^{A}(q_t)-\hat x_{0,I}^{B}(q_t)\|_2\right].
$$

派生 score 距离按 timestep 分桶并使用与 Stage 04 一致的 clamp，只作诊断。期望本文方法的 $D_{emb}^{x0}$ 低于普通条件模型和无 `L_inv` 消融。

### 1.2 任务敏感性

固定场景 $S$ 和具身 $e$，选择同一场景中方向或通道不同的目标 $G_a,G_b$：

$$
D_{goal}^{x0}=\mathbb E\left[\|\hat x_{0,I}(q_t,G_a)-\hat x_{0,I}(q_t,G_b)\|_2\right].
$$

必须看到 $D_{goal}^{x0}$ 明显非零；否则共享分支可能输出常数。报告：

$$
R_{inv}=\frac{D_{goal}^{x0}}{D_{emb}^{x0}+\varepsilon}.
$$

本文方法应提高 $R_{inv}$，而不是仅把两种距离都压到零。

### 1.3 组件贡献和探针

- task probe：从 `z_shared` 预测目标方向和冻结 route-set 的 route mode；
- embodiment probe：从 `z_shared` 预测 A/B；
- domain probe：从 `z_shared` 预测 `dataset_id/domain_id`，识别数据域混杂；
- residual ratio：`||eps_E||/(||eps_I||+eps)`；
- branch intervention：置零、打乱或跨样本交换 shared/embodiment 分支，观察性能变化。

probe 使用冻结表示、独立线性分类器和 scene-level split。当前 Proxy 只做 synthetic 接口检查，不解释真实平台的解耦效果；Strict 阶段才分析 Car/Dog。目标是 task probe 有效，embodiment/domain probe 不再轻易识别平台/数据域，同时导航性能不下降。

## 2. 消融模型

至少包含：

1. 完整模型；
2. `-L_inv`；
3. `-L_swap`；
4. `-L_sep`；
5. `-embodiment residual`；
6. 仅 `L_diff` 的加性双分支；
7. pair label shuffle 负对照；
8. 普通 `emb_cond_diffusion`。

若计算资源有限，主消融固定目标平台 10% budget 和 5 个联合 repeats；在 1% 上复核完整模型与关键消融。缩减必须由负责人批准。

## 3. 采样协议

- 当前仅运行 synthetic fixture 的 probe/干预接口 smoke；真实采样协议在 Strict 数据到位并审计后冻结；
- 固定 5–10 个 diffusion timesteps，覆盖高噪声到低噪声；
- 每个 pair 使用相同 `q_t/t/noise`；
- 每个目标敏感性样本至少有两个有效目标；
- probe 的 train/val/test 按 scene 分组，不能让相邻窗口泄漏；
- probe 输入 detach，禁止反向训练主模型。

## 4. 代码落点

```text
tartan/research_score/evaluation/
├── score_response.py
├── probes.py
├── branch_interventions.py
├── ablations.py
└── disentanglement_report.py
tartan/research_score/scripts/run_disentanglement.sh
tartan/research_score/tests/
├── test_same_query.py
├── test_probe_split.py
└── test_branch_intervention.py
```

## 5. 执行命令

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

# v1.2.2：现有 Proxy 只执行 synthetic/interface 单元测试和
# `not_scientifically_paired` 拒绝诊断，不执行正式解耦实验。
# 严格 Car–Dog 数据就绪后，改用 strict_pair 配置执行本阶段。
```

## 6. 输出

```text
06_disentanglement/
├── score_response_per_pair.parquet
├── ablation_main_table.csv
├── ablation_main_table.md
├── probe_results.csv
├── branch_intervention.csv
├── shared_response_plot.png
├── residual_norm_by_timestep.png
├── collapse_diagnostics.json
├── run_registry.parquet
├── review_packet.md
└── stage_report.md
```

共享响应图至少展示同一场景下：换具身、换目标两种扰动对 shared score 的影响，并带配对连线或置信区间。

## 7. 验收门槛

本阶段不能只凭“`D_emb` 小”通过。必须同时满足：

- `D_emb^{x0}` 相对对照下降；
- `D_goal^{x0}` 保持明显非零；
- task probe 有效；
- embodiment probe 不再轻易识别平台；
- domain probe 结果被单独报告，不能将数据域不变等同具身不变；
- 去掉/打乱 shared 分支显著伤害目标完成；
- 去掉/打乱 embodiment 分支显著伤害平台可行性；
- 完整模型的导航主指标没有因解耦约束明显崩坏。

若 shared 常数化、residual 吞掉全部输出或 probe 识别的只是数据集来源，本阶段不得通过，应回到阶段 03/05 修正配对和 loss。

所有消融 run 无论结果好坏均标记 `retention=ablation_candidate`，完整保留 checkpoint、逐 pair/episode 指标和可视化；不得只保留完整模型或最好结果。

本阶段 agent 提交报告后停止，不自动启动 Swap 实验。
