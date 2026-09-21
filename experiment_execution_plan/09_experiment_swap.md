---
stage: 09
plan_version: "1.2.3"
status: NOT_STARTED
protocol_version: score-decomp-transfer-v1.2.3
depends_on: [stage_08]
profiles: [proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 09：实验三——Proxy/Strict 反事实交换与受控验证

> v1.2.2 边界：Proxy 仅验证 synthetic Swap 接口，不报告效果；真实反事实交换只在 Strict 数据通过审计后执行。

> 论文问题：共享意图与目标具身修正能否重新组合？  
> 前置：[阶段 08](./08_experiment_disentanglement.md) 证明分支不是常数或完全混杂。

## 1. 核心组合

当前仅在 synthetic Diff/ANYmal fixture 上生成四种组合，用于验证 Swap 的张量与公式接口，不训练模型、不报告迁移效果。未来 Strict 数据到位后保持同一组合接口并映射为真实 Car/Dog：

| 组合 | shared context | embodiment context | 评价平台 |
|---|---|---|---|
| A→A | A | A | A |
| A→B | A | B | B |
| B→B | B | B | B |
| B→A | B | A | A |

交换实验使用目标平台的动力学/几何可行性和下游控制器：A→B 用 B 约束，B→A 用 A 约束。Strict profile 才将 A/B 写成 Car/Dog。

## 2. 核心场景集

第一版只做三类受控场景：

1. **窄通道**：宽度接近平台足迹边界，检验具身修正是否改变净空与路线；
2. **急转弯**：接近 A/B 之一的曲率边界，检验 correction 是否正确响应目标平台；
3. **多路径绕障**：固定使用地图—目标 `route_set`，参考有效 mode 在查看模型输出前生成、去重并人工抽查。

连续尺寸、转弯半径及未见能力组合只作为 `extension_capability_sweep`，核心三类场景通过后另行启动，不阻塞第一版论文。

## 3. 推理规则

- 每个 pair、每个组合、每个 repeat 生成固定 `K` 条扩散样本，K 只来自 YAML；
- 四个组合共享随机种子和初始噪声集合；
- 不用 GT 轨迹重排序；
- 候选选择器和安全层在同一评价平台内保持一致；
- 同时保存 raw distribution 与最终执行轨迹；
- 若安全层改变轨迹，分别报告生成可行率和执行后成功率。

## 4. 指标定义

### 4.1 Swap Goal Success Rate

交换组合在目标平台控制器下到达共同目标的 episode 比例。

### 4.2 Trajectory Feasibility Rate

满足目标平台足迹碰撞、曲率/角速度、坡度、台阶和支撑约束的候选轨迹数除以总候选数。各约束分项同时报告，不能只给一个合成分数。

### 4.3 Valid Path Mode Coverage

对每个受控场景预先用地图与目标生成最多 6 个有效参考 route modes，并由人工抽查。生成轨迹按几何距离/障碍侧签名匹配：

$$
Coverage=\frac{\#\{\text{至少被一个可行样本覆盖的有效 mode}\}}
{\#\{\text{目标平台有效 mode}\}}.
$$

未匹配轨迹记为 unknown，不强行归类。参考 modes 在看模型输出前固定。

### 4.4 交换代价

报告 A→B 相对 B→B、B→A 相对 A→A 的 SR/SPL/可行率差值和置信区间。Proxy 只解释流程；Strict 才用于 Car–Dog 论文结论。

## 5. 代码落点

```text
tartan/research_score/evaluation/
├── swap_runner.py
├── controlled_scenes.py
├── feasibility.py
├── route_modes.py
└── swap_report.py
tartan/research_score/scripts/run_swap.sh
tartan/research_score/tests/
├── test_directional_swap.py
├── test_target_platform_constraints.py
├── test_shared_noise.py
└── test_route_mode_matching.py
```

## 6. 必须先做的单元/集成检查

- A→B 确实使用 A shared 和 B residual；
- 目标监督/评测均为 B，反向同理；
- 四种组合共享相同初始噪声；
- 换具身条件不改变 shared 输出缓存；
- route mode 与冻结的 `route_set`、candidate ID 和 mask 一致；
- 可行性判定对手工构造碰撞、超曲率和超坡度轨迹返回失败；
- Proxy/Strict pair 与训练集无交集，profile 和 claim flags 正确。

## 7. 执行命令

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

# v1.2.2：现有 Proxy 只执行定向 Swap 的 synthetic/interface 测试，
# 不产生真实性能结果。严格 Car–Dog 数据就绪后，
# 以 strict_pair 配置启动正式 Swap 实验。
```

## 8. 输出

```text
07_swap/
├── controlled_scene_manifest.json
├── reference_route_modes/
├── swap_per_candidate.parquet
├── swap_per_episode.parquet
├── swap_main_table.csv
├── swap_main_table.md
├── swap_trajectory_distributions/
├── feasibility_breakdown.png
├── mode_coverage.png
├── statistical_tests.json
├── run_registry.parquet
├── review_packet.md
└── stage_report.md
```

典型场景图必须同时画 Diff→Diff、Diff→ANYmal、ANYmal→ANYmal、ANYmal→Diff 的样本分布、障碍、目标和平台 footprint；禁止只挑成功场景。Proxy 图标题明确写 `proxy_pair_auxiliary`。

所有组合和扩散样本均为潜在消融证据，使用不可变 run ID 完整保留；失败的交换方向不得省略。

## 9. 人工验收

- 双向 swap 均执行；
- 交换后目标保持与平台可行性同时成立；
- mode coverage 的 route-set 和参考类别已在推理前冻结；
- Proxy 结果未被表述为严格因果或 Car–Dog 证据；
- 所有失败组合都有原因分解。

验收前停止，不做最终论文图表包装。
