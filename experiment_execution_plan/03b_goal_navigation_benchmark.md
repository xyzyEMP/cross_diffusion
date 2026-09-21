---
stage: 03B
plan_version: "1.2.3"
status: AWAITING_HUMAN_APPROVAL
protocol_version: score-decomp-transfer-v1.2.3
depends_on: [stage_03]
profiles: [transfer_primary, proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 03B：无泄漏目标导航 Benchmark Gate

> 前置：[阶段 03](./03_data_and_pairing.md) 的数据、固定弧长和 route-set 已获人工确认。  
> 本阶段只验证任务与指标，不训练研究模型。未通过时不得进入 Stage 04–07。

## 1. 阶段目标

将当前依赖记录未来路线的静态运动学评测替换为固定目标的滚动规划协议，并用解析可验证的正负样例证明 SR、Collision Rate、SPL、Stuck 和 Goal Progress 的实现正确。

## 2. Episode 契约

episode 创建时保存一次固定世界坐标目标 `goal_global`。允许从日志选择一个满足固定弧长要求的终点作为任务定义，但 rollout 开始后：

- 不得读取 anchor 之后的 GT 位姿、速度或路线构造输入；
- 每次重规划只使用当前状态、历史观测、当前地图、固定目标和平台能力；
- `RouteSetBuilder` 从当前地图到目标重新生成最多 6 条候选并保留 route ID；
- 执行器只执行预测轨迹前缀，再用执行后状态重新规划；
- oracle-route 只在独立 `oracle_upper_bound` profile 中运行，输出不得进入主指标。

终止条件固定写入配置：到达目标、碰撞、地形硬失败、连续无进展或超时。目标半径、无进展窗口、控制频率、重规划间隔和最大 episode 步数只从 YAML 读取。

`astar_disconnected` 统一触发安全停止，终止原因记为 `route_failure`；该 episode 保留在 SR/CR/SPL 分母中，不计为碰撞，且所有方法使用完全相同的处理。

## 3. 指标定义

$$
SR=\frac{1}{N}\sum_i S_i,
\qquad
CR=\frac{1}{N}\sum_i \mathbb 1[\text{episode }i\text{发生碰撞}],
$$

$$
SPL=\frac{1}{N}\sum_i S_i\frac{\ell_i}{\max(\ell_i,p_i)}.
$$

其中 $\ell_i$ 是目标平台可行图上的最短路径长度，$p_i$ 是实际执行路径长度。不可计算最短路的 episode 标为 `invalid_map_episode` 并进入拒绝清单，不能静默删除。同步报告 Stuck Rate、Goal Progress、Terrain Failure、终止原因和有效/拒绝分母。

## 4. 需要新增的代码

```text
tartan/research_score/evaluation/
├── goal_benchmark.py
├── episode.py
├── route_set_runtime.py
├── shortest_path.py
├── collision.py
├── termination.py
└── metrics_navigation.py
tartan/research_score/tests/
├── test_goal_no_future.py
├── test_navigation_metrics.py
├── test_shortest_path.py
├── test_collision_geometry.py
├── test_termination.py
└── test_route_replanning.py
tartan/research_score/scripts/run_stage03b.sh
```

现有 `tartan/evaluation/closed_loop.py` 保持为 oracle 历史基线；新 benchmark 独立实现，禁止通过改名掩盖其 future GT 依赖。

## 5. 必须通过的解析测试

1. 空旷直线、成功到达且走最短路：`SR=1, CR=0, SPL=1`；
2. 成功但绕远一倍：`SR=1, CR=0, SPL=0.5`；
3. 中途碰撞：`SR=0, CR=1, SPL=0`；
4. 原地停止到超时：`SR=0, Stuck=1, SPL=0`；
5. 改变 future GT：输入、route-set、rollout 和指标逐字段不变；
6. 改变固定目标：route-set 和 Goal Progress 必须变化；
7. 已知小图的 Dijkstra/A* 最短路长度与手算一致；
8. 目标平台 footprint 改变时碰撞/可行图按预期变化；
9. route-set 候选重新规划不沿用未来日志 route；
10. 相同 episode、seed 和控制器连续运行结果一致。

## 6. 最小真实数据 Smoke

从 held-out 之外的开发地图选择至少：

- 5 个空旷/正常目标；
- 5 个窄通道；
- 5 个坡度或台阶附近目标；
- Diff 与 ANYmal 均运行，以核验 footprint 和地形可行图差异。

先运行 `shortest_path_follower`、`intentional_collision_policy`、`stop_policy` 三个无学习策略，再运行现有模型的 oracle 上界作边界参考。所有策略使用相同 episode manifest。

## 7. 执行命令

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

$PYTHON_BIN -m tartan.research_score.preflight \
  --stage 03B --profile transfer_primary \
  --config tartan/research_score/configs/stage03b_goal_benchmark.yaml \
  --report "$OUTPUT_ROOT/preflight/stage03b_goal_benchmark"

$PYTHON_BIN -m pytest \
  tartan/research_score/tests/test_goal_no_future.py \
  tartan/research_score/tests/test_navigation_metrics.py \
  tartan/research_score/tests/test_shortest_path.py \
  tartan/research_score/tests/test_collision_geometry.py \
  tartan/research_score/tests/test_termination.py \
  tartan/research_score/tests/test_route_replanning.py -q

bash tartan/research_score/scripts/run_stage03b.sh \
  --config tartan/research_score/configs/stage03b_goal_benchmark.yaml \
  --output "$OUTPUT_ROOT/02b_goal_benchmark"
```

## 8. 产物与人工验收

```text
02b_goal_benchmark/
├── config.resolved.yaml
├── episode_manifest.parquet
├── analytic_cases.json
├── per_episode_metrics.parquet
├── rejected_episodes.parquet
├── shortest_path_audit.json
├── no_future_invariant.json
├── visualizations/
├── review_packet.md
└── stage_report.md
```

负责人只在解析测试全部通过、真实 smoke 可解释、拒绝分母透明、修改 future GT 不改变任何非 oracle 结果时批准。批准前 agent 停止，不进入模型实现。
