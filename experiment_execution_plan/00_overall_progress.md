---
document: overall-roadmap
plan_version: "1.2.3"
protocol_version: score-decomp-transfer-v1.2.3
status: STAGE_03B_IN_PROGRESS
profiles: [transfer_primary, strict_pair]
requires_human_approval: true
---

# 小论文实验实施总览

> 方案来源：[飞书最新版《行为意图分布解耦：NoMaD + Diffusion-Planner + 解耦 + Swap》](https://tcn84lvthr0x.feishu.cn/wiki/KF5EwaDytiCcO9kCvEGcrFgrnjh)  
> 目标仓库：服务器上的 Diffusion-Planner 根目录，以 `local.env:PROJECT_ROOT` 为准  
> 日常进度入口：[00_dashboard.md](./00_dashboard.md)

## 1. 研究目标与当前阶段

以 nuPlan 汽车预训练的 Diffusion-Planner 为共享基座，引入“共享意图预测 + 具身残差修正”，通过一致性、定向 Swap 和分离约束学习跨具身轨迹分布。

实验分成三种明确 profile：

- `transfer_primary`：以 nuPlan Car checkpoint 为预训练源，向 TartanGround ANYmal 的 1%/10%/100% 数据预算迁移，回答普通预训练迁移和数据效率；严格配对缺失时不把完整 `L_inv/L_swap` 模型写入正式主结论。
- `proxy_pair_auxiliary`：只保留 synthetic/interface fixture、保存恢复和非科学配对诊断；当前真实 Tartan Diff↔ANYmal 不进入正式 loss、训练、主表或科学结论。
- `strict_pair`：未来严格 Car–Dog 对齐数据到位后，通过同一 dataset contract 重跑完整方法；只有该 profile 可支撑严格跨车狗一致性和反事实结论。

## 2. 最终回答的三个问题

1. 迁移：源平台数据能否提高目标平台的导航性能和数据效率？
2. 解耦：共享分支是否对换具身稳定、对换任务敏感，且具身分支承担平台差异？
3. Swap：A 的共享意图与 B 的具身分支组合后，能否保持任务目标并满足 B 的约束？

## 3. 阶段顺序

每一阶段都执行：方案冻结 → Preflight → 实现 → 五类测试 → 最小真实数据 smoke → 独立审查 → 人工确认 → 正式运行。未获人工批准不得进入下一阶段。

| 阶段 | 初始状态 | 目的 | 文件 | 人工闸门 |
|---|---|---|---|---|
| 01 | APPROVED | 固定协作、Preflight 和审查规则 | [执行契约](./01_shared_execution_contract.md)、[Preflight](./01_preflight_framework.md) | 已通过 |
| 02 | APPROVED | 冻结原模型并验证环境/仓库 | [02_protocol_and_baseline_freeze.md](./02_protocol_and_baseline_freeze.md) | 已通过 |
| 03 | APPROVED | 统一主迁移样本、固定弧长和 split；Proxy 仅作诊断 | [03_data_and_pairing.md](./03_data_and_pairing.md) | 已通过 |
| 03B | IN_PROGRESS | 验证无泄漏目标导航与 SR/CR/SPL | [03b_goal_navigation_benchmark.md](./03b_goal_navigation_benchmark.md) | 手工正负轨迹和最短路测试通过 |
| 04 | NOT_STARTED | 实现表示桥接与共享—具身分解 | [04_model_decomposition.md](./04_model_decomposition.md) | 源空间回归与新空间桥接分别正确 |
| 05 | NOT_STARTED | 实现 loss 与训练 | [05_losses_and_training.md](./05_losses_and_training.md) | tiny-set 与负对照通过 |
| 06 | NOT_STARTED | 实现公平对照和 smoke | [06_baselines_and_smoke.md](./06_baselines_and_smoke.md) | 输入、预算、计算公平 |
| 07 | NOT_STARTED | 迁移性能实验 | [07_experiment_transfer.md](./07_experiment_transfer.md) | 主迁移进入预训练迁移表；Proxy 只作辅助；Strict 后补完整方法 |
| 08 | NOT_STARTED | 解耦和消融实验 | [08_experiment_disentanglement.md](./08_experiment_disentanglement.md) | 排除共享分支常数坍缩 |
| 09 | NOT_STARTED | 双向 Swap 实验 | [09_experiment_swap.md](./09_experiment_swap.md) | 目标保持与目标具身可行同时成立 |
| 10 | NOT_STARTED | 汇总统计和复现包 | [10_results_and_reproducibility.md](./10_results_and_reproducibility.md) | 冻结数据可一键重建 |

状态只允许：`NOT_STARTED`、`READY`、`IN_PROGRESS`、`BLOCKED`、`AWAITING_REVIEW`、`APPROVED`、`SUPERSEDED`。只有负责人可以写入 `APPROVED`。

## 4. 已冻结约束

1. TartanGround 和 nuPlan 视为用户已拥有；具体路径从 `local.env` 或配置读取，Preflight 负责检查。
2. 当前预训练 checkpoint 是 `x_start` 参数化。共享与具身辅助 loss 第一版统一在 `x0` 空间计算；epsilon/score 仅作派生诊断，避免低噪声时的时间权重爆炸。
3. 源 checkpoint 的 observation/state normalizer 冻结；只为新增 ability vector 计算训练集统计。
4. 现有 Tartan `route_lanes` 来自未来真值，只能作为 oracle 回归/上界。
5. 新轨迹采用固定物理弧长并重采样为 80 点。Stage 03 仅用训练数据，在 `{8,10,12,15,20} m` 中选择最大合格值：源 Car 使用预先冻结的 50,000 条原生 NPZ 样本及单侧 95% Wilson 下界，目标 ANYmal 使用全部 train moving windows 精确覆盖率，二者均须达到 90%；val/test 不参与选择。
6. 条件固定为局部/全局目标与当前地图生成的 `route_set`，最多 6 条经拓扑去重的候选；非 oracle 构造不得访问 future GT。
7. 主迁移线固定为 nuPlan Car checkpoint→ANYmal；当前真实 Tartan Diff↔ANYmal 已因非共同任务而退出正式实验一，仅保留隔离的诊断产物。
8. 未来 strict Car–Dog 数据只替换 dataset adapter 和 manifest，不改模型、loss 与评测接口。
9. 每个预算采用 5 个“一一绑定的数据抽样 seed + 训练 seed”联合重复，不做 5×5 交叉；如以后改变，必须登记协议变更。
10. 原 checkpoint 的 8 s/80 帧时间轨迹仅用于源空间回归；新模型通过显式桥接器工作在固定弧长/80 点空间，两种表示不宣称天然等价。
11. 所有独立 run 使用不可变 `run_id`；成功、失败、中断和潜在消融均按执行契约留证。

## 5. 主实验矩阵

### 迁移实验

- 预算：1%、10%、100%，0% 仅补充诊断；
- `transfer_primary`：目标域从头训练、nuPlan 全参数微调、nuPlan Adapter、非配对联合训练、普通具身条件化 Diffusion；
- `proxy_pair_auxiliary`：不运行正式模型训练，只运行 synthetic/interface fixture 和配对前置条件失败诊断；
- `strict_pair`：未来将完整方法加入正式 Car→Dog 主表；
- 指标：SR、Collision Rate、SPL，并同步报告 Stuck、Goal Progress 和时延；
- 三条 profile 不合并作同一科学比较。

### 解耦实验

- 换具身时共享分支变化小；
- 换目标时共享分支必须明显变化；
- 增加 task/domain/embodiment probes、分支置零/打乱和 pair shuffle；
- 对抗 loss 仅在匹配场景 pair 上使用，并显式记录 `dataset_id/domain_id`，避免把数据域误当具身。

### Swap 实验

- A→A、A→B、B→B、B→A 双向运行；
- 当前使用 Proxy pair；未来 Strict profile 映射为 Car/Dog；
- route-set 协议下保留多路径 mode coverage，但候选必须由地图与目标生成并经人工审计。

## 6. 输出结构

```text
tartan/research_score/
├── configs/ data/ model/ training/ evaluation/ scripts/ tests/
└── preflight/
tartan/outputs/research_score_v1_2/
├── 00_baseline_freeze/ 01_data_audit/ 02_model_validation/
├── 02b_goal_benchmark/ 03_training_validation/ 04_baseline_smoke/
├── 05_transfer/ 06_disentanglement/ 07_swap/
└── 08_paper_package/
```

每个 run 保存 `config.resolved.yaml`、`config.sha256`、`command.txt`、`run_manifest.json`、日志、checkpoint、逐样本结果和汇总。`transfer_primary/proxy_pair_auxiliary/strict_pair` 输出根目录必须分开。输出目录存在时默认失败，只允许对同一 manifest 使用 `--resume`。

## 7. 人类控制文件

- [控制面板](./00_dashboard.md)：只看状态、证据、阻塞和待决定事项；
- [决策日志](./decisions.md)：route、pair、loss、seed 等冻结决策；
- [风险登记](./risk_register.md)：风险、责任人、触发条件和缓解措施；
- [协议变更](./protocol_changelog.md)：所有影响实验公平性的变化；
- [审查包模板](./review_packet_template.md)：负责人每阶段只看一页；
- [Agent 交接模板](./agent_handoff_template.md)：分别委派实施与独立审查，且强制阶段停止；
- [阶段报告索引](./reports/README.md)：集中登记实施报告、独立审查和真实证据路径；
- [项目 AGENTS 模板](./project_AGENTS_template.md)：批准后复制到目标仓库根目录。
- [服务器交接](./SERVER_HANDOFF.md)：环境模板、上传校验和 Stage 02 第一条指令。

## 8. 完成判定

- `transfer_primary` 完成 nuPlan→ANYmal 普通迁移主线；
- `proxy_pair_auxiliary` 完成接口 fixture 与拒绝审计；完整配对方法等待 `strict_pair` 数据后执行；
- Strict 数据到位后，以相同接口重跑正式三组实验；
- 正式实验至少 5 个联合重复，按地图/轨迹 cluster 给 95% CI；
- shared 同时满足跨具身稳定与任务敏感；
- 双向 Swap 使用目标具身约束；
- 所有主结果可由冻结配置、数据 manifest、checkpoint 和命令重建。
