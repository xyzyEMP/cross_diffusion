---
stage: 07
plan_version: "1.2.4"
status: AWAITING_REVIEW
protocol_version: score-decomp-transfer-v1.2.4
depends_on: [stage_06]
profiles: [transfer_primary, proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 07：实验一——跨具身迁移性能

> v1.2.2 边界：实验一正式结果仅来自 `transfer_primary`；Proxy 只输出接口测试与拒绝诊断，不运行训练、不进入表格。

> 主迁移问题：nuPlan Car 预训练能否改善 ANYmal 的少样本导航性能？  
> Proxy 辅助问题：synthetic/interface fixture 能否通过张量、损失和交换接口测试？当前真实 Diff↔ANYmal 不构成共同任务配对，因此不运行完整方法训练与效果评测。

## 1. 三条隔离矩阵

### 1.1 `transfer_primary`（当前主结果）

- 方法：`pretrain_finetune/pretrain_adapter/joint_train/emb_cond_diffusion`；
- 方向：nuPlan Car checkpoint→ANYmal；
- ANYmal 预算：1%、10%、100%，均指完整 ANYmal 训练集的名义预算；因完整 episode 约束，实际为 1/2/16 条训练 episode 和 128/256/2048 个密集滑窗；
- 当前固定 seed 11，共 `4×3=12` 个正式训练 run；
- 不启用 Diff↔ANYmal Proxy 的 `L_inv/L_swap`。

### 1.2 `proxy_pair_auxiliary`（仅接口诊断）

- 方法：只运行 synthetic fixture 上的前向、反向、交换和保存恢复测试，不建立正式 method comparison；
- 数据：人工构造的接口 fixture；真实 Tartan Diff+ANYmal 只保留 `not_scientifically_paired` 拒绝审计；
- ANYmal 预算：1%、10%、100%；5 个联合 repeat IDs；
- 共 `3×3×5=45` 个正式 run；
- 结果单独成表，只解释完整方法管线与代理消融。

### 1.3 `strict_pair`（未来正式完整方法）

严格 Car–Dog 数据到位后运行七种方法 × 三预算 × 五重复，共 105 个 run。当前保持 `BLOCKED_WAITING_DATA`。

所有矩阵测试固定 held-out maps，checkpoint 只由 validation 选择。`source_zero_shot` 作为 0% 补充诊断。Preflight 根据 smoke 估算 GPU-hours 和磁盘；资源不足只能提交统一协议变更。

## 2. 数据划分与目标导航评测协议

- 先按完整 episode 固定 train/validation/test = 16/3/5，即 66.7%/12.5%/20.8%；三者 episode 交集必须为零。
- 训练保留每 1 s 一个锚点的密集滑窗，将每个未来轨迹的前 8 m 重采样为 80 点，用于充分利用有限目标域数据。
- validation 保留 384 个密集窗口的 diffusion loss 用于 checkpoint 选择，另构建 38 个非重叠8 m 段用于轨迹质量诊断。
- 测试在 5 条 held-out episode 内按累计弧长依次切分非重叠8 m 段；下一段从前一段终点之后的首个 10-frame 地图锚点开始，当前实际共 63 个闭环任务；尾部不足8 m 的段不进入主评价。
- 主表报告 63 个段级任务的 SR/CR/SPL，同时报告 5 条 episode 等权宏平均；不将同一 episode 内的段声称为完全独立场景。

formal profile 只有在 [Stage 03B](./03b_goal_navigation_benchmark.md) 批准后才能启动。每个 episode 固定目标，后续不读取未来轨迹；每次重规划从地图与目标重建最多 6 条 route-set。以下全部配置化：

- 控制频率、重规划间隔和固定长度轨迹到时间执行的转换；
- 成功：episode 结束前进入固定目标半径，默认 1.0 m；
- 碰撞：足迹与不可通行区域相交，episode collision 记 1；
- 失败：超时、碰撞终止、地形失败或连续无进展；
- 最短路径长度由同一评测地图上的中性可通行图计算；
- 控制器和安全截停对所有方法相同。

四种方法都在同一批 63 个 ANYmal held-out 非重叠段上评价；SPL 的最短路径使用 ANYmal 可行图，因此同一测试段的分母一致。

Stage 03B 已用无学习策略验证手工直线路径、绕远路径、碰撞路径和停止路径的 SR/CR/SPL，并证明改变 future GT 不改变非 oracle 输入。

## 3. 指标计算

成功率：

$$
SR=\frac{1}{N}\sum_{i=1}^{N}S_i.
$$

碰撞率：

$$
CR=\frac{1}{N}\sum_{i=1}^{N}\mathbb 1[\text{episode }i\text{发生碰撞}].
$$

SPL：

$$
SPL=\frac{1}{N}\sum_{i=1}^{N}S_i\frac{\ell_i}{\max(\ell_i,p_i)}.
$$

其中 $\ell_i$ 是最短可行路径长度，$p_i$ 是实际执行路径长度。同步报告 Stuck Rate、Goal Progress、完成时间和推理时延，避免“靠停止降低碰撞”。

## 4. 运行策略

### 4.1 先跑正式矩阵的小门槛

当前正式比较固定为 4 方法×3预算的 seed 11，共 12 个 run。Proxy 已退出正式实验一，不进入训练矩阵。必须同时报告 validation loss 收敛与统一 goal-conditioned 闭环的 SR、CR、SPL，不能只按 denoising loss 排名。

seed 11 v3 的 12 项保留方法训练已完成且无错误。正式训练按 ANYmal optimizer update 对齐：最多 10,000 次、至少 5,000 次，每 250 次在完整 384-window validation set 上评价，连续 10 次无改善早停；联合方法额外使用相同数量的 Car 更新。全量微调、联合训练与条件扩散在 5,000 updates 时已满足早停；adapter 在 5,750–8,500 updates 停止且 best checkpoint 已保存。这 12 个 checkpoint 与 v1.2.4 的训练数据协议兼容，不需重训；只需用 63 个非重叠8 m 任务重新评价。

v1.2.4 的 12/12 项非重叠8 m 闭环评价已完成。全量微调在 1%/10%/100% 下的 SR 为 0.5556/0.4921/0.6349，整体最稳定；具身条件扩散为 0.5079/0.1270/0.5714，10% 预算存在明显退化；无条件联合训练为 0.4603/0.4762/0.3810；冻结 adapter 为 0.3492/0.2222/0.2698。四种方法的 route failure rate 均为 10/63=0.1587，按 D024 统一保留在分母中。

### 4.2 完整运行

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

$PYTHON_BIN -m tartan.research_score.preflight \
  --stage 07 --profile transfer_primary \
  --config tartan/research_score/configs/experiment_transfer_v1_2_3.yaml \
  --report "$OUTPUT_ROOT/preflight/stage07_transfer_primary"

bash tartan/research_score/scripts/run_transfer_matrix.sh \
  --config tartan/research_score/configs/experiment_transfer_v1_2_3.yaml \
  --profile transfer_primary --output "$OUTPUT_ROOT/05_transfer"
```

脚本应生成任务清单，支持逐任务 resume，不覆盖已完成 run。每个任务先注册不可变 `run_id`；成功、失败、OOM 和人工中断均写入 registry。恢复只允许复用同一 run ID 和一致的 config/data/checkpoint hash。所有正式矩阵 run 按潜在数据效率或方法消融证据完整保留。

## 5. 统计规则

- 主评价单位为非重叠8 m 闭环段，共63个；辅助宏平均单位为5条 episode；
- 不将段级数量解释为完全独立场景数；如报告不确定性，按 episode 聚类。
- `transfer_primary` 在同一 budget 下比较全量微调、冻结 adapter、无条件联合训练和具身条件扩散；
- `proxy_pair_auxiliary` 不产生主比较或性能排序；
- 禁止跨 profile 把 `ours` 与 nuPlan 主迁移方法直接解释为公平优劣；
- 在同一 8 m 测试段上对四种方法做配对汇总；
- 报告段级均值、episode 等权宏平均、有效段数和有效 episode 数；
- 预注册主指标顺序：SR → CR → SPL。

## 6. 自动质量检查

- 当前应有的 12 个主迁移训练 run 和对应 12 个非重叠闭环评价是否齐全；
- 输入/split/controller hash 是否一致；
- 每个 run 是否使用正确 budget；
- checkpoint 是否由 validation 选择；
- 是否存在 NaN、空 episode、重复 episode；
- 失败 episode 是否被保留；
- 最短路径不可计算的 episode 单独列出，不偷偷丢弃；
- 训练/测试地图交集必须为零；
- run registry 包含失败/中断任务，输出目录不存在覆盖行为。

## 7. 必须生成的结果

```text
05_transfer/
├── transfer_primary/runs/<method>/<budget>/<repeat_id>/...
│   ├── transfer_per_episode.parquet
│   ├── transfer_seed_summary.csv
│   ├── transfer_main_table.csv
│   ├── transfer_main_table.md
│   └── target_budget_curve.png
├── proxy_pair_auxiliary/runs/<method>/<budget>/<repeat_id>/...
│   ├── proxy_per_episode.parquet
│   ├── proxy_seed_summary.csv
│   ├── proxy_auxiliary_table.csv
│   └── proxy_auxiliary_table.md
├── strict_pair/BLOCKED_WAITING_DATA.md
├── run_registry.parquet
├── failure_breakdown.png
├── statistical_tests.json
├── audit.json
├── review_packet.md
└── stage_report.md
```

主迁移表明确写 `nuPlan→ANYmal` 且不包含 Proxy 完整方法；Proxy 表文件名和标题必须含 `proxy_pair_auxiliary`。Strict profile 到位后输出独立 `strict_pair` 目录。

## 8. 人工验收问题

1. 四种预训练迁移策略在 1%/10%/100% 下的相对差异是否稳定？
2. SR 提升是否以碰撞或路径低效为代价？
3. 100% 下是否仍有收益，或只是追平目标域充分训练？
4. 结果是否由单一 episode 或少数8 m 段驱动？
5. Proxy 完整方法是否被严格限制在辅助结论，且失败/中断 run 均保留？

人工确认前不进入解耦实验。
