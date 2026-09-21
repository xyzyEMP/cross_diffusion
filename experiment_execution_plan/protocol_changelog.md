---
document: protocol-changelog
current_version: score-decomp-transfer-v1.2.4
status: ACTIVE
---

# 协议版本记录

## v1.2.4 / score-decomp-transfer-v1.2.4 / 2026-09-18

- 正式方法矩阵固定为 `pretrain_finetune/pretrain_adapter/joint_train/emb_cond_diffusion` 四种预训练迁移策略×1%/10%/100% ANYmal 训练预算×seed 11，共12项；不再扩展随机初始化基线。
- 1%/10%/100% 只表示 ANYmal 训练集名义预算；实际为1/2/16条完整训练 episode、128/256/2048个密集滑窗。validation 和 test 对所有预算固定。
- 保留现有训练数据：按完整 episode 隔离后，每1 s设置锚点，将未来轨迹的前8 m重采样为80点。这些密集滑窗只用于训练和 validation loss，不声称为独立闭环试验。
- 正式测试在5条 held-out episode内按累计弧长连续切分非重叠8 m段，下一段从前一段终点之后的首个10-frame地图锚点开始，尾部不足8 m的段不进入主评价；实际共63个段级闭环任务。主表报告段级 SR/CR/SPL，同时报告5条 episode等权宏平均。
- seed 11 v3 的12项保留方法训练使用相同密集训练窗口，与本修订兼容；收敛审计通过，无需重训，只重建测试任务并闭环评价。旧的5-episode整段评价保留为 exploratory，不并入 v1.2.4 主表。

## v1.2.3 / score-decomp-transfer-v1.2.3 / 2026-09-12

- 原固定 seed 的 episode 随机切分在诊断中将 8 个 moving-capable episode 分成 train=6、val=2、test=0，使 test 仅有1个偶发 moving window，无法支撑规划评价。
- 改为 `branch-stratified-complete-episode-sha256-v1`：按“是否含至少 10 个 moving window”对完整 episode 分层，各层内按固定 seed 与 SHA256 排序分配。
- 不拆分 episode，不使用模型结果，不调整 8米弧长、数据预算、评价指标或实验一目标。
- Stage 03 经第二轮独立审查 PASS 后由负责人批准；Stage 03B 将 `astar_disconnected` 统一记为安全停止 + `route_failure`，保留在分母中且所有方法一致。

## v1.2.2 / score-decomp-transfer-v1.2.2 / 2026-09-11

- 正式 nuPlan Car 源缓存切换为 Diffusion-Planner 原生 NPZ：100 万条训练索引用于 index provenance，`ego_agent_future(80,3)` 用于源时域轨迹和表示桥接。Pluto `(8,3)` trajectory cache 标记为 `INVALID_SOURCE_DIAGNOSTIC`，不得进入正式数据流。
- 在完整训练索引上以路径规范化字符串的 SHA256 排序，预先冻结前 50,000 条为源 Car 弧长审计样本；保存算法版本、样本 ID 和样本集合 hash，查看覆盖率后禁止更换。
- 对 50,000 条源样本计算 moving/stop 分流及各候选覆盖率；候选只有在源 Car 覆盖率的单侧 95% Wilson 下界和 Tartan ANYmal 全量 train moving 精确覆盖率均不低于 90% 时才合格，再选择最大候选。
- 统计审计不代替数据语义校验：另以固定分层样本验证 NPZ schema、数值、8 s/80 点语义及原 loader/adapter 一致性。报告分别标注百万索引覆盖、5 万弧长统计审计和内容验证规模。
- 本修订不改变实验一的 Car→ANYmal 迁移目标、模型分解、损失、基线、split、预算或 Stage 03B 闸门；由负责人于 2026-09-11 明确批准。

## v1.2.1 / score-decomp-transfer-v1.2.1 / 2026-09-11

- 经负责人明确批准，将真实 anchor/window 按累计未来弧长分成 `moving_planning` 与 `stop_or_short`；全部样本保留，只有 moving 窗口参与 `{8,10,12,15,20}m` 的 90% 覆盖选择。
- 禁止用 nuPlan `scenario_type` 推断弧长，禁止用完整 Tartan episode 总长替代窗口长度；选择只读取 train moving manifest。
- 实验一主张限定为同一 ModularNeighborhood 地图上的未见 episode 迁移，不支持 unseen-map claim。
- 现有 Diff–ANYmal 数据从正式实验一、弧长选择和配对 loss 中移除，只保留 `not_scientifically_paired` 诊断及 synthetic/interface fixture；Strict 继续 `BLOCKED_WAITING_DATA`。
- 固定目标由预先冻结的路径距离规则离线生成并写入 manifest；route builder 运行时只能读取 current map/current state/fixed goal，future-GT mutation 负对照保持 goal 不变。
- 触发旧证据：`/tj-share/cross_diffusion_workdir/research_score_v1_2/01_data_audit/stage03_index_20260911T100714Z_nogit/`。旧 run 仍属于 v1.2 阻塞证据，不重写；所有 remediation 配置和 manifest 必须记录 v1.2.1。
- 该修订由负责人于 2026-09-11 明确批准；Stage 03 remediation 完成后仍须独立审查和人工闸门。

## v1.2 / score-decomp-transfer-v1.2 / 2026-09-09

- 将实验一拆为 `transfer_primary`（nuPlan Car→ANYmal）与 `proxy_pair_auxiliary`（Tartan Diff↔ANYmal），未来 `strict_pair` 单独接入。
- 固定 route 条件为目标 + 当前地图生成的最多 6 条 route-set，禁止 future GT。
- 固定轨迹为物理弧长 + 80 点；弧长从 `{8,10,12,15,20} m` 中按训练集至少 90% 覆盖选择最大值。
- 明确源 checkpoint 的 8 s/80 帧回归空间与固定弧长研究空间不等价，新增表示桥接与双空间测试。
- Stage 03 后新增 Stage 03B，在研究训练前解析验证固定目标闭环、SR、CR、SPL、最短路和终止条件。
- 冻结七种方法、1%/10%/100%预算和 5 个联合 repeats；主迁移与 Proxy 辅助不跨 profile 作公平优劣比较。
- 所有 run 使用不可变 ID；成功、失败、中断和潜在消融均留证，清理必须由负责人批准。
- v1.1 产物若尚未正式运行则直接由 v1.2 取代；若已有运行，只能标记 `exploratory_v1.1`，不得并入 v1.2 表格。

## v1.1 / score-decomp-proxy-v1 / 2026-09-09

- 数据路径由写死改为 `local.env + resolved config + Preflight`；数据本身默认用户拥有。
- route 从“8 秒/80 帧”改成配置化固定长度；具体物理长度、点数和条件形式待负责人冻结。
- 当前配对从“假定已有 Strict Car–Dog”改为“Tartan 两智能体 Proxy”；未来 Strict 走同一接口。
- Proxy 只支持管线验证，不进入正式 Car–Dog 主结论。
- 辅助一致性/Swap loss 第一版在 x0 空间计算；epsilon/score 仅派生诊断。
- 增加 domain_id、domain probe 和匹配 pair 限制，处理数据域/具身混杂。
- 加性双分支仅 `L_diff` 提升为建议正式对照；普通微调与 adapter 拆分。
- 5 个 data/train seeds 改为一一绑定的 5 次联合重复。
- 增加统一 Preflight、独立审查、review packet、资源估算、风险与决策日志。

## v1.0 / score-decomp-draft / 2026-09-09

- 初版 10 阶段实验计划。
- 已识别 `x_start` checkpoint、oracle-route 泄漏、双向 Swap 和分支坍缩问题。
- 状态：`SUPERSEDED`；保留在 Git 历史中，不再作为执行合同。

## 变更规则

影响数据 split、route、指标、基线、loss、seed、控制器或测试集的修改必须：

1. 新增版本号；
2. 写明原因和受影响的旧 run；
3. 决定旧结果是 `compatible`、`exploratory` 还是 `invalidated`；
4. 负责人批准后才运行。
