---
document: project-dashboard
plan_version: "1.2.4"
status: STAGE_07_AWAITING_REVIEW
updated: 2026-09-18
owner: project_lead
---

# 小论文实验控制面板

> 本页只保留人类需要看的状态。技术路线见 [总体实施方案](./00_overall_progress.md)。

阶段完成后的实施与审查证据统一登记在 [阶段报告索引](./reports/README.md)。
服务器上传与首个任务使用 [SERVER_HANDOFF.md](./SERVER_HANDOFF.md)。

## 当前结论

- 正式实验 profile 为当前 `transfer_primary` 与未来严格配对到位后的 `strict_pair`；`proxy_pair_auxiliary` 仅保留 synthetic/interface fixture 和非科学配对诊断命名空间。
- 实验一主线：nuPlan Car checkpoint → TartanGround ANYmal。
- 当前真实 Diff↔ANYmal 不满足共同任务和坐标配对条件，退出正式实验一；只验证数据接口，不训练或评价正式配对 loss。
- route：固定目标 + 当前地图生成的最多 6 条 `route_set`；未来真值禁止进入非 oracle 输入。
- 轨迹：训练集选择固定物理弧长并统一重采样为 80 点；源 8 s 表示只作回归。
- 正式比较保留四种预训练迁移方法：全量微调、冻结 adapter、无条件联合训练和具身条件扩散；不再扩展随机初始化基线。
- 训练保留 episode 隔离后的8 m/80点密集滑窗；正式测试改为 5 条 held-out episode 中的63个非重叠8 m 闭环任务。
- 当前可支持结论：nuPlan→ANYmal 的迁移结果，以及 synthetic fixture 层面的配对接口可执行性；不能据此声称真实 Diff↔ANYmal 或 Car↔Dog 因果配对有效。
- 当前不可支持结论：严格 Car–Dog 因果一致性或反事实迁移效果。
- 当前没有方案层面的未决参数；固定弧长数值由 Stage 03 按冻结规则从训练集计算并提交审查。

## 阶段面板

| 阶段 | 状态 | 最近证据 | 阻塞 | 需负责人决定 |
|---|---|---|---|---|
| 01 协作与 Preflight | APPROVED | 用户已批准 v1.2 十二项冻结决策 | 无 | 无 |
| 02 原模型冻结 | APPROVED | `stage02_remediation_final_transfer_primary_20260911T021214Z_nogit` | 无 | 无 |
| 03 数据与配对 | APPROVED | v1.2.3 二次独立审查 PASS，8m/80点 | 无 | 无 |
| 03B 目标导航指标 | APPROVED | `stage03b_goal_benchmark_v123_final_20260913T010000Z_nogit` | 无 | 无 |
| 04 模型分解 | COMPLETE | 加性双分支、bridge、源回归测试通过 | 无 | 无 |
| 05 Loss/训练 | COMPLETE | 核心 loss、AMP、resume 与真实数据训练入口通过 | Strict pair 待提供 | 无 |
| 06 对照 Smoke | COMPLETE | 五种合法方法 GPU smoke 完成；严格方法接口测试完成 | Strict pair 待提供 | Stage 07 前确认计算预算 |
| 07 迁移实验 | AWAITING_REVIEW | seed 11 v3 的12/12项训练收敛；v1.2.4 的12/12项63段非重叠8 m闭环评价完成，已输出段级与 episode-macro SR/CR/SPL | 同一地图、5条测试 episode | 负责人复核 v1.2.4 结果 |
| 08 解耦实验 | NOT_STARTED | — | Stage 07 | 无 |
| 09 Swap 实验 | NOT_STARTED | — | Stage 08 | 无 |
| 10 论文复现包 | NOT_STARTED | — | Stage 07–09 | 无 |

## 已冻结、无需 Agent 猜测

1. 固定物理弧长与固定 80 点同时成立；弧长按训练集覆盖规则确定，不看 val/test。
2. 条件形式固定为 `route_set`，最多 6 条地图—目标候选路线。
3. Proxy 不进入正式训练、弧长选择、主表或 `L_inv/L_swap`；目标平台仍为 ANYmal，严格 Car–Dog 数据到位后单独启用 `strict_pair`。
4. 主迁移结果与 Proxy 辅助结果独立目录、独立表格、独立结论。
5. 所有独立运行不可覆盖，潜在消融完整留存。

## 状态更新规则

- 实施 agent：可写 `IN_PROGRESS/BLOCKED/AWAITING_REVIEW`；
- 审查 agent：只附审查结果，不写 `APPROVED`；
- 负责人：唯一可以写 `APPROVED/SUPERSEDED` 的角色。

每次状态变化同时更新 [协议变更](./protocol_changelog.md) 或 [风险登记](./risk_register.md)，并链接证据文件。
