---
document: agent-handoff-template
plan_version: "1.2.3"
status: TEMPLATE
updated: 2026-09-09
---

# 单阶段 Agent 交接模板

将下列内容复制给实施 agent，并只替换尖括号字段。不要一次委托多个阶段。

```markdown
你负责实现实验计划 Stage <NN>，不得进入下一阶段。

开始前必须完整阅读：
1. experiment_execution_plan/01_shared_execution_contract.md
2. experiment_execution_plan/01_preflight_framework.md
3. experiment_execution_plan/decisions.md
4. experiment_execution_plan/risk_register.md
5. experiment_execution_plan/<STAGE_FILE>.md

当前 profile：<transfer_primary | proxy_pair_auxiliary | strict_pair | common_stage>。
`transfer_primary` 是 nuPlan→ANYmal 主迁移；`proxy_pair_auxiliary` 仅保留 synthetic/interface fixture 和非科学配对拒绝诊断，不得训练正式模型或进入主表。Strict 数据未到位时禁止 Car–Dog 正式结论。

执行顺序必须是：读取并核对冻结方案 → Preflight → 实现 → 单元/接口/科学不变量/回归/负对照测试 → tiny real-data smoke → 保存证据 → 生成 stage_report.md 与 review_packet.md → 停止。

限制：
- 配置中仍有 null/TBD，或 `length_m` 没有 Stage 03 训练审计 hash 时禁止启动训练；
- 禁止读取 future GT 构造非 oracle 输入；
- 禁止静默修改超参数、缩小协议或覆盖历史输出；
- 每次运行必须生成唯一 run_id；失败、中断和潜在消融同样留证；
- 禁止更改已经冻结的源 normalizer；
- 发现需要负责人决策的问题时，写入 review_packet 并停止，不得自行决定；
- 不得将状态写成 APPROVED。

完成后回复：修改文件、实际命令、测试结果、证据绝对路径、剩余 P0/P1/P2、是否建议进入独立审查。
```

## 独立审查 Agent 模板

```markdown
你只审查 Stage <NN>，不替实施 agent 修复代码，也不启动下一阶段。

阅读冻结计划、resolved config、git diff、Preflight、全部测试、smoke 产物、stage_report 和 review_packet。重点检查：future leakage、pair_level/claim 越界、目标/Swap 方向、mask 与 shape、源模型回归、公平性 hash、失败样本保留、不可复现设置。

输出固定为：
- 结论：通过建议 / 有条件通过 / 不通过
- P0 阻塞问题
- P1 必须修复问题
- P2 建议改进
- 已核验证据绝对路径
- 未能核验的项目
- 是否建议负责人放行 formal run

不得把阶段状态改成 APPROVED。
```

实施报告和审查报告登记到 [reports/README.md](./reports/README.md)，最终由负责人决定是否更新 [00_dashboard.md](./00_dashboard.md) 为 `APPROVED`。
