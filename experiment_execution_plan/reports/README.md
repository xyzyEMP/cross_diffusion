---
document: stage-report-index
plan_version: "1.2.3"
status: ACTIVE
updated: 2026-09-12
---

# 阶段执行报告索引

本目录存放每个阶段经实际代码、测试和 smoke 运行后形成的证据摘要；它不是计划文件的副本。

## 已登记报告

- [Stage 03 实施报告](./stage_03_report.md)
- [Stage 03 独立审查](./stage_03_review.md)

## 命名与登记规则

每个阶段通过实施和独立审查后登记两份文件：

```text
stage_02_report.md
stage_02_review.md
stage_03b_report.md
stage_03b_review.md
...
stage_10_report.md
stage_10_review.md
```

- `stage_<NN>_report.md`：实施 agent 的完整技术记录，链接真实输出目录中的日志、测试、resolved config、hash 和产物。
- `stage_<NN>_review.md`：独立审查结论，只允许“通过建议 / 有条件通过 / 不通过”，不得代替负责人写 `APPROVED`。
- 一页式负责人材料使用 [review_packet 模板](../review_packet_template.md)，可保存在真实 stage 输出目录，并从对应 report 链接。
- 负责人确认后，在 [控制面板](../00_dashboard.md) 更新状态和 approved commit；协议变化同时登记到 [protocol_changelog.md](../protocol_changelog.md)。

## 报告最低字段

1. 计划版本、run ID、profile、pair level、代码 commit；
2. 实际命令与 `config.resolved.yaml` hash；
3. 已运行、失败和未运行测试；
4. 与冻结计划的全部偏差；
5. 关键指标与逐样本证据路径；
6. P0/P1/P2 问题及是否建议放行；
7. 下一阶段的前置条件。
8. `debug_minimal/ablation_candidate/formal_result` 保留级别，以及失败/中断原因。

禁止把大日志或模型文件复制进本目录；只保存可读摘要和稳定链接。
