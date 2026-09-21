# Diffusion-Planner 跨具身实验长期规则（复制为项目根目录 AGENTS.md）

1. 只执行用户指定阶段；完成后生成 `stage_report.md` 和 `review_packet.md` 并停止。
2. 每次实现和正式运行前必须执行统一 Preflight；P0 失败不得继续。
3. 禁止静默降级、覆盖历史输出、删除失败样本、根据测试集调参。
4. 非 oracle 主实验禁止使用 future GT 构造输入。
5. `transfer_primary` 固定为 nuPlan→ANYmal；`proxy_pair_auxiliary` 只允许 synthetic/interface fixture 与非科学配对诊断，不得训练正式模型或进入主表。Strict 数据必须通过审计后才可运行配对实验。
6. route 固定为目标 + 地图生成的最多 6 条 route-set；轨迹固定 80 点，物理弧长必须来自 Stage 03 训练集选择产物及 hash。
7. 源 checkpoint 的 observation/state normalizer 冻结；新增 ability normalizer 只用训练集统计。
8. 所有配置只有 YAML 一个来源，并保存 `config.resolved.yaml` 和 SHA256。
9. 必须保存命令、环境、代码/数据/checkpoint hash、逐样本结果和全部失败记录。
10. 测试至少覆盖 environment、contract、scientific invariant、regression、negative control。
11. OOM、磁盘不足或路径问题只能停止并报告，不能自行改变 batch、方法、采样数或数据范围。
12. 未获负责人明确确认，不得进入下一阶段或启动正式大规模矩阵。
13. 已有工作树修改属于用户；不得回滚无关改动。
14. 实施和独立审查由不同任务完成；审查任务只读，不替实现辩护。
15. 所有 run 使用不可变唯一 ID；禁止覆盖。成功、失败和中断均留 manifest；潜在消融保留完整 checkpoint 与逐样本结果。
16. 任何清理先生成清单并取得负责人明确批准。

执行前阅读：

- `experiment_execution_plan/00_dashboard.md`
- 当前阶段计划；
- `experiment_execution_plan/01_shared_execution_contract.md`
- `experiment_execution_plan/01_preflight_framework.md`
- `experiment_execution_plan/decisions.md`
- `experiment_execution_plan/risk_register.md`
