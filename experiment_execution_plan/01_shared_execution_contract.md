---
document: execution-contract
stage: 01
plan_version: "1.2.3"
status: APPROVED
requires_human_approval: true
---

# 阶段 01：所有执行 Agent 共用的工作契约

> 与任一阶段计划同时交给执行 agent。Preflight 细则见 [01_preflight_framework.md](./01_preflight_framework.md)。

## 1. 权限与停止规则

执行 agent 只完成当前阶段，提交 `stage_report.md` 和 `review_packet.md` 后停止。只有负责人可以将阶段标记为 `APPROVED` 并允许下一阶段。

禁止：

- 删除或覆盖原数据、checkpoint 和历史输出；
- 在 route 协议未冻结时自行决定长度、点数或条件形式；
- 用测试集选超参数、阈值或 checkpoint；
- 用 future GT 构造非 oracle 主输入；
- 把 Proxy 结果表述为 Strict Car–Dog 结论；
- 把 `diff↔anymal` Proxy 辅助结果混入 `nuPlan→anymal` 主迁移表并作同一科学解释；
- 失败后静默缩数据、减方法、改 batch/采样数或删失败样本；
- 未经负责人批准启动下一阶段或正式大矩阵。

## 2. 路径与配置

数据视为已拥有，但路径不得写死在 Python 或计划中。执行环境建立不进 Git 的 `local.env`：

```bash
PROJECT_ROOT=...
PYTHON_BIN=...
TARTAN_ROOT=...
NUPLAN_ROOT=...
SOURCE_ARGS=...
SOURCE_CKPT=...
OUTPUT_ROOT=...
CF_PAIR_ROOT=...       # Strict 数据到位后才要求
```

所有超参数只来自 YAML。shell 只传 `--config` 和 `--profile transfer_primary|proxy_pair_auxiliary|strict_pair`，不重复学习率、route 长度等值。每次运行输出不可变的 `config.resolved.yaml` 与 `config.sha256`；不允许出现未解析的 `null/TBD`。

三个 profile 的用途固定：

- `transfer_primary`：nuPlan Car checkpoint 向 Tartan ANYmal 的 1%/10%/100% 迁移；不使用不严格配对的 `L_inv/L_swap` 形成正式完整方法结论；
- `proxy_pair_auxiliary`：仅运行 synthetic/interface fixture 与非科学配对拒绝诊断，不进入正式训练、完整 loss 或结果主表；
- `strict_pair`：未来严格 Car–Dog 数据到位后的正式完整方法实验。

CLI 可接受编排选择器 `all_available`，它只能依次展开为当前可用的正式 profile；每个实际 run 的 resolved config、manifest 和目录仍必须写三个真实 profile 之一，禁止把 `all_available` 当成结果 profile。

## 3. 八步闭环

1. 冻结本阶段方案和验收标准；
2. 单独执行 Preflight，非零退出即停止；
3. 实现代码；
4. 运行环境、契约、科学不变量、回归、负对照五类测试；
5. 最小真实数据 smoke：forward、backward、保存、恢复、评测；
6. 独立审查 agent 只读检查代码和产物；
7. 负责人查看一页 `review_packet.md`；
8. 获得明确批准后才执行正式运行。

## 4. 代码规则

- 新代码放 `tartan/research_score/`，原 `tartan/` baseline 保持可运行；
- 对 `diffusion_planner/` 只增加默认关闭的兼容接口；
- 源 normalizer 冻结，只为新增 ability vector 计算训练统计；
- 数据、模型、loss、训练、评测、Preflight 分模块；
- 所有随机源固定；配置解析后记录实际生效值；
- 大规模运行前先估算总 run、单 run 时长/显存、总 GPU-hours、磁盘增长、checkpoint 间隔和恢复策略；
- OOM 或资源不足只能停止并提交偏差申请，不能自行改变协议。

## 5. 测试规则

- Environment：路径、权限、样本、CUDA、显存、磁盘、依赖；
- Contract：键、shape、dtype、device、mask、配置、上下游接口；
- Scientific invariant：无 future leakage、同噪声 pair、Swap 方向/目标平台正确；
- Regression：新模块关闭时与源 checkpoint 一致；
- Negative control：打乱 pair、换错具身、改变 future GT 等必须产生预期结果。

“写了测试”不等于通过；报告必须包含真实命令、退出码和摘要。

## 6. 每个运行必须留证

每次启动前生成全局唯一且不可变的 `run_id`，建议为 `<profile>_<method>_<budget>_<repeat>_<UTC时间>_<短commit>`。输出目录存在时默认失败；禁止 `--overwrite`。恢复只能在同一 `run_id` 内使用 `--resume`，并在 manifest 记录恢复点。

```text
config.resolved.yaml
config.sha256
command.txt
run_manifest.json
preflight/report.json
preflight/report.md
stdout.log
checkpoints/
metrics/per_sample.*
metrics/summary.*
artifacts/
```

manifest 记录 git commit/dirty、Python/Torch/CUDA/GPU、seed、profile、pair_level、数据/checkpoint/config SHA256 和上游产物 hash。

- 正式实验和可能成为消融的独立小实验：永久保留最佳/最终 checkpoint、逐样本指标、汇总、配置、命令、日志和可视化；
- 纯代码调试：可不保留大 checkpoint，但必须保留配置、命令、测试结果、错误摘要和 run manifest；
- 成功、失败、OOM、人工中止均保留状态与原因；
- 任何批量清理先生成 `cleanup_manifest.md`，列出路径、大小、保留证据和可恢复性，只有负责人批准后执行；
- 是否进入最终消融表由协议一致性决定，不能因为指标较差而删除运行。

## 7. 实施与审查分离

- 实施 agent：实现、测试、smoke、修复并提交阶段报告；
- 审查 agent：不改代码，只找 P0/P1 问题，输出 `PASS/CONDITIONAL/FAIL`、证据路径和是否建议正式运行；
- 负责人：只回答 1–3 个决策问题并决定是否 `APPROVED`。

## 8. 报告要求

`stage_report.md` 写完整技术记录；`review_packet.md` 使用 [模板](./review_packet_template.md) 压缩负责人所需信息。两份报告都必须列出未运行测试、与冻结协议的偏差和下一阶段前置条件，末尾写：**本 agent 已停止，未执行下一阶段。**

本契约经负责人确认后，进入 [阶段 02](./02_protocol_and_baseline_freeze.md)。
