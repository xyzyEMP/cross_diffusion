---
document: server-handoff
plan_version: "1.2.3"
protocol_version: score-decomp-transfer-v1.2.3
status: HISTORICAL_STAGE_02_HANDOFF
updated: 2026-09-09
---

# 服务器交接与第一条 Agent 指令

## 1. 上传后的目录

将整个 `experiment_execution_plan/` 放到服务器 Diffusion-Planner 项目根目录。将 [project_AGENTS_template.md](./project_AGENTS_template.md) 复制为项目根目录 `AGENTS.md`，但若服务器已有 `AGENTS.md`，先合并并保留已有规则，不得直接覆盖。

在项目根目录创建不进入 Git 的 `local.env`：

```bash
PROJECT_ROOT=/server/path/Diffusion-Planner
PYTHON_BIN=/server/path/conda/envs/diffusion_planner/bin/python
TARTAN_ROOT=/server/path/TartanGround
NUPLAN_ROOT=/server/path/nuplan
SOURCE_ARGS=/server/path/Diffusion-Planner/checkpoints/args.json
SOURCE_CKPT=/server/path/Diffusion-Planner/checkpoints/model.pth
OUTPUT_ROOT=/server/path/Diffusion-Planner/tartan/outputs/research_score_v1_2
CF_PAIR_ROOT=
```

`CF_PAIR_ROOT` 当前允许为空，但 `strict_pair` 必须保持 `BLOCKED_WAITING_DATA`。其余路径必须为绝对路径。

## 2. 第一条实施 Agent 指令

```markdown
你是 Stage 02 实施 agent。项目根目录由 ./local.env 的 PROJECT_ROOT 指定。你只能完成 Stage 02，完成报告后必须停止。

开始前完整阅读：
- AGENTS.md
- experiment_execution_plan/00_dashboard.md
- experiment_execution_plan/00_overall_progress.md
- experiment_execution_plan/01_shared_execution_contract.md
- experiment_execution_plan/01_preflight_framework.md
- experiment_execution_plan/02_protocol_and_baseline_freeze.md
- experiment_execution_plan/decisions.md
- experiment_execution_plan/risk_register.md

先检查 Git 工作树并保留用户已有修改。按照 Stage 02 实现统一 Preflight 的最小框架、源模型审计和回归测试；执行真实 CUDA checkpoint load、固定 seed 输出签名、现有 tartan 测试和不覆盖历史结果的 oracle-route 小样本 smoke。

必须遵守：
1. 源是 nuPlan Car checkpoint；Diff↔ANYmal 只属于后续 proxy_pair_auxiliary。
2. 不实现 Stage 03，不训练研究模型，不自行改变 route/轨迹协议。
3. 当前 oracle future route 必须如实标注，不得写成自主导航。
4. 所有路径来自 local.env，禁止写死开发机路径。
5. 不允许 CPU 静默替代 CUDA，不允许覆盖已有输出。
6. 新 run 使用唯一 run_id；失败和中断也保留 manifest。
7. 输出 config.resolved.yaml、config.sha256、command.txt、source_model_audit.json、source_input_schema.json、source_output_signature.npz、完整测试日志、stage_report.md 和 review_packet.md。
8. 不得将阶段状态写成 APPROVED。

若遇到错误，先定位并修复 Stage 02 范围内的问题；若需要改变冻结协议或缺少外部数据/权限，停止并记录 P0/P1/P2，不得自行扩展范围。

最终回复只列：修改文件、实际命令、测试通过/失败数、证据绝对路径、P0/P1/P2、是否建议进入独立审查。
```

## 3. Stage 02 独立审查指令

```markdown
你是 Stage 02 独立审查 agent。只读审查，不修改代码，不执行 Stage 03。

阅读冻结计划、git diff、Preflight、resolved config、测试日志、CUDA smoke、stage_report.md 和 review_packet.md。重点核验 checkpoint strict load、固定 seed 重复性、源 normalizer、oracle-route 边界、CUDA 是否真实执行、历史输出是否未覆盖、路径是否配置化、失败证据是否保留。

输出：PASS/CONDITIONAL/FAIL、P0、P1、P2、证据绝对路径、未核验项、是否建议负责人批准。不得写 APPROVED。
```

负责人确认 Stage 02 后，才从 [agent_handoff_template.md](./agent_handoff_template.md) 生成 Stage 03 指令。

## 4. 上传前后校验

上传前在本机保存整个目录 hash；上传后在服务器重新计算并比较：

```bash
find experiment_execution_plan -type f -print0 | sort -z | xargs -0 sha256sum > experiment_execution_plan.sha256
```

由于清单文件会包含自身时产生递归变化，清单应写在计划目录外或在命令中排除该文件。计划内任何后续修改必须登记到 `protocol_changelog.md`。
