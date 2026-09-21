---
stage: 06
plan_version: "1.2.4"
status: COMPLETE
protocol_version: score-decomp-transfer-v1.2.3
depends_on: [stage_05]
profiles: [transfer_primary, proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 06：对照方法实现与公平性 Smoke Test

> v1.2.2 边界：Proxy 只做 synthetic/interface smoke，不构成正式基线组；涉及真实配对的公平性比较等待 Strict 数据。

> 前置：[阶段 05](./05_losses_and_training.md) 已通过。  
> 本阶段只用小规模数据和单 seed，确认实验矩阵公平；人工确认后才运行完整实验一。

## 1. 必须实现的主方法

| method_id | 方法 | 初始化 | 数据 | 进入的 profile | 特殊结构/损失 |
|---|---|---|---|---|---|
| `pretrain_finetune` | nuPlan 预训练后全参数微调 | nuPlan checkpoint | ANYmal | transfer primary | 固定为全参数，不与 adapter 混写 |
| `pretrain_adapter` | nuPlan 预训练后参数高效适配 | nuPlan checkpoint | ANYmal | transfer primary | 等参数量 Adapter/LoRA |
| `joint_train` | 非配对源目标联合训练 | nuPlan checkpoint | nuPlan Car + ANYmal | transfer primary | 无 `L_inv/L_swap` |
| `emb_cond_diffusion` | 具身条件化 Diffusion | nuPlan checkpoint | nuPlan Car + ANYmal | transfer primary | platform/ability condition，无配对 loss |
| `additive_diff_only` | 加性共享—具身双分支 | nuPlan checkpoint | Diff + ANYmal | proxy auxiliary；未来 strict | 仅 `L_diff`，无 inv/swap/sep |
| `ours_score_decomp` | 本文完整方法 | nuPlan checkpoint | 未来 Strict Car–Dog pairs | strict only | 当前只做 synthetic 接口 smoke；正式训练等待 Strict 数据 |

补充但不进入主表首列的方法：`source_zero_shot`、`oracle_route_upper_bound`、`platform_specific_oracle`。

## 2. 公平性约束

同一 profile 内的比较方法必须共享：

- 同一 train/val/test split 和目标平台 budget manifest；
- 同一当前观测输入、goal-route builder、数据增强和归一化；
- 同一输出 horizon、采样次数、推理步数和候选选择规则；
- 同一下游运动学/控制器、碰撞检查器和成功阈值；
- 相同 validation 选择规则和最大训练 compute 上限；
- 相同 5 个一一绑定联合重复 ID。

允许变化的只有方法定义所必需的初始化、训练数据组合和模型结构。参数量、训练 GPU-hours 和推理延迟必须报告。

`emb_cond_diffusion/additive_diff_only/ours` 报告参数量，并尽量控制新增参数量相当。不同 profile 不因共享 checkpoint 就直接横向作科学比较。

所有使用 nuPlan checkpoint 的方法共享同一个表示桥接定义、初始化和训练规则。任何可学习 bridge 的参数量、预训练数据和更新步数计入方法 compute。`source_zero_shot` 若通过确定性曲线重采样进入固定弧长评测，必须报告不足弧长的 coverage/mask，且只作 0% 诊断。

## 3. 代码落点

```text
tartan/research_score/configs/methods/
├── pretrain_finetune.yaml
├── pretrain_adapter.yaml
├── joint_train.yaml
├── emb_cond_diffusion.yaml
├── additive_diff_only.yaml
└── ours_score_decomp.yaml
tartan/research_score/training/method_factory.py
tartan/research_score/evaluation/fairness_audit.py
tartan/research_score/scripts/run_method_smoke.sh
tartan/research_score/scripts/run_stage06.sh
tartan/research_score/tests/test_method_factory.py
tartan/research_score/tests/test_fair_budget.py
tartan/research_score/tests/test_eval_parity.py
```

## 4. Smoke 矩阵

- 目标平台 budget：先使用固定小子集或 1%；
- seed：只用 `0`；
- 每种方法：短训练 200–1000 update，以能跑通为准；
- validation：每种方法使用完全相同的 10–30 episodes；
- inference：固定采样数和相同 controller；
- 目的：检查代码和协议，不比较论文优劣。

## 5. 必须通过的测试与审计

- 六种 method config 均可构建、forward、backward、save、resume、evaluate；
- `transfer_primary` 方法不得加载 Proxy fixture；`proxy_pair_auxiliary` 不创建正式训练 run，任何接口诊断均不得写入主迁移结果目录；
- fairness audit 对每种方法输出 input hash、split hash、budget hash、controller hash；
- 除允许项外，hash 必须一致；
- 各方法看到的目标平台 trajectory 数完全一致；
- `ours` 只有在有效 pair batch 中计算 `L_inv/L_swap`，报告 pair_level；
- `emb_cond_diffusion` 的参数量和输入条件有明确记录；
- 评测不读取 method-specific GT 信息；
- 每种方法至少产生一个成功和失败可视化，若数据中确实存在。

## 6. 执行命令

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

$PYTHON_BIN -m tartan.research_score.preflight \
  --stage 06 --profile transfer_primary \
  --config tartan/research_score/configs/stage06_transfer.yaml \
  --report "$OUTPUT_ROOT/preflight/stage06_transfer_primary"

bash tartan/research_score/scripts/run_stage06.sh \
  --config tartan/research_score/configs/stage06_transfer.yaml \
  --profile transfer_primary --repeat-id 0 \
  --output "$OUTPUT_ROOT/04_baseline_smoke"
```

脚本必须支持重入：已成功的 run 不被覆盖，失败 run 写明原因；可用 `--resume` 恢复。

## 7. 阶段产物

- 六种方法的 resolved config；
- 公平性 hash 对照表；
- 参数量、训练步数、GPU 时间、推理时延；
- smoke validation 结果；
- 至少一组统一输入上的预测叠加图；
- 每个 smoke run 的不可变 `run_id`、状态和保留级别；
- `review_packet.md`；
- `stage_report.md`。

## 8. 人工验收

本阶段不以“ours 是否最好”为通过条件，只检查六种方法是否真实可运行、profile 隔离且各自比较公平。任何方法失败都不得直接删除；应保留 run manifest，修复或由负责人登记协议变更。

人工确认 smoke 后，才进入 [实验一](./07_experiment_transfer.md)。

本阶段 agent 提交报告后停止，不启动正式矩阵。
