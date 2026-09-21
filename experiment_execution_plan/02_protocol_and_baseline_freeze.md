---
stage: 02
plan_version: "1.2.3"
status: NOT_STARTED
protocol_version: score-decomp-transfer-v1.2.3
depends_on: [stage_01]
profiles: [transfer_primary, proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 02：Preflight 引导、协议确认与原始基线冻结

> 前置：[执行契约](./01_shared_execution_contract.md) 已通过人工确认。  
> 下一阶段：[数据与严格配对](./03_data_and_pairing.md)。

## 1. 阶段目标

先实现可供后续阶段复用的最小 Preflight 引导入口，再形成不可含糊的源模型协议，并证明原 Diffusion-Planner checkpoint、现有 TartanGround 适配代码和 GPU 环境都可复现。此阶段不训练新方法。

## 2. 必须确认的事实

1. `checkpoints/args.json` 的源模型为 `x_start`、80 帧、10 Hz、8 s；这是源 checkpoint 事实，不是新研究 route 的最终定义。
2. 原模型输入键、形状、dtype 和归一化规则全部形成机器可读 schema。
3. 原 Tartan `build_model_features()` 使用未来 GT 构造 `route_lanes`，将其标为 `oracle_route=true`。
4. 新研究轨迹使用训练集选择的固定物理弧长和 80 个重采样点，输入使用固定目标与地图生成的最多 6 条 route-set；Stage 02 只登记冻结规则，不预先伪造 Stage 03 的数据统计结果。
5. 规划模型输出保持机体级 SE(2) 轨迹；四足足端和关节命令不在本论文输出范围。

## 3. 需要新增的代码和文件

```text
tartan/research_score/
├── __init__.py
├── configs/base.yaml
├── preflight/{cli,registry,result}.py
├── data/schema.py
├── scripts/audit_source_model.py
├── scripts/run_stage02.sh
└── tests/test_source_regression.py
```

### `data/schema.py`

定义 `TypedDict` 或 dataclass：

- `CanonicalObservation`：场景、目标、历史、route、platform；
- `CanonicalTrajectory`：`xy`、`cos_yaw`、`sin_yaw`、有效掩码、原始时间戳；
- `ModelBatch`：与原模型键兼容的张量；
- `RunIdentity`：数据版本、split、method、budget、seed。

### `audit_source_model.py`

输出：

- checkpoint SHA256 与参数量；
- 缺失/多余 state dict key；
- 模型类型、SDE、future length；
- 固定输入与固定随机种子下的输出 hash、均值、标准差；
- adapter 尚未启用时的原模型性能摘要；
- 输入中哪些来自观测，哪些来自未来真值。

### Preflight 引导入口

先实现 [统一 Preflight 规范](./01_preflight_framework.md) 中的最小 registry、配置/path/CUDA/checkpoint/output-overwrite 检查，并为失败情况写测试。其余 stage-specific checks 在对应阶段扩展。

## 4. 要执行的检查

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

$PYTHON_BIN -m pytest tartan/tests -q
$PYTHON_BIN -m pytest tartan/research_score/preflight/tests -q
$PYTHON_BIN -m tartan.research_score.preflight \
  --stage 02 --profile transfer_primary \
  --config tartan/research_score/configs/stage02.yaml \
  --report "$OUTPUT_ROOT/preflight/stage02"
$PYTHON_BIN -m pytest tartan/research_score/tests/test_source_regression.py -q
$PYTHON_BIN -m tartan.research_score.scripts.audit_source_model \
  --args "$SOURCE_ARGS" \
  --checkpoint "$SOURCE_CKPT" \
  --device cuda \
  --output "$OUTPUT_ROOT/00_baseline_freeze"
```

另外运行现有 Tartan open-loop 的小样本 smoke，不覆盖历史 `final` 和 `town_focused_v2`：

```bash
$PYTHON_BIN -m tartan.evaluation.evaluate \
  --data-root "$TARTAN_ROOT" \
  --args-file "$SOURCE_ARGS" \
  --checkpoint "$SOURCE_CKPT" \
  --output-dir "$OUTPUT_ROOT/00_baseline_freeze/tartan_oracle_smoke" \
  --samples-per-trajectory 1 --batch-size 2 --diffusion-repeats 1 --device cuda \
  --no-visualizations
```

## 5. 必须通过的测试

- checkpoint `strict=True` 加载成功；
- 固定 seed 连续两次输出相同；
- CUDA forward 成功，无 NaN/Inf；
- 原有 `tartan/tests` 全部通过；
- schema 能表达原 nuPlan batch 和 Tartan batch；
- 审计报告明确指出 oracle future leakage，不能写成自主目标导航结果。
- `base.yaml` 固定 `condition_mode=route_set`、`max_candidates=6`、`num_points=80` 和弧长选择规则；`length_m` 只能由 Stage 03 审计产物回填并携带其 hash，不能由 agent 手填。

## 6. 阶段产物

```text
tartan/outputs/research_score_v1_2/00_baseline_freeze/
├── source_model_audit.json
├── source_input_schema.json
├── source_output_signature.npz
├── tartan_oracle_smoke/
├── review_packet.md
└── stage_report.md
```

## 7. 人工验收标准

- 原模型和当前环境真实跑通；
- checkpoint 参数化被确认是 `x_start`；
- 主实验、oracle 上界和 smoke test 的边界写清楚；
- 没有修改历史输出；
- 输出签名足以供阶段 04 做数值回归。

验收前停止，不进入数据构造。
