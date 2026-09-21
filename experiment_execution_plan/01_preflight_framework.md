---
document: preflight-specification
plan_version: "1.2.3"
status: APPROVED
owner: implementation_agent
requires_human_approval: true
---

# 统一 Preflight 安全检查框架

## 1. 目标

每次写阶段代码前和每次正式运行前，先用独立入口检查环境、数据、接口、配置和科学协议。Preflight 只读数据，不训练、不下载、不修改原始数据；任何 P0 检查失败必须非零退出。

统一命令：

```bash
python -m tartan.research_score.preflight \
  --stage 04 \
  --profile proxy_pair_auxiliary \
  --config tartan/research_score/configs/stage04.yaml \
  --report tartan/outputs/research_score_v1_2/preflight/stage04
```

## 2. 代码结构

```text
tartan/research_score/preflight/
├── __init__.py
├── cli.py
├── registry.py
├── result.py
├── checks/
│   ├── environment.py
│   ├── filesystem.py
│   ├── config.py
│   ├── checkpoint.py
│   ├── data.py
│   ├── interfaces.py
│   ├── splits.py
│   ├── pairing.py
│   ├── resources.py
│   └── upstream.py
└── tests/
```

每个 check 返回 `id/severity/status/evidence/remediation`。状态：`PASS/WARN/FAIL/SKIP`；severity：`P0/P1/P2`。任一 P0 FAIL 时 CLI 退出码非零。

## 3. 通用检查

- `local.env` 所需路径是否解析、可读/可写；
- TartanGround、nuPlan 样本是否存在且非空；
- Python、包版本、PyTorch、CUDA、GPU 型号/显存；
- 当前磁盘容量和预计输出增长；
- checkpoint、args、数据 manifest 与上游结果的 SHA256；
- YAML 必需字段、类型、范围、未知键、未解析 `null/TBD`；
- 输出目录是否已存在，是否会覆盖历史 run；
- git commit、dirty 文件和与本阶段冲突的用户修改；
- 上游阶段是否为 `APPROVED`，证据 hash 是否一致。

## 4. 科学协议检查

- route 表示模式、物理长度/点数/采样规则是否已冻结；
- trajectory 长度和 route 长度是否一致；
- 输入依赖中是否出现 `future_gt`；
- train/val/test 的 map、trajectory、pair 是否泄漏；
- 实际 run 的 `profile` 是否为 `transfer_primary/proxy_pair_auxiliary/strict_pair`，并与数据和输出根目录一致；编排选择器 `all_available` 必须展开后逐 profile 检查；
- 配对数据的 `pair_level` 是否为 `proxy` 或 `strict`；
- Proxy 是否带 `supports_claims.car_dog_scientific_claim=false`；
- Strict profile 是否真的有 `CF_PAIR_ROOT`、审计报告和合格 pair；
- paired batch 是否同噪声、同 timestep；
- Swap 的 source shared、target embodiment、target supervision 是否一致；
- 源 normalizer 是否冻结；新增 ability normalizer 是否只用 train。

## 5. 分阶段扩展检查

| 阶段 | 额外检查 |
|---|---|
| 02 | 源 checkpoint 严格加载、固定 seed 输出签名 |
| 03 | 数据字段、三 profile manifest、split、固定弧长选择、route-set 无泄漏 |
| 03B | 固定目标、SR/CR/SPL 公式、最短路、闭环终止和手工正负轨迹 |
| 04 | 输入 shape、零残差回归、显存预算 |
| 05 | loss 权重、冻结参数、pair batch 比例、resume |
| 06 | 方法配置公平性、参数量、训练步数、控制器 hash |
| 07 | run 数、GPU-hours、磁盘、测试 episode 完整性 |
| 08 | 同 query、probe split、domain/embodiment 混杂 |
| 09 | 双向组合、目标具身约束、route-mode 协议 |
| 10 | 全部 run 齐全、原始数据可追溯、重建命令 |

## 6. 必须生成

```text
report.json
report.md
config.resolved.yaml
config.sha256
environment.json
resource_estimate.json
```

报告首页只写：总体 PASS/FAIL、P0/P1 列表、实际 profile/pair_level、实际路径和是否允许继续。

## 7. Preflight 自身测试

- 缺数据路径、错 checkpoint hash、CUDA 不可用、磁盘不足均应 FAIL；
- formal route 有 `null/TBD` 应 FAIL，prototype profile 可 WARN；
- Proxy 被配置为支持 Car–Dog claim 应 FAIL；
- `transfer_primary` 读取 Proxy pair loss 或把 Proxy 指标写入主迁移表应 FAIL；
- `proxy_pair_auxiliary` 使用 Car→Dog 科学结论或输出到主迁移目录应 FAIL；
- Strict 缺配对审计应 FAIL；
- 缺少全局唯一 `run_id` 应 FAIL；输出目录存在且不是同一 manifest 的显式 `--resume` 应 FAIL；
- 伪造 split 交集应 FAIL；
- 正常最小配置返回 0。
