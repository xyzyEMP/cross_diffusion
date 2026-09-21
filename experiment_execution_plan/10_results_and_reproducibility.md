---
stage: 10
plan_version: "1.2.3"
status: NOT_STARTED
protocol_version: score-decomp-transfer-v1.2.3
depends_on: [stage_07, stage_08, stage_09]
profiles: [transfer_primary, proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 10：统计汇总、论文图表与复现包

> v1.2.2 边界：Proxy 只进入附录的接口/拒绝审计，不生成性能表、消融表或科学结论。

> 前置：实验一、二、三均经人工确认。  
> 本阶段不再改变训练方法和测试阈值，只做冻结结果上的聚合、核验和包装。

## 1. 阶段目标

当前生成 `transfer_primary` 的 nuPlan→ANYmal 迁移报告，并单独保存 `proxy_pair_auxiliary` 的 synthetic/interface 测试与真实配对拒绝诊断；后者不形成方法消融或性能结论。未来 Strict 数据到位后再生成正式完整方法论文结果。所有正式与诊断目录、图题和结论完全隔离。

## 2. 论文核心产物

### 主文

1. **表 1：nuPlan→ANYmal 主迁移结果**——五种非严格配对方法 × 1%/10%/100%，SR、CR、SPL；
2. **图 1：ANYmal 数据量曲线**——五种主迁移方法的 SR/SPL，明确不含 Proxy 完整方法；
3. **附录接口审计**——synthetic Proxy 的张量、梯度、Swap 与保存恢复测试，以及真实 Diff↔ANYmal 被拒绝的原因；不作为性能表；
4. **图 2：共享分支响应图**——换具身与换目标；
5. **表 3：双向 Swap 结果**——SR、可行率、mode coverage；
6. **图 3：受控场景轨迹分布**——窄通道、急转弯、多路径。

### 附录

- 每地图、每联合 repeat 结果；
- 参数量、训练 GPU-hours、推理时延；
- success/collision/stuck/terrain failure 分解；
- probe、residual norm 与分支 intervention；
- Proxy/Strict 配对审计摘要和 claim boundary；
- 主迁移、Proxy 辅助和 Strict 三 profile 的隔离审计；
- oracle-route 与无泄漏 goal-route 的边界对照；
- 可选 capability sweep（若执行）。

## 3. 聚合代码

```text
tartan/research_score/evaluation/
├── aggregate.py
├── statistics.py
├── tables.py
├── figures.py
└── verify_artifacts.py
tartan/research_score/scripts/
├── build_paper_package.sh
└── reproduce_all_tables.sh
tartan/research_score/tests/
├── test_metric_formulas.py
├── test_cluster_bootstrap.py
└── test_artifact_completeness.py
```

## 4. 统计和报告规则

- 所有比例指标同时给分子/分母；
- 95% CI 使用 map/trajectory cluster bootstrap；
- 方法比较尽量对同一 episode 配对；
- 5 个联合 repeats 不视为无限独立 episode，保留 repeat 层级；
- 主表不使用 test 调参后重新选择的 checkpoint；
- 缺失 run、失败 run 和 excluded episode 单独列出；
- 小数位统一，但计算使用原始精度；
- 图表由原始 parquet/csv 脚本生成，不手工改数。
- 失败、中断和被排除 run 均出现在 run registry，不能因不利结果而移除；
- 只有除被消融因素外协议 hash 一致的 run 才能进入同一消融表。

## 5. 一键重建

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

$PYTHON_BIN -m tartan.research_score.preflight \
  --stage 10 --profile all_available \
  --config tartan/research_score/configs/stage10_v1_2.yaml \
  --report "$OUTPUT_ROOT/preflight/stage10_all_available"

bash tartan/research_score/scripts/build_paper_package.sh \
  --config tartan/research_score/configs/stage10_v1_2.yaml \
  --profile all_available --input-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/08_paper_package"

bash tartan/research_score/scripts/reproduce_all_tables.sh \
  "$OUTPUT_ROOT/08_paper_package"
```

第一条命令聚合冻结结果；第二条命令在干净临时目录重建所有表和图并校验 hash。

## 6. 最终目录

```text
08_paper_package/
├── transfer_primary/{README.md,result_index.md,tables/,figures/,statistics/}
├── proxy_pair_auxiliary/{README.md,result_index.md,tables/,figures/,statistics/}
├── strict_pair/BLOCKED_WAITING_DATA.md
├── run_registry.parquet
├── protocol_frozen.yaml
├── manifests/
├── raw_metric_links/
├── environment.lock
├── reproduce.sh
├── review_packet.md
└── final_audit.json
```

`result_index.md` 对每个论文结论列出：对应表/图、生成脚本、输入数据、运行 hash 和解释边界。

## 7. 最终自动核验

- 三组实验要求的所有运行是否齐全；
- 表格是否能追溯到逐 episode 记录；
- SR/CR/SPL 公式单元测试是否通过；
- CI 重采样单位是否为 map/trajectory cluster；
- 文中方法名称与 config `method_id` 是否一一对应；
- 论文图是否混用 oracle-route 和 goal-route；
- 主迁移/Proxy/Strict 是否隔离，主迁移是否混入 Proxy loss，Proxy 图表是否错误声称 Car–Dog；
- Swap 是否双向且使用目标具身约束；
- 所有链接、命令和 checkpoint hash 是否有效；
- 从空临时目录运行 `reproduce.sh` 是否成功。

## 8. 人工最终验收

负责人重点检查：

1. 表 1 是否准确回答 nuPlan→ANYmal 迁移，Proxy 是否仅作为接口/拒绝审计且未形成性能排序；
2. 表 2/图 2 是否排除了共享分支常数坍缩；
3. 表 3/图 3 是否同时支持目标保持和具身可行；
4. 负结果、失败场景和方法局限是否诚实保留；
5. 任何读者是否能从 `README.md` 运行命令并找到原始证据。

主迁移验收与 Proxy 验收分别记录；Proxy 通过只代表代理管线 `APPROVED`，Strict 仍保持 `BLOCKED_WAITING_DATA`。若核心假设未被支持，应修正论文主张，而不是删除不利结果。

本阶段 agent 完成复现核验和最终报告后停止，由项目负责人决定论文结论与后续扩展实验。
