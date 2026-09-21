---
stage: 03
plan_version: "1.2.3"
status: NOT_STARTED
protocol_version: score-decomp-transfer-v1.2.3
depends_on: [stage_02]
profiles: [transfer_primary, proxy_pair_auxiliary, strict_pair]
requires_human_approval: true
---

# 阶段 03：统一数据、三 Profile、固定弧长与无泄漏 Route-set

> 前置：[阶段 02](./02_protocol_and_baseline_freeze.md) 已通过。  
> 主迁移为 nuPlan Car→ANYmal；当前真实 Tartan Diff↔ANYmal 只保留非科学配对诊断，Proxy 仅运行 synthetic/interface fixture；未来 Strict Car–Dog 数据复用同一接口。

## 1. 阶段目标

建立跨平台共用 schema，并生成相互隔离的 `transfer_primary` 正式 manifest、`proxy_pair_auxiliary` 诊断 fixture 和 `strict_pair` 等待状态。ANYmal 目标数据生成 1%、10%、100% 嵌套预算和 same-map unseen-episode split，不主张未见地图泛化。未来再接入严格 Car–Dog 数据。非 oracle 输入不读取预测时刻之后的 GT。

## 2. 样本定义

每个普通样本至少包含：

```yaml
sample_id: str
domain: nuplan | tartanground | paired_dataset
embodiment: str                 # 真实平台名；不强制限定为 car/dog
scene_id: str
trajectory_id: str
anchor_index: int
timestamp: float
observation:
  ego_history: [H, state_dim]
  local_map: {...}
  goal_xy_local: [2]
  route_lanes: [R, L, D]
target:
  future_se2: [num_points, 3]
  valid_mask: [num_points]
metadata:
  route_source: map_goal | oracle_future
  map_id: str
```

所有配对样本额外包含：

```yaml
pair_id: str
platform_a_sample_id: str
platform_b_sample_id: str
platform_a_embodiment: str
platform_b_embodiment: str
common_scene_id: str
common_goal_id: str
T_a_to_common: [3, 3]
T_b_to_common: [3, 3]
pair_valid: bool
pair_level: proxy | strict
pair_source: tartan_cross_robot | future_car_dog
supports_claims:
  pipeline_validation: bool
  car_dog_scientific_claim: bool
pair_quality_flags: [str]
provenance: {...}
```

这里的 A/B 是数据接口角色：`proxy_pair_auxiliary` 只以 synthetic Diff/ANYmal fixture 验证字段与交换接口，不能用于训练或科学统计；`strict_pair` 中才映射为真实 Car/Dog。代码、缓存键和指标聚合均使用真实 `embodiment` 名称，不能把 fixture 或被拒绝的真实轨迹伪装成 Car/Dog。

## 3. 固定弧长轨迹与 RouteSpec

所有长度只从配置读取：

```yaml
trajectory:
  representation: xy_cos_sin
  length_mode: fixed_arc_length
  length_candidates_m: [8, 10, 12, 15, 20]
  min_train_coverage: 0.90
  length_selection: largest_candidate_meeting_coverage
  length_m: derived_from_train_audit
  num_points: 80
  resampling: normalized_arc_length
  preserve_timestamps: true
route:
  condition_mode: route_set
  source: map_goal
  max_candidates: 6
  deduplication: topology_and_overlap
```

`RouteSetBuilder` 只接受当前可观测地图、固定目标、当前状态和平台中立图参数，输出最多 6 条拓扑去重候选。候选不足时保留真实数量并使用 mask，禁止复制同一路径伪造多样性。现有 `build_model_features(future_gt, ...)` 保留为 `OracleRouteBuilder`，只用于回归与上界。

弧长选择只读取 train 数据：源 Car 按 v1.2.2 冻结的 50,000 条统计审计集计算单侧 95% Wilson 下界，目标 ANYmal 对全部 train moving windows 计算精确覆盖率；Proxy 不参与选择。选取两个正式分支判据均达到 90% 的最大候选值。若没有候选满足，Stage 03 状态为 `BLOCKED` 并提交分布，不得降低阈值、替换抽样或读取 val/test 后选择。

## 4. 需要新增的代码

```text
tartan/research_score/data/
├── schema.py
├── canonicalize.py
├── route_builder.py
├── trajectory_spec.py
├── representation_bridge.py
├── nuplan_dataset.py
├── tartan_dataset.py
├── paired_dataset.py
├── split_builder.py
├── budget_sampler.py
├── audit.py
└── manifests.py
tartan/research_score/scripts/
├── build_manifests.py
├── audit_pairs.py
└── run_stage03.sh
tartan/research_score/tests/
├── test_canonicalize.py
├── test_route_no_future.py
├── test_route_set.py
├── test_length_selection.py
├── test_representation_bridge.py
├── test_splits.py
└── test_pairs.py
```

### 4.1 nuPlan 源数据兼容约束

`nuplan_dataset.py` 必须复用源 Diffusion-Planner 的 observation、state、map/route 特征构建与数值归一化语义；不得为统一 Tartan 接口而重新估计源 normalizer。源 checkpoint 的 8 s/80 帧输入输出单独保留为 `source_temporal` 回归空间。`representation_bridge.py` 负责显式转换到 `fixed_arc_length_80` 研究空间，并报告有效覆盖、裁剪、padding 和往返误差；两种空间不得被宣称天然等价。

nuPlan 缓存至少以以下内容组成 key：`sample_id + source_preprocess_version + route_spec_hash + source_checkpoint_hash + schema_version`。manifest 同时记录缓存生成命令和代码 commit。任一项变化都必须重建缓存，禁止静默复用旧特征。

正式源索引使用 Diffusion-Planner 原生 `diffusion_planner_training.json` 及其 NPZ 文件。NPZ 中的 `ego_agent_future(80,3)` 才是 `source_temporal_8s_80` 的轨迹依据；Pluto cache 中 `(8,3)` 的 trajectory 不是该模型训练缓存，只能作为失败诊断保留。百万条索引只做流式 provenance、路径存在性和重复项审计，不以“索引已覆盖”冒充“百万文件内容已逐个验证”。

源 Car 固定弧长的审计规则如下：

1. 对规范化 NPZ 路径计算 SHA256 并排序，在读取任何覆盖率结果前冻结前 50,000 条；保存完整 ID 列表、算法版本和列表 hash。
2. 只从这 50,000 条读取 `ego_agent_future`，按实际 `xy` 累计弧长和 80 点/10 Hz 物理 horizon 执行 `moving_planning` / `stop_or_short` 分流。
3. 对每个候选长度计算 moving 样本覆盖率及单侧 95% Wilson 下界；源 Car 以 Wilson 下界不低于 90% 为合格条件。
4. Tartan ANYmal 使用全部 train moving windows 的精确覆盖率，不抽样；val/test 均不得用于选择。
5. 在两个正式分支均合格的候选中选择最大值。若没有候选满足条件，Stage 03 必须阻断，不得更换抽样、增加样本或降低阈值。

该 50,000 条集合是统计审计集，不自动等于后续训练预算。正式训练 manifest 仍由完整源索引和实际训练配置生成，并明确区分 `index_coverage`、`length_audit_sample`、`content_validation_sample` 与 `training_materialization`。

## 5. 统一轨迹变换

用于一致性和 swap 的轨迹先转换到共同固定长度空间：

- 当前机体坐标系，米制；
- `xy + cos(yaw) + sin(yaw)`；
- 几何轨迹按归一化弧长重采样为固定点数；
- 速度/真实时间作为具身信息，不进入共享一致性；
- 未覆盖共同空间的部分使用 mask，不裁剪成伪一致。

必须同时保留原始带时间轨迹，供导航、速度和闭环指标使用。

## 6. split 与预算

固定一次生成，之后所有方法复用：

- `train/val/test` 按完整 episode 分组；先按是否含至少 10 个 `moving_planning` window 分层，再在层内按固定 seed/SHA256 排序分配；
- 配对按 `pair_id + common_scene_id` 整体切分；
- 未见地图测试集的 `map_id` 不得出现在 train/val；
- 目标平台 1% 和 10% 是 100% train 的嵌套子集；
- 使用 5 个联合重复 ID；每个 ID 同时决定数据抽样和后续训练 seed，不与另 5 个 seed 交叉；
- 抽样单位是完整轨迹/episode，不是随机窗口；
- val 不随预算变化，测试集完全固定；
- 记录每个 split 的地图数、轨迹数、窗口数、距离与地形分布。
- `transfer_primary` 的 source 为 nuPlan Car、target 为 ANYmal；不读取 Proxy pair loss；
- `proxy_pair_auxiliary` 只含 synthetic/interface fixture 与真实 Diff↔ANYmal 的拒绝诊断，不生成正式训练 split 或预算；
- `strict_pair` 当前只生成 schema fixture 和 `BLOCKED_WAITING_DATA` 状态；
- 正式与诊断命名空间严格隔离，不允许合并统计。

## 7. Proxy 与 Strict 审计

`pair_valid=true` 均需满足场景/目标/坐标/有效区间/split 审计。额外规则：

- 场景、起点、目标和任务语义对齐；
- 两个坐标变换可逆且重投影误差低于预先固定阈值；
- 两条轨迹有效区间满足规划 horizon；
- 没有同一 pair/scene 跨 split；
- 配对来源和处理版本完整；
- 不因结果是否成功来筛选配对；
- Proxy manifest 必须强制 `pipeline_validation=true`、`car_dog_scientific_claim=false`；
- Strict manifest 才允许 `car_dog_scientific_claim=true`，且必须包含未来提供方的 provenance 与对齐审计；
- 报告和输出路径必须包含 `transfer_primary`、`proxy_pair_auxiliary` 或 `strict_pair`，禁止混合聚合。

审计阈值写入配置并由 val/audit 集确定，不在 test 上调整。未通过 pair 保存原因。当前 Strict 缺失不是 Proxy 管线阻塞，但 Strict 正式实验状态保持 `BLOCKED`。

## 8. 必须通过的测试

- `length_m` 不来自带 hash 的训练集审计产物时 Preflight 失败；
- 非 oracle route builder 的依赖图不含 `future_gt`；
- 构造 route 后更改 future GT，输入张量保持完全不变；
- route-set 候选数不超过 6、mask 正确、拓扑/重合率去重有效；
- 没有候选满足 90% 训练覆盖时必须阻断而非自行降阈值；
- train/val/test 的 `map_id`、`trajectory_id`、`pair_id` 交集为空；
- 1% ⊂ 10% ⊂ 100%；
- A/B canonical trajectory 变换可往返重建到容差范围；
- 同一 nuPlan 样本经原 loader 与统一 adapter 得到的源模型张量逐字段一致；
- 修改预处理版本、RouteSpec 或 checkpoint hash 时缓存 key 必须改变；
- 源 observation/state normalizer 的 hash 与阶段 02 冻结值一致；
- mask 后的 padding 不贡献 loss；
- 数据加载多 worker 可重复；
- 审计至少 30 个覆盖 train/val/test 与 moving/stop 的目标路线可视化；Proxy 仅保留 synthetic/interface fixture 与非科学配对诊断；
- 伪造 Proxy claim 为 Car–Dog 时 Preflight 必须失败。

## 9. 运行与产物

```bash
# 从服务器项目根目录启动
set -a; source ./local.env; set +a
cd "$PROJECT_ROOT"

$PYTHON_BIN -m tartan.research_score.preflight \
  --stage 03 --profile transfer_primary \
  --config tartan/research_score/configs/stage03_v1_2_3.yaml \
  --report "$OUTPUT_ROOT/preflight/stage03_transfer_primary"

$PYTHON_BIN -m pytest tartan/research_score/tests/test_canonicalize.py \
  tartan/research_score/tests/test_route_no_future.py \
  tartan/research_score/tests/test_route_set.py \
  tartan/research_score/tests/test_length_selection.py \
  tartan/research_score/tests/test_representation_bridge.py \
  tartan/research_score/tests/test_splits.py \
  tartan/research_score/tests/test_pairs.py -q

$PYTHON_BIN -m tartan.research_score.scripts.build_native_v122 \
  --config tartan/research_score/configs/stage03_v1_2_3.yaml \
  --output "$OUTPUT_ROOT/01_data_audit" --run-id "$RUN_ID"

$PYTHON_BIN -m tartan.research_score.scripts.audit_pairs \
  --profile proxy_pair_auxiliary --pair-root "$TARTAN_ROOT" \
  --output "$OUTPUT_ROOT/01_data_audit/proxy_pair_auxiliary/pairs"
```

产物：正式 `transfer_primary` 的 split/budget manifest、`selected_length.json` 及训练统计 hash、`route_set_audit.parquet`、拒绝原因统计、至少 30 个可视化样本、数据 hash、`review_packet.md` 和 `stage_report.md`。Proxy 只产生拒绝诊断，Strict 只产生 `BLOCKED_WAITING_DATA` manifest。

## 10. 人工验收标准

- 人工抽查 Proxy 配对叠加图；
- 主输入不含 future leakage；
- split 和预算嵌套测试全部通过；
- 主迁移 manifest 明确为 nuPlan→ANYmal，且不含 Proxy 科学结论；
- Proxy 的 Diff↔ANYmal 合格/拒绝数量透明；
- 负责人确认训练集规则选出的固定弧长和 route-set 可视化；
- Strict 部分明确 `BLOCKED_WAITING_DATA`，但已用合成 fixture 验证 adapter/schema 可接入。

验收前停止，不进入模型实现。
