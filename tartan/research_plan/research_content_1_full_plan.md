# 研究内容一完整计划书

## 执行条件感知的目标—拓扑显式意图提取与闭环扩散轨迹规划

## 1. 研究目标、对象与可验证命题

本研究以已复现的 ICLR 2025 Diffusion-Planner 为汽车端预训练基础，以 nuPlan、TartanGround 差速轮式 `diff`、全向轮式 `omni` 和四足 `anymal` 轨迹为数据对象，研究一个明确位于场景表示与连续轨迹之间的决策变量。最终系统接收历史状态、当前局部地图、固定任务目标和平台能力，输出未来 8 秒的机体级 SE(2) 轨迹，并在运动学执行反馈下滚动重规划。

本研究不把“意图”定义为不可解释的隐向量，也不把全数据集统一的“左、直、右、停”当作核心类别。意图由当前场景中的候选通行结构定义：模型首先判断应向哪个局部子目标推进以及哪些通行结构有利于任务，再根据执行平台的尺寸、转向、横移、坡度和台阶能力形成最终可执行选择。

需要通过实验判断的命题如下：

1. `Goal + Topology` 是否比 `Goal-only`、固定 DLM 类别和无结构 latent 更准确地描述同终点多通道行为。
2. 解析式未来软后验能否在没有人工意图标签的条件下提供可靠监督，并由只看当前信息的学生网络预测。
3. 将共享任务效用与平台可行性拆分后，模型是否能在固定场景与目标、只改变执行能力时保持任务层稳定，并在能力边界处合理切换通道。
4. 显式意图接入 Diffusion-Planner 后，是否改善目标条件开环轨迹和不依赖日志未来路线的闭环成功率。
5. 在 1%、5%、10% 目标平台数据下，显式意图是否比普通 checkpoint 微调具有更好的样本效率。

如果上述命题没有在 held-out trajectory 和 held-out environment 上获得一致证据，则不能宣称显式意图改善了跨执行条件迁移。

## 2. 文献依据与本研究采用的具体部分

| 研究依据 | 已有工作提供的证据 | 本研究采用的部分 | 不直接照搬的部分 |
|---|---|---|---|
| [Diffusion-Planner](https://arxiv.org/abs/2501.15564) | Transformer 场景编码、DiT 轨迹扩散与可插拔 guidance | 预训练轨迹生成主干、`cross_c` 和路线条件接口 | 汽车专属 route 语义和 nuPlan 输入不能直接用于四足 |
| [TNT](https://proceedings.mlr.press/v155/zhao21b/zhao21b.pdf) | 目标状态可表达中期运动模式，并可条件生成完整轨迹 | 显式局部空间目标 | 不使用固定时间终点作为跨平台目标 |
| [MTR](https://papers.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html) | 全局意图定位与局部运动细化优于同一 latent 回归全部模式 | intention query 读取场景与路径 token | 不直接使用按车辆终点聚类的静态 motion queries |
| [PGP](https://proceedings.mlr.press/v164/deo22a.html) | lane-graph traversal 可表示横向路线多模态并条件解码 | 候选图路径、真实未来到路径的自动监督 | 将 lane graph 推广为平台中立通行图；不假设固定 lane 语义 |
| [Topology-Driven Planning](https://arxiv.org/abs/2401.06021) | 不同同伦类别可代表不同绕障策略，再分别优化轨迹 | 候选路径拓扑去重和通道级决策 | 第一版为静态地图，不处理动态行人的时空同伦 |
| [Trajectron++](https://arxiv.org/abs/2001.03093) | 训练时后验可使用未来，推理时先验只依赖历史 | 未来条件教师—当前观测学生原则 | 使用解析且有物理语义的候选路径后验，不使用无语义 CVAE 类别 |
| [GNM](https://arxiv.org/abs/2210.03370)、[ViNT](https://arxiv.org/abs/2306.14846) | 异构移动机器人数据可使用统一目标到达接口联合训练和少样本适配 | 二维目标条件、平台尺度标准化和跨机器人训练协议 | 第一版不引入图像目标和视觉导航主干 |
| [IntentionNet](https://arxiv.org/abs/2407.03122) | local path 或离散局部动作可以作为高层规划与低层导航接口；Spot 有实机验证 | 将显式路线意图作为轨迹解码条件 | 不使用只能表达有限情况的 DLM 作为最终核心表示 |
| [XSkill](https://proceedings.mlr.press/v229/xu23a.html) | 跨具身共享技能表示可以条件化 diffusion policy | 中间表示与扩散执行解码分离 | 第一版不使用无监督技能聚类和 Sinkhorn 码本 |
| [UniAct](https://openaccess.thecvf.com/content/CVPR2025/html/Zheng_Universal_Actions_for_Enhanced_Embodied_Foundation_Models_CVPR_2025_paper.html) | 通用行为表示可由具身信息还原为不同动作 | 共享决策与平台执行细节分层 | 不建立通用离散动作词表 |
| [IAIL](https://pubmed.ncbi.nlm.nih.gov/41849566/) | 异构机器人适配应对齐高层意图而非复制底层运动 | 对齐层级的研究原则 | 不依赖自然语言意图标注和大规模候选检索 |
| [Causal Matching](https://proceedings.mlr.press/v139/mahajan21b.html) | 同一基础对象在域干预后的表示匹配比粗粒度域不变更合理 | 固定地图与目标、只改变执行能力的配对 | 不声称一般性域不变即可识别全部因果变量 |
| [Interventional Causal Representation Learning](https://proceedings.mlr.press/v202/ahuja23a.html) | 干预数据有助于高层因果变量识别 | 显式记录 `do(E)` 与响应变量 | 本研究的规划器干预是简化模型下的受控干预，不等价于真实机器人完美干预 |
| [TartanGround](https://tartanair.org/tartanground/) | 同时提供轮式、四足、位姿、点云、语义占据和本体数据 | 多执行平台、场景点云和记录轨迹 | 数据内不同平台轨迹不能天然视为严格反事实配对 |

以上文献分别支持目标、拓扑、跨具身中间表示和干预方法，但不存在一篇工作已经证明本研究的完整组合有效。因此结果必须通过后文定义的消融和闭环实验验证。

## 3. 当前代码事实与改造约束

### 3.1 原模型真实接口

原训练循环 [train_epoch.py](../../diffusion_planner/train_epoch.py) 输入：

```text
ego_current_state
neighbor_agents_past
lanes / lane speed / traffic
route_lanes
static_objects
```

监督为 ego 和邻居未来轨迹。原损失 [loss.py](../../diffusion_planner/loss.py) 将当前状态与未来轨迹拼接，加入扩散噪声，然后分别计算 `ego_planning_loss` 和 `neighbor_prediction_loss`。

原 Encoder [encoder.py](../../diffusion_planner/model/module/encoder.py) 只编码邻居、静态物体和 lanes；`ego_current_state` 不会成为 Encoder token。原 Decoder [decoder.py](../../diffusion_planner/model/module/decoder.py) 使用：

```text
cross_c = encoder_outputs["encoding"]
route_lanes → RouteEncoder → route_encoding
y = route_encoding + timestep_encoding
```

`cross_c` 进入 DiT cross-attention，`y` 进入 adaLN 调制。因此显式意图必须进入这两个条件路径中的至少一个，否则辅助意图头可能不影响轨迹。

### 3.2 当前 Tartan 适配事实

[features.py](../data/features.py) 当前：

- 使用未来真值轨迹构造 `lanes` 和 `route_lanes`；
- `neighbor_agents_past` 全零；
- `static_objects` 全零；
- `ego_current_state` 为固定数组。

[evaluate.py](../evaluation/evaluate.py) 与 [closed_loop.py](../evaluation/closed_loop.py) 使用真实历史估计速度并进行输出后处理，但闭环每次重新规划仍从日志未来读取剩余路线。

因此研究代码必须新建，不覆盖当前 baseline。当前结果保留并重命名为概念上的：

```text
Oracle-route zero-shot upper bound
```

### 3.3 向后兼容原则

1. 保留 `tartan/data`、`tartan/planning`、`tartan/evaluation` 作为 zero-shot baseline。
2. 新方法全部位于 `tartan/research`。
3. 原 checkpoint 首先严格加载进未修改的基础 Encoder/Decoder；新增模块随后独立初始化。
4. 对原 DiT 仅增加可选的 `route_encoding` 参数；未提供时继续调用原 `RouteEncoder(route_lanes)`，保证旧执行脚本不变。
5. 新 checkpoint 保存基础权重来源、模块版本、配置、数据 manifest、git commit 和归一化统计。

## 4. 因果变量、意图结构和模型输出

### 4.1 结构变量

设：

$$
S=(B,\mathcal G,\mathcal P)
$$

其中 $B$ 是局部 BEV，$\mathcal G$ 是平台中立通行图，$\mathcal P$ 是候选路径集合。任务目标为 $G$，执行能力为：

$$
E=[w,l,r_{min},v_{max},\omega_{max},a_{lat},s_{max},h_{max}]
$$

分别表示足迹宽度、长度、最小转弯半径、最大速度、最大角速度、横移能力、最大坡度和最大台阶。

### 4.2 共享任务效用

对每个候选路径 $P_k$，共享网络输出：

$$
u_k^{shared}=f_{shared}(H,B,\mathcal G,G,P_k)
$$

该值不直接输入平台能力，表示路线对完成任务的基础价值，例如目标推进、长度、未知区域和环境结构关系。

### 4.3 平台可行性与残差

几何可行性模块输出：

$$
m_k(E)\in[0,1]
$$

平台轻量残差输出：

$$
\Delta u_k^E=f_E(H,P_k,E)
$$

最终分布：

$$
p_E(k)=\operatorname{softmax}
\left(u_k^{shared}+\Delta u_k^E+\log(m_k(E)+10^{-6})\right)
$$

这里不把所有平台差异推给规则门控：规则处理明确的几何边界，`delta_u` 学习规则无法表达的软偏好。为防止平台残差吞掉共享效用，使用小容量两层 MLP，并对其幅值施加正则。

### 4.4 显式意图输出

模型对每个样本必须输出：

```python
{
    "shared_utility": float[B, K],
    "feasibility": float[B, K],
    "embodiment_residual": float[B, K],
    "route_probability": float[B, K],
    "local_subgoal": float[B, 2],
    "goal_token": float[B, 1, 192],
    "topology_token": float[B, 1, 192],
    "ability_token": float[B, 1, 192],
    "reachable_probability": float[B, 1],
}
```

这些量全部保存到评测 NPZ，不能只保留最终轨迹。

## 5. 数据定义与防泄漏规则

### 5.1 统一样本模式

在 `tartan/research/data/schema.py` 定义带版本号的 dataclass：

```python
@dataclass
class GoalTopologySample:
    schema_version: str
    sample_id: str
    source_dataset: str
    environment: str
    trajectory_id: str
    robot_type: str
    anchor_index: int
    history: np.ndarray                 # [21, state_dim]
    task_goal: np.ndarray               # [2]
    terrain_bev: np.ndarray             # [7, 160, 160]
    graph_nodes: np.ndarray             # [N, node_dim]
    graph_edges: np.ndarray             # [2, M]
    candidate_paths: np.ndarray         # [6, 32, path_dim]
    candidate_mask: np.ndarray          # [6]
    candidate_subgoals: np.ndarray      # [6, 2]
    embodiment: np.ndarray              # [8]
    feasibility: np.ndarray             # [6]
    future_trajectory: np.ndarray       # [80, 4], training/eval only
    route_posterior: np.ndarray         # [6], training/eval only
    label_confidence: float
```

每个文件同时保存 `input_fields` 与 `supervision_fields`。DataLoader 在验证和测试时断言 `future_trajectory`、`route_posterior` 只进入 metric/loss 分支，不进入 `model_inputs`。

### 5.2 历史状态

历史长度固定 21 帧、10 Hz，覆盖当前及过去 2 秒。每帧包含：

```text
x, y, cos(yaw), sin(yaw), vx, vy, yaw_rate, valid
```

全部变换到当前机体坐标系。速度使用中心差分；序列边缘使用单边差分。速度与角速度分别进行 Savitzky–Golay 或长度 5 的稳健平滑，但位置真值不平滑。

新增 `history_encoder.py`，使用与原 AgentFusionEncoder 相同风格的时间 MLP-Mixer，将历史压成 `[B,1,192]` 的 `ego_history_token`。不得继续只用固定 `ego_current_state`。

### 5.3 任务目标

#### TartanGround

从 anchor 后的记录轨迹计算累计弧长。训练时从 `[15,25] m` 均匀采样一个目标距离，选择第一个达到该弧长的未来位姿；不足 15 m 的片段不进入目标规划训练。目标转到当前局部坐标系。

为防止目标与 8 秒监督终点接近导致复制，要求：

```text
goal_arc_length - future_arc_length >= 3 m
```

若 ANYmal 长轨迹局部速度较低导致可用样本过少，可把下限调整为 `max(10 m, future_arc_length + 3 m)`，但调整必须记录进配置并报告样本数量，不能按测试结果临时改变。

#### nuPlan

优先沿 scenario route roadblocks 的中心线累计 20 m 取得目标；如 roadblock 断裂，则使用与 route 连通的最近 lane/connector；只有预处理失败时使用日志未来位置作为弱目标。弱目标样本增加 `goal_source=logged_future` 标记，单独做消融。

#### 闭环

episode 开始时固定目标的全局坐标。每次重规划只把该固定目标转换到新的自车局部坐标；不得重新从日志选目标，也不得读取 anchor 后的 GT route。

### 5.4 数据切分

切分单位为完整 `(environment, robot_type, trajectory_id)`，禁止同一长轨迹的不同窗口跨集合。

实施两个协议：

1. `trajectory-held-out`：每个主环境和机器人按确定性 hash 划分 70%/15%/15% 轨迹。
2. `environment-held-out`：训练环境与测试环境完全不同，用于检验场景泛化。

少样本 1%、5%、10% 按目标平台完整轨迹数量采样，每个比例使用 5 个固定 seed。不能按窗口随机取 1%，否则相邻帧会夸大数据量和性能。

## 6. 局部 BEV 与通行图构建

### 6.1 BEV 坐标和通道

`bev_builder.py` 生成以当前机体为基准、航向朝前的 40 m × 40 m 局部栅格。纵向范围为后方 10 m 到前方 30 m，横向范围为左右各 20 m，使 15–25 m 的远端目标仍位于局部图内：

```text
resolution = 0.25 m
shape = 160 × 160
channels = 7
```

通道定义：

1. `occupancy`：确定障碍概率；
2. `elevation`：相对当前地面高度，截断到 `[-2,2] m`；
3. `slope`：局部平面拟合坡度，归一化到 `[0,1]`；
4. `step`：邻域最大高差，截断到 `[0,0.6] m`；
5. `support`：平台中立的点云表面覆盖率，不使用具体足迹；
6. `semantic`：压缩地物类别 ID 的 embedding 索引或 one-hot 分组；
7. `unknown`：无可靠点云覆盖的显式掩码。

现有 [terrain.py](../data/terrain.py) 的高程缓存可作为全局表面来源，但新 builder 必须：

- 不把大孔洞无条件插值成地面；
- 保存每格点数和观测置信度；
- 对高程、语义和覆盖率使用同一坐标变换；
- 在缓存文件名中加入分辨率、环境和算法版本 hash。

### 6.2 平台中立自由空间

构图前只应用对所有平台都明确成立的硬障碍：建筑主体、墙、深空洞和数据集边界。台阶、窄通道、陡坡不在这一阶段全部删除，而作为候选路径属性保留，供不同 `E` 判断。

这样同一个场景、起点和目标能得到同一候选集合，执行条件干预才只改变可行性，而不是同时改变候选生成过程。

### 6.3 图构建算法

`graph_builder.py` 执行：

1. 对基础自由空间做欧氏距离变换，得到 clearance；
2. 使用 medial-axis skeleton 或 Voronoi 骨架；
3. 将度不等于 2 的像素转为图节点；
4. 将节点之间的连续骨架压缩为 polyline edge；
5. 删除长度小于 0.75 m 且不连接目标方向的毛刺；
6. 每 0.5 m 重采样 edge；
7. 在 edge 上聚合 elevation、slope、step、support、semantic、unknown；
8. 使用 Dijkstra 计算到目标最近可达节点的测地距离。

nuPlan 端通过 `nuplan_graph_builder.py` 将 lane 和 connector 变为同一 schema。共享 path encoder 只接收两端都有的字段：

```text
x, y, sin(yaw), cos(yaw), normalized_arclength,
clearance, goal_geodesic_distance, unknown_fraction
```

限速、交通灯、lane type 作为 car adapter 的私有 token，不进入共享路径字段。

### 6.4 构图失败处理

若起点离图超过 1.0 m、目标附近 2.0 m 内无节点或不存在候选路径：

- 样本标记为 `graph_failure`；
- 不用于 route 分类训练；
- 保留用于 `reachable=0` 辅助任务；
- 单独报告失败率，不能静默丢弃导致测试集变容易。

## 7. 候选拓扑生成与标签提取

### 7.1 候选生成

`candidates.py` 使用以下确定性流程：

1. 从起点最近节点到目标最近节点运行 Yen K-shortest paths，先取最多 20 条。
2. 计算任意两条路径的有向边 Jaccard overlap；overlap > 0.8 时仅保留基础代价更低者。
3. 对局部占据障碍连通分量计算路径的障碍侧签名；签名一致且几何 Fréchet 距离低于阈值的路径归为同一拓扑类。
4. 每个拓扑类保留代价最低路径。
5. 按基础代价排序取最多 6 条，不足部分 padding 并设置 `candidate_mask=False`。
6. 每条路径沿弧长重采样为 32 点。

平台中立基础代价：

$$
J_0(P)=L(P)+2.0R_{unknown}(P)+0.5d_G(P_{end})
$$

系数仅决定候选排序，不作为最终模型最佳参数；修改必须作为标签版本变化。

### 7.2 局部子目标

对每条候选，在路径弧长 10 m 处取子目标；路径不足 10 m 时使用末端：

$$
g_k=P_k(\min(10\text{ m},L(P_k)))
$$

子目标依赖空间结构，不依赖平台 8 秒能走多远。

### 7.3 未来轨迹软后验

`labels.py` 将训练未来轨迹投影到每条候选，计算：

$$
D_k=0.45D_{proj}+0.20D_{end}+0.10D_{heading}+0.25D_{topology}
$$

- `D_proj`：未来轨迹点到候选 polyline 的双向平均投影距离；
- `D_end`：未来末端到候选前向可达部分的距离；
- `D_heading`：投影点处切向与轨迹航向差；
- `D_topology`：真实轨迹与候选的障碍侧签名是否一致。

各距离先以训练集稳健尺度归一化，不能直接把米、弧度和二值量相加。软后验：

$$
q_T(k)=\frac{e^{-D_k/T}}{\sum_{j=1}^{K} e^{-D_j/T}},\quad T=0.25
$$

置信度：

$$
c=1+\frac{\sum_{k=1}^{K} q_T(k)\ln q_T(k)}{\ln K}
$$

如果真实轨迹到所有候选的平均投影距离均超过 1.5 m，则 `candidate_covered=False`，该样本不参与 route KL，但仍参与图失败统计和轨迹 baseline。

### 7.4 人工审计

实现 `audit_labels.py`，固定随机种子分别抽取：

- nuPlan 500 个样本；
- ANYmal 500 个样本；
- diff/omni 合计 500 个样本；
- 至少 30% 来自多候选场景；
- 覆盖高、中、低置信度区间。

审计界面同时显示 BEV、目标、真实轨迹、6 条候选、后验概率和可行性。人工记录：

```text
candidate set valid?
best candidate matches behavior?
subgoal valid?
map failure?
ambiguous?
```

高置信样本人工最佳路线一致率未达到 90%，或候选覆盖率未达到 95% 时，不进入模型联合训练；先修构图与标签。

## 8. 平台能力和可行性门控

### 8.1 能力参数来源

第一版沿用 [config.py](../config.py) 中已用于 zero-shot 评价的尺寸、坡度和台阶上限，但将参数移动到版本化 YAML：

```yaml
embodiments:
  diff:
    footprint_width: 0.70
    footprint_length: 1.10
    max_speed: 5.0
    max_yaw_rate: 1.5
    max_curvature: 1.5
    lateral_motion: 0.0
    max_slope_deg: 25.0
    max_step_height: 0.20
  omni: ...
  anymal: ...
```

这些是几何代理参数，不宣称等于真实机器人的完整动力学边界。后续实机或仿真标定必须生成新能力配置版本。

### 8.2 路径属性

每条候选计算：

```text
minimum_clearance
curvature_p95
slope_p95 / slope_violation_fraction
step_p95 / step_violation_fraction
support_p05 / support_failure_fraction
unknown_fraction
```

### 8.3 连续门控

不直接使用全 0/1 硬门控，以免阈值附近梯度消失。对每个属性定义 sigmoid margin，例如净空：

$$
m_{width}=\sigma((c_{min}-w/2-b)/\tau_c)
$$

坡度：

$$
m_{slope}=\sigma((s_{max}-s_{p95})/\tau_s)
$$

总可行性采用加权几何平均并保留下限：

$$
m_k=\exp\left(\sum_j\alpha_j\log(m_{k,j}+10^{-6})\right)
$$

明确的建筑碰撞或地图外路径设为 0；未知区域只降低概率，不直接判死。

### 8.4 干预接口

`intervention_sweep.py` 固定同一 `sample_id` 的地图、起点、目标和候选，依次改变：

```text
width: 0.4 → 2.2 m
min turning radius: 0 → 8 m
max slope: 10° → 40°
max step: 0.05 → 0.40 m
lateral mobility: 0 / 1
```

每次保存 `shared_utility`、`feasibility`、`route_probability` 和最终轨迹。`shared_utility` 不应随 `E` 改变；最终选择可以在可行性边界附近改变。

## 9. 模型实现

### 9.1 代码目录

```text
tartan/research/
├── configs/
│   ├── research_content_1.yaml
│   └── embodiments.yaml
├── data/
│   ├── schema.py
│   ├── build_samples.py
│   ├── bev_builder.py
│   ├── graph_builder.py
│   ├── nuplan_graph_builder.py
│   ├── candidates.py
│   ├── labels.py
│   ├── audit_labels.py
│   └── splits.py
├── model/
│   ├── history_encoder.py
│   ├── terrain_encoder.py
│   ├── graph_encoder.py
│   ├── path_encoder.py
│   ├── feasibility_gate.py
│   ├── intention_bottleneck.py
│   ├── model_space_transform.py
│   └── cf_gtib_planner.py
├── training/
│   ├── dataset.py
│   ├── collate.py
│   ├── losses.py
│   ├── train_intention.py
│   └── train_joint.py
├── evaluation/
│   ├── intention_metrics.py
│   ├── open_loop.py
│   ├── closed_loop_goal.py
│   ├── intervention_sweep.py
│   └── summarize.py
└── tests/
```

### 9.2 Terrain Encoder

`terrain_encoder.py` 使用轻量 ResNet 或 patch CNN：

```text
input  [B, 7, 160, 160]
stem   stride 2
stage1 stride 2
stage2 stride 2
stage3 stride 2
output feature map [B, 192, 20, 20]
```

将 feature map 展平后，不保留全部 400 token；按图节点和候选路径位置 bilinear sample，得到局部 terrain token，控制显存并保持与路径对应。

### 9.3 Graph/Path Encoder

每条候选 `[32,path_dim]` 经过：

1. point MLP `path_dim → 128`；
2. 加入弧长位置编码；
3. 2 层 Transformer Encoder；
4. masked mean pooling；
5. 投影到 192 维。

输出 `[B,6,192]`。候选之间再经过 2 层 self-attention，使模型比较不同路线而不是分别打分。

### 9.4 History Encoder

输入 `[B,21,8]`，经过 point MLP、时间位置编码和 3 层 Mixer/Transformer，输出 `[B,1,192]`。历史 token 同时进入共享意图模块和 Diffusion `cross_c`。

### 9.5 Intention Bottleneck

模型输入：

```text
history_token       [B,1,192]
terrain_path_tokens [B,K,192]
candidate_tokens    [B,K,192]
goal_token_input    [B,1,192]
base_scene_tokens   [B,N,192]
```

设置 `goal_query` 与 `route_query` 两个学习 query，通过 cross-attention 读取上述 token。共享效用头对每个候选计算：

```python
u_shared = shared_scorer(
    concat(route_query.expand(K), candidate_tokens, goal_relation)
)
```

能力 MLP 输出 `ability_token`，小容量残差头输出 `delta_u_E`。可行性由 `FeasibilityGate` 给出。最终计算 `p_E`、subgoal 和 reachable probability。

### 9.6 与 Diffusion-Planner 的连接

`cf_gtib_planner.py` 持有原 `Diffusion_Planner` 实例。执行：

```python
base_outputs = base_model.encoder(base_inputs)
intent = intention_model(research_inputs, base_outputs["encoding"])

base_outputs["encoding"] = torch.cat([
    base_outputs["encoding"],
    history_token,
    intent["goal_token"],
    intent["topology_token"],
    intent["ability_token"],
], dim=1)

route_encoding = weighted_route_encoding(
    candidate_route_encodings,
    intent["route_probability"],
)

decoder_outputs = base_model.decoder(
    {**base_outputs, "route_encoding": route_encoding},
    base_inputs,
)
```

候选路线必须分别通过原 `RouteEncoder` 后再按概率加权：

$$
r=\sum_kp_E(k)E_{route}(P_k)
$$

不能先对不同路径坐标加权再编码，因为两条绕障路径的坐标平均可能穿过障碍。

为复用原 `RouteEncoder`，`candidate_to_route_lanes.py` 将每条 `[32,path_dim]` 候选转换成原接口的 `[25,20,12]`：有效部分按弧长切成最多 25 个重叠 polyline segment，每段重采样 20 点；前 4 维填写 `x, y, dx, dy`，左右边界由局部 clearance 截断后生成，交通灯设为 unknown。原 `RouteEncoder` 实际只读取前 4 维，因此 TartanGround 不需要伪造限速或信号灯。实现时将 `[B,K,25,20,12]` reshape 为 `[B*K,25,20,12]`，一次调用原 RouteEncoder 得到 `[B,K,192]`，再使用 `p_E` 加权。

对 [decoder.py](../../diffusion_planner/model/module/decoder.py) 和 DiT 的唯一兼容改动为：如果 `encoder_outputs` 提供 `route_encoding`，则直接使用；否则保持原 `route_lanes → RouteEncoder` 流程。DPM-Solver 的训练和推理参数字典都必须传递该可选张量。

### 9.7 模型空间尺度

原 checkpoint 的位置 normalizer 按汽车位移训练，直接用 ANYmal 米制轨迹会产生尺度偏移。新增 `model_space_transform.py`：

1. 从各平台训练轨迹统计 8 秒有效位移的中位数 `D_E`；
2. 以汽车 `D_car` 为参考，定义 `s_E=D_car/D_E`；
3. 进入原 Diffusion 模型前，仅将路线和轨迹 `x,y` 乘 `s_E`；
4. 航向、坡度和物理可行性仍在真实物理空间计算；
5. 输出后将 `x,y` 除以 `s_E`。

`s_E` 只能由训练 split 统计。需要对比：固定尺度、平台尺度和当前 zero-shot 后处理速度适配，确认收益来源。

## 10. 损失函数与训练参数

### 10.1 意图损失

路线软监督：

$$
L_{route}=c\,D_{KL}(\operatorname{sg}(q_T)\|p_E)
$$

局部目标：

$$
g_T=\sum_kq_T(k)g_k,\qquad
L_{goal}=c\,\operatorname{SmoothL1}(\hat g,g_T)
$$

可达性标签由候选集合是否存在 `m_k>0.5` 且图连接目标生成：

$$
L_{reach}=\operatorname{BCE}(\hat r,r)
$$

不可行概率质量：

$$
L_{feas}=\sum_kp_E(k)(1-m_k)
$$

平台残差正则：

$$
L_{res}=\frac{1}{K}\sum_{k=1}^{K}\left|\Delta u_k^E\right|
$$

### 10.2 扩散损失

沿用原 `x_start` ego diffusion loss。TartanGround 没有动态目标跟踪监督时：

```python
neighbor_prediction_loss = 0
loss = ego_planning_loss + intention losses
```

不能制造全零邻居未来后仍按有效目标计算邻居损失。

### 10.3 总损失

研究内容一第一版：

$$
L=L_{diff}
+1.0L_{route}
+2.0L_{goal}
+0.5L_{reach}
+0.5L_{feas}
+0.05L_{res}
$$

权重属于初始设置。仅在 validation 上按预先定义搜索空间调整：

```text
lambda_route ∈ {0.5, 1.0, 2.0}
lambda_goal ∈ {1.0, 2.0}
lambda_feas ∈ {0.2, 0.5, 1.0}
lambda_res ∈ {0.01, 0.05, 0.1}
```

测试集不能参与权重选择。

### 10.4 课程训练

联合训练前 20% steps 使用教师后验 `q_T` 加权候选 route encoding；20%–60% 按线性概率在 `q_T` 与学生 `p_E` 间 scheduled sampling；最后 40% 完全使用学生 `p_E`。验证和测试始终只用学生。

与直接 argmax teacher forcing 相比，route embedding 的概率混合保持可微，并避免几何路径平均。

## 11. 训练执行方案

### 11.1 数据预处理运行

配置文件必须记录：

```yaml
history_steps: 20
future_steps: 80
sample_rate_hz: 10
goal_arc_length_range_m: [15, 25]
bev_resolution_m: 0.25
bev_extent_m: 40
candidate_count: 6
candidate_points: 32
subgoal_lookahead_m: 10
label_temperature: 0.25
```

预处理生成：

```text
tartan/research/artifacts/data_v1/
├── samples/*.npz
├── graph_cache/*.npz
├── manifests/{train,val,test}.json
├── statistics.json
├── rejected_samples.csv
└── audit/
```

`statistics.json` 保存所有平台的速度、位移、candidate coverage、置信度和尺度统计。

### 11.2 独立意图训练

运行 `train_intention.py`，不加载 Diffusion Decoder。优化器 AdamW：

```text
lr = 3e-4
weight_decay = 1e-4
batch_size = GPU 可承受的最大值，目标 64
warmup = 5%
epochs = 50，按 val route NLL early stop
gradient clip = 5
mixed precision = bf16/fp16
```

每个 epoch 保存：route NLL、Top-1/Top-3、subgoal error、ECE、feasible probability mass、各平台和环境拆分。

只有同时满足下列条件才进入联合训练：

- validation candidate coverage ≥ 95%；
- 高置信人工 route 一致率 ≥ 90%；
- 学生 Route Top-3 明显高于按候选基础代价排序的非学习基线；
- 打乱目标后 route/subgoal 性能显著下降，证明模型确实使用目标；
- 测试推理代码通过未来字段访问审计。

### 11.3 联合训练

初始化：

1. 严格加载 `checkpoints/model.pth` 到原模型；
2. 加载已训练 intention checkpoint；
3. 新增投影层使用 Xavier，进入 `cross_c` 的残差门控初始为 0；
4. 前 5 epochs 冻结原 Encoder 与 Decoder；
5. 接着解冻 Decoder 最后一层和最后一个 DiT block，学习率为新增模块的 0.1；
6. 若 validation 仍改善，再解冻全部 Decoder；默认不解冻汽车 scene Encoder。

优化器使用参数组：

```text
new modules:      1e-4
decoder unfrozen: 1e-5
base encoder:     0 or 5e-6
```

early stop 依据组合分数：

$$
J_{val}=ADE+0.5FDE+2.0(1-success_{goal})+2.0R_{terrain}
$$

同时保留 route NLL 最佳 checkpoint，避免组合分数掩盖意图退化。

## 12. 开环评价协议

### 12.1 防泄漏条件

开环模型输入只含：

```text
history
fixed task goal
current local BEV/graph
candidate paths
embodiment
```

未来轨迹仅用于：

- 解析 posterior 评价；
- ADE/FDE；
- 人工行为一致检查。

测试运行时 monkey-patch 或 schema guard 禁止模型读取 `future_trajectory`、`route_posterior` 和 GT route。

### 12.2 意图指标

- Candidate Coverage：未来行为是否被候选集合覆盖；
- Route Top-1/Top-3：学生对解析后验最佳候选的预测；
- Route NLL/KL；
- Subgoal L2 error；
- ECE 与 Brier score；
- Feasible Probability Mass：$\sum_kp_E(k)m_k$；
- Reachability AUROC/F1；
- 多通道子集和能力边界子集单独报告。

### 12.3 轨迹指标

- ADE/FDE；
- 到任务目标测地距离减少量；
- route selection correctness；
- occupancy collision rate；
- slope/step/support violation；
- speed、acceleration、yaw-rate、curvature violation；
- 模型随机采样多样性和有效候选比例。

### 12.4 必要因果使用性检查

同一模型权重比较：

1. 正常 predicted intention；
2. oracle posterior intention；
3. 打乱 batch 内 intention；
4. 全零 intention token；
5. 目标镜像或替换。

如果打乱或置零 intention 不影响轨迹，说明 Decoder 忽略了瓶颈，不能声称显式意图参与决策。

## 13. 不依赖未来路线的目标条件闭环

### 13.1 新闭环文件

新增 `evaluation/closed_loop_goal.py`，不修改当前 [closed_loop.py](../evaluation/closed_loop.py)。每个 episode 保存：

```text
fixed_global_goal
executed_trajectory
planned_trajectories at every replan
candidate paths at every replan
shared utilities
feasibility vectors
route probabilities
selected topology IDs/signatures
termination reason
terrain/contact proxies
```

### 13.2 episode 初始化

从测试长轨迹选 anchor，并沿记录轨迹弧长 15–25 m 取得一个固定全局目标。记录轨迹只用于选择可比较的目标和离线评价；episode 开始后规划器不能访问 anchor 后轨迹。

初始条件设置：

```text
nominal
lateral offset +0.30 m, yaw +5°
lateral offset -0.30 m, yaw -5°
```

额外鲁棒性测试可使用 ±0.50 m、±10°，但与主指标分开。

### 13.3 每次重规划的确定流程

每 1 秒执行一次以下步骤：

1. 从执行器反馈的当前状态更新 2 秒历史窗口；
2. 将固定全局目标转换到当前局部坐标；
3. 从全局语义表面裁剪当前局部 BEV；
4. 重建或更新局部通行图；
5. 从当前位置到固定目标生成最多 6 条候选；
6. 计算当前平台可行性；
7. 意图网络预测共享效用与最终路线分布；
8. Diffusion-Planner 生成多条未来轨迹；
9. 只在相同意图条件内根据地形和运动学代价选择候选；
10. 执行前 1 秒轨迹，通过现有 `_execute_prefix` 的噪声运动学模型得到新状态；
11. 保存全部中间变量并进入下一次循环。

与现有闭环不同，第 2–5 步不调用日志未来路线。

### 13.4 终止条件

- `success`：当前位置到目标的图上距离 < 1.5 m，且最后 1 秒没有硬几何失败；
- `collision_failure`：足迹与确定障碍相交；
- `terrain_failure`：持续 1 秒以上 slope/step/support 超限；
- `no_route`：连续 3 次重规划不存在候选可行路径；
- `stuck`：连续 5 秒目标测地距离减少 < 0.5 m，且系统未合理安全停止；
- `timeout`：执行 40 秒仍未到达。

安全停止本身不算成功；若目标确实不可达，应以正确的 `unreachable` 识别单独计分。

### 13.5 闭环指标

- Goal Success Rate；
- SPL：Success weighted by normalized Path Length；
- 终点欧氏与测地距离；
- Collision/Terrain Failure Rate；
- Unreachable Detection F1；
- Recovery Success under perturbation；
- Replan Count 与平均推理时延；
- 意图切换率：连续重规划 topology signature 改变次数；
- 非必要切换率：候选可行性未显著变化时的切换；
- 能力边界响应延迟；
- 安全停止次数和错误停止率。

闭环报告必须同时给出整体、环境、平台、扰动和多通道难例拆分，并按原始长轨迹做 cluster bootstrap 95% 置信区间。

## 14. 对比、消融与统计检验

### 14.1 主要对比

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | Constant Velocity | 非学习参考 |
| B1 | 当前 Raw checkpoint + oracle route | 原迁移表现 |
| B2 | 当前 Terrain-aware + oracle route | 现有上界，不与自主目标方法混称公平对比 |
| B3 | Goal-only | 检验仅目标是否足够 |
| B4 | Goal + BEV direct diffusion | 检验不显式拓扑的目标规划 |
| B5 | DLM 左/直/右/停 | 检验固定类别限制 |
| B6 | Topology-only | 检验无局部目标情况 |
| B7 | Goal + Topology，无 feasibility gate | 检验门控必要性 |
| M | 完整研究内容一 | 目标、拓扑、能力门控、扩散轨迹 |

### 14.2 表征消融

- 无解析 soft posterior，改硬 argmin；
- 增加学习型未来教师，检验是否真有额外收益；
- 固定 8 秒终点代替空间子目标；
- 固定 DLM 代替候选 pointer；
- 不使用 unknown mask；
- 不使用 ego history token；
- 不做平台模型空间尺度变换；
- 不把 intention token 接入 Decoder，仅做辅助预测。

### 14.3 干预消融

- 不使用能力门控；
- 所有平台强制共享最终路线分布；
- 只用二值门控；
- 连续门控；
- 规则门控 + 平台残差；
- 打乱能力向量。

### 14.4 统计方法

- 样本误差不能直接视为独立，因为同一长轨迹窗口相关；
- 以 `(environment, robot_type, trajectory_id)` 为 cluster bootstrap 单位；
- 每项核心差值报告 estimate、95% CI 和 cluster count；
- 少样本实验对 5 个数据 seed 报告均值、标准差和轨迹聚类区间；
- 主要假设优先比较预注册的 M vs B3、M vs B4、M vs B7，避免大量未校正的探索性比较。

## 15. 单元测试、集成测试和失败保护

### 15.1 数据测试

- 局部/全局坐标往返误差 < 1e-4；
- 所有候选从起点连通到目标或显式标记 partial；
- padding candidate 概率严格为 0；
- future 字段不进入 model input；
- split 间 trajectory ID 不重叠；
- 目标弧长满足配置范围；
- BEV unknown 不被误当 free。

### 15.2 模型测试

- 原 checkpoint 在 `route_encoding=None` 时输出与修改前数值一致；
- route probability 和为 1；
- 全不可行时 reachable probability 下降且不产生 NaN；
- 对 `E` 做 sweep 时 `shared_utility` 完全不变；
- 候选概率加权发生在 route embedding 空间，而不是坐标空间；
- 梯度能够到达 history/terrain/path/intention encoders；
- Tartan 无邻居时 neighbor loss 为精确 0。

### 15.3 闭环测试

- 使用哨兵对象替代日志未来，任何访问立即抛错；
- 固定目标在坐标系更新后保持全局位置不变；
- 平坦空地图应能到达直线目标；
- 增大宽度跨过窄通道阈值时选路应发生可解释变化；
- 不可达目标应终止为 `no_route/unreachable`，不能无限循环；
- 每个 episode 随机种子和中间计划可复现。

## 16. 文件输出与实验可复现性

每次实验输出：

```text
tartan/outputs/research_content_1/<run_name>/
├── config_resolved.yaml
├── checkpoint_manifest.json
├── dataset_manifest.json
├── split_manifest.json
├── training_history.csv
├── intention_metrics.json
├── open_loop/
│   ├── per_sample_metrics.csv
│   ├── predictions/*.npz
│   └── report.md
├── closed_loop/
│   ├── per_episode_metrics.csv
│   ├── rollouts/*.npz
│   └── report.md
├── interventions/
│   ├── sweeps.csv
│   └── figures/
└── final_report/
```

`checkpoint_manifest.json` 必须记录：基础 checkpoint SHA256、新增 checkpoint SHA256、PyTorch/CUDA、训练 seed、git commit、数据 schema、label version 和 normalizer version。

## 17. 实施工期与阶段门

| 时间 | 必须完成的代码与结果 | 阶段门 |
|---|---|---|
| 第 1–2 周 | schema、目标生成、0.25 m BEV、图缓存 | 无 future leakage；地图坐标测试通过 |
| 第 3–4 周 | 候选生成、拓扑去重、解析后验、审计工具 | coverage ≥95%；高置信人工一致率 ≥90% |
| 第 5–6 周 | history/terrain/path encoder、独立意图训练 | Top-3 优于非学习路径代价基线；ECE 可接受 |
| 第 7–8 周 | wrapper、预计算 route encoding、模型空间变换、联合训练 | 原 checkpoint 兼容测试通过；意图打乱影响轨迹 |
| 第 9–10 周 | 无日志未来的目标闭环、终止与指标 | 空地图和真实地图 smoke test 通过 |
| 第 11–12 周 | 主要基线、表征消融、能力 sweep | 能力干预曲线和闭环结果完整 |
| 第 13–14 周 | 1/5/10% 少样本、多 seed、统计检验 | 形成可重复主表与置信区间 |
| 第 15 周 | 报告、失败案例、代码清理 | 所有结论可追溯到原始 CSV/NPZ |

阶段门不满足时不继续堆叠模块。例如候选覆盖不足时继续训练意图网络没有意义；意图打乱不影响轨迹时继续做跨平台一致损失也无法证明中间表征被使用。

## 18. 结果解释边界

研究内容一完成后，可以支持的结论是：

> 在离线地图或可靠局部地图输入下，目标—拓扑结构化瓶颈能够从无人工意图标注轨迹中提取可解释决策变量，并在执行能力条件下驱动轮式和四足平台的目标条件轨迹生成与简化运动学闭环。

不能直接支持的结论包括：

- 已经完成真实四足接触动力学规划；
- 已经证明真实世界因果关系被完全识别；
- 已经完成仅凭在线 RGB/LiDAR 的真实机器人部署；
- 几何 contact score 等价于真实足端稳定性；
- TartanGround `diff` 等价于 nuPlan 汽车。

论文和开题报告中应分别使用：

- Tartan 内 `diff/omni ↔ anymal`：验证同地图执行条件机制；
- `nuPlan car → Tartan anymal`：验证跨数据集少样本迁移价值。

两类实验回答不同问题，不能合并为同一个“车—狗严格反事实”结论。

## 19. 预期创新与失败判据

### 19.1 预期创新

1. 统一汽车 lane graph 与非结构化自由空间图的场景条件 Goal–Topology 意图接口。
2. 用解析未来后验替代人工语言或固定动作类别，从记录轨迹自动提取可解释意图。
3. 将平台中立任务效用、执行能力可行性和平台残差显式分离，使共享与分叉都有确定位置。
4. 将意图分布以 candidate route embedding 和 intention tokens 两条路径接入扩散规划器，并用干预与打乱实验验证其真实作用。
5. 建立不读取日志未来路线的目标条件闭环，避免当前 oracle route 掩盖自主决策问题。

### 19.2 失败判据

出现以下任一结果，应调整或否定当前意图结构：

- Candidate coverage 在高质量地图上仍低于 90%；
- 人工审计发现 topology 标签无法稳定区分行为；
- Goal + Topology 不优于 Goal-only 的多通道选择；
- oracle intention 也不能改善轨迹，说明意图接口与 Decoder 连接无效；
- 打乱 intention 后轨迹基本不变，说明模型绕过瓶颈；
- feasibility gate 使目标成功率下降但安全性没有改善；
- 受控 `do(E)` sweep 中共享效用随 E 大幅变化，说明模型分层失效；
- 少样本结果只在窗口随机切分有效，在 trajectory/environment-held-out 上消失。

这些失败结果同样具有研究价值，因为它们能够区分“表示结构错误”“标签错误”“Decoder 不使用意图”和“跨域先验本身无收益”四类原因。

## 20. 文献列表

1. Zheng et al. [Diffusion-Based Planning for Autonomous Driving with Flexible Guidance](https://arxiv.org/abs/2501.15564), ICLR 2025；[官方代码](https://github.com/ZhengYinan-AIR/Diffusion-Planner)。
2. Zhao et al. [TNT: Target-driveN Trajectory Prediction](https://proceedings.mlr.press/v155/zhao21b.html), CoRL 2020/2021 Proceedings。
3. Shi et al. [Motion Transformer with Global Intention Localization and Local Movement Refinement](https://papers.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html), NeurIPS 2022；[官方代码](https://github.com/sshaoshuai/MTR)。
4. Deo et al. [Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals](https://proceedings.mlr.press/v164/deo22a.html), CoRL 2021；[官方代码](https://github.com/nachiket92/PGP)。
5. de Groot et al. [Topology-Driven Parallel Trajectory Optimization in Dynamic Environments](https://arxiv.org/abs/2401.06021), IEEE T-RO。
6. Salzmann et al. [Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data](https://arxiv.org/abs/2001.03093), ECCV 2020。
7. Shah et al. [GNM: A General Navigation Model to Drive Any Robot](https://arxiv.org/abs/2210.03370), ICRA 2023。
8. Shah et al. [ViNT: A Foundation Model for Visual Navigation](https://arxiv.org/abs/2306.14846), CoRL 2023；[项目页与代码](https://general-navigation-models.github.io/vint/index.html)。
9. Gao et al. [IntentionNet: Map-Lite Visual Navigation at the Kilometre Scale](https://arxiv.org/abs/2407.03122)。
10. Xu et al. [XSkill: Cross Embodiment Skill Discovery](https://proceedings.mlr.press/v229/xu23a.html), CoRL 2023。
11. Zheng et al. [Universal Actions for Enhanced Embodied Foundation Models](https://openaccess.thecvf.com/content/CVPR2025/html/Zheng_Universal_Actions_for_Enhanced_Embodied_Foundation_Models_CVPR_2025_paper.html), CVPR 2025。
12. Chen et al. [Cross-robot behavior adaptation through intention alignment](https://pubmed.ncbi.nlm.nih.gov/41849566/), Science Robotics 2026。
13. Schölkopf et al. [Toward Causal Representation Learning](https://doi.org/10.1109/JPROC.2021.3058954), Proceedings of the IEEE 2021。
14. Ahuja et al. [Interventional Causal Representation Learning](https://proceedings.mlr.press/v202/ahuja23a.html), ICML 2023。
15. Mahajan et al. [Domain Generalization using Causal Matching](https://proceedings.mlr.press/v139/mahajan21b.html), ICML 2021。
16. Patel et al. [TartanGround: A Large-Scale Dataset for Ground Robot Perception and Navigation](https://tartanair.org/tartanground/), IROS 2025。
