# 研究内容一开题大纲：执行条件感知的目标—拓扑显式意图表征

## 1. 研究内容定位

### 1.1 总体课题中的位置

总体课题研究汽车轨迹规划知识向轮式、全向和四足执行平台的少样本迁移。研究内容一不直接解决足端控制，也不把不同平台的完整轨迹强制对齐，而是先建立一个位于“场景感知”和“平台轨迹生成”之间、能够训练、解释和评价的显式中间决策接口。

研究内容一拟解决：

> 如何从无人工意图标注的汽车与地面机器人轨迹中，自动提取“去哪里、从哪类通行结构前进”的结构化意图，并在平台能力约束下将其转化为可执行路线条件，最终支持目标条件的闭环轨迹规划？

### 1.2 研究边界

- 输出为未来 8 秒的机体级 SE(2) 轨迹，不输出 ANYmal 足端、关节或电机命令。
- 输入首先使用离线构建的局部地图、历史状态和二维目标坐标；在线 RGB/LiDAR 到局部地图的感知前端不作为研究内容一的主变量。
- 研究内容一完成显式意图结构、自动标签提取、意图预测、可行性门控和目标条件闭环验证。
- 严格的车—狗反事实成对一致训练、少样本迁移规律和选择性一致损失的完整研究作为后续研究内容；本部分提供其变量、接口和基础反事实诊断。

## 2. 现有基础及问题

### 2.1 已完成基础

- 已复现 nuPlan 上的 Diffusion-Planner 预训练模型与仿真任务。
- 已打通 TartanGround 位姿到模型张量、轨迹预测、地形 guidance、候选重排序、速度适配、运动学投影、安全截停和静态地图运动学闭环。
- 已形成 `origin_v1`、`final` 和 `town_focused_v2` 等可复现实验结果。

### 2.2 当前 baseline 的关键限制

1. 当前 `route_lanes` 由未来 8 秒真值轨迹生成，属于 oracle-route 条件，不是自主路线决策。
2. Tartan 适配中周围目标和静态物体张量均为零，神经网络的主要有效输入接近 oracle route。
3. 真实历史运动主要用于模型输出后的速度调整，尚未形成学习型历史 token。
4. 语义点云用于 guidance、后处理与评价，尚未进入学习型场景 Encoder。
5. 当前闭环每次重规划仍读取后续记录路线，不能评价“给定目标自主选路”。

因此，现有结果只能作为 oracle-route zero-shot upper bound；研究内容一需要建立无未来泄漏的 goal-conditioned baseline。

## 3. 科学问题与研究假设

### 3.1 科学问题

1. 哪一层决策信息可以在汽车、差速轮式、全向轮式与四足平台之间保持统一语义？
2. 如何从地图和记录轨迹自动提取该信息，而不依赖语言或人工逐条标注？
3. 如何避免将平台能力导致的合理路线差异错误地当作不一致？
4. 显式意图是否能改善少样本条件下的目标推进、通道选择和闭环规划，而不仅是改善 latent 可视化？

### 3.2 研究假设

- H1：局部子目标比无结构 latent 更可解释、更容易稳定监督。
- H2：局部子目标不足以区分同终点的多条绕障路线，加入场景条件的通行拓扑分布能够改善多通道选择。
- H3：共享任务偏好与具身可执行选择必须分层；无条件对齐最终通道会在能力分叉场景产生负迁移。
- H4：使用固定场景与目标、仅改变执行能力的受控干预，可以检验中间表示是否保持任务稳定性并在能力边界处合理切换策略。
- H5：将预测意图显式接入 Diffusion-Planner 的 `cross_c` 与路线调制条件，可使意图真正影响轨迹生成，而不只是辅助分类。

## 4. 理论变量与因果结构

定义：

$$
S=\text{环境结构},\quad G=\text{任务目标},\quad E=\text{平台执行能力}
$$

$$
U=f_{\mathrm{shared}}(S,G,\mathcal P)
$$

$$
F_E=f_{\mathrm{feas}}(S,\mathcal P,E)
$$

$$
Z_E=f_{\mathrm{gate}}(U,F_E,E),\qquad
\tau_E=D(S,Z_E,E)
$$

其中：

- $\mathcal P=\{P_1,\ldots,P_K\}$ 为同一场景下平台中立的候选通行结构；
- $U$ 为不直接依赖平台能力的共享任务效用；
- $F_E$ 为候选路线在平台 $E$ 下的可行性；
- $Z_E$ 为经过能力门控后的具身可执行意图；
- $\tau_E$ 为平台轨迹。

执行条件干预固定 $S,G,\mathcal P$，仅改变：

$$
do(E=e_{\mathrm{car}}),\quad do(E=e_{\mathrm{diff}}),\quad do(E=e_{\mathrm{anymal}})
$$

研究内容一不声称从观察数据完全识别真实物理因果图，而是建立明确的结构因果假设，并通过受控能力替换验证其机制。

## 5. 最终意图结构

### 5.1 共享任务意图

$$
Z^{\mathrm{shared}}=
[Z_{\mathrm{goal}},Z_{\mathrm{topo\_preference}},Z_{\mathrm{phase}}]
$$

- `goal`：固定空间尺度上的局部子目标；
- `topo_preference`：候选通行结构的共享任务效用，而不是全局固定“左/右/直”类别；
- `phase`：接近、通过、到达、停止等任务阶段，作为辅助监督。

### 5.2 具身可执行意图

$$
Z^E=[p_E(P_1),\ldots,p_E(P_K),v_E,\mathrm{reachable}_E]
$$

- 候选路线的最终平台分布；
- 平台相关速度趋势；
- 当前目标是否可达。

### 5.3 不进入共享意图的量

- 逐时刻精确位置、航向和曲率；
- 轮式转角或四足关节动作；
- 平台特定速度尺度；
- 足端接触序列。

## 6. 核心算法

1. 将 nuPlan lane graph 与 TartanGround 自由空间图统一为含位置、方向、连通性、净空和目标测地距离的路径图。
2. 在同一局部图上生成最多 6 条拓扑不同候选路径，并通过路径重合率与障碍侧签名去重。
3. 从训练未来轨迹与候选路径的几何、终点、航向和拓扑匹配距离，计算解析式软后验 $q_T(k)$，作为训练时未来教师。
4. 当前观测学生预测平台中立效用 $u_k^{shared}$。
5. 基于平台尺寸、曲率、坡度、台阶和支撑计算可行性 $m_k(E)$。
6. 通过

$$
p_E(k)=\operatorname{softmax}
\big(u_k^{shared}+\Delta u_k^E+\log(m_k(E)+\epsilon)\big)
$$

得到具身可执行路线分布。
7. 将 `goal token`、`topology token` 和 `ability token` 拼入 Diffusion-Planner 的 cross-attention context；将候选路线的概率加权编码作为 DiT 路线条件。
8. 输出平台级 SE(2) 轨迹，并在固定目标的滚动闭环中重新感知、重建候选、预测意图和重规划。

## 7. 数据构造

### 7.1 目标

- 不按固定未来秒数选目标；按记录轨迹累计弧长选取 15–25 m 远端目标。
- 目标必须位于典型规划空间范围之外，避免退化为 8 秒轨迹终点复制。
- 部署与闭环评价中使用固定二维目标，不读取后续真值路线。

### 7.2 局部地图

- 初始范围：机体前方 30 m、后方 10 m、左右各 20 m，共 40 m × 40 m；分辨率：0.25 m。该范围能够覆盖 15–25 m 的远端目标。
- 通道：occupancy、elevation、slope、step、support、semantic、unknown。
- 未观测区域保持 unknown，不填充为自由空间。

### 7.3 样本切分

- 以环境和完整轨迹为分组单位，禁止相邻滑窗跨训练/验证/测试集。
- 同时报告轨迹内分组测试和环境外测试。
- 少样本比例按完整目标平台轨迹抽取，不按窗口随机抽取。

## 8. 模型与代码落点

采用“新研究包装器 + 原模型最小兼容修改”，保留现有 zero-shot baseline：

```text
tartan/research/
├── configs/research_content_1.yaml
├── data/{schema,build_samples,bev_builder,graph_builder,candidates,labels,splits}.py
├── model/{history_encoder,terrain_encoder,path_encoder,feasibility_gate,
│          intention_bottleneck,cf_gtib_planner}.py
├── training/{dataset,losses,train_intention,train_joint}.py
├── evaluation/{open_loop,closed_loop_goal,intention_metrics,intervention_sweep}.py
└── tests/
```

原始 `diffusion_planner/` 仅为 DiT 增加可选的预计算路线编码入口；未提供该入口时保持原 checkpoint 行为不变。

## 9. 训练与评价闭环

### 9.1 训练顺序

1. 生成并审计无泄漏的目标—拓扑训练数据。
2. 独立训练意图预测器，只有通过候选覆盖率、路径分类与校准验收后才接扩散规划器。
3. 加载汽车 checkpoint，先冻结原主干，训练新增模块。
4. 联合训练意图与 ego diffusion loss，逐步解冻 Decoder 后层。
5. 进行平台能力干预诊断，确认共享效用稳定且具身策略在可行性边界处变化。

### 9.2 开环评价

- Route Top-1/Top-3、候选覆盖率、subgoal error、ECE；
- ADE/FDE、目标进度、路线偏离、地形违规和可行轨迹率；
- oracle intention、predicted intention、打乱 intention 三组因果使用性检查。

### 9.3 目标条件闭环评价

- episode 开始时固定目标；后续不得读取日志未来路线；
- 每 1 秒根据执行状态重新构建局部地图、候选路径和意图；
- 执行仿真保持 10 Hz；
- 到达目标、超时、碰撞/几何失败或连续无进展时终止；
- 报告目标成功率、SPL、碰撞/地形失败、重规划次数、意图切换率和恢复成功率。

## 10. 对比与消融

### 10.1 必要基线

1. 当前 Oracle-route zero-shot upper bound；
2. Goal-only；
3. Goal + BEV，不使用拓扑；
4. 固定 DLM：左/直/右/停；
5. Topology-only；
6. Goal + Topology；
7. Goal + Topology + Feasibility Gate；
8. 完整研究内容一模型。

### 10.2 必要消融

- 移除局部目标；
- 移除拓扑候选；
- 移除解析未来软后验，改为硬标签；
- 移除可行性门控；
- 不向 Diffusion Decoder 注入 intention token；
- 继续使用 oracle route；
- 使用未来教师的 oracle 上界；
- 打乱意图标签。

## 11. 预期创新点

1. 提出适用于汽车道路图和四足自由空间图的目标—拓扑结构化意图接口。
2. 使用解析未来后验从无人工意图标注轨迹中自动提取场景条件意图，并在推理时由当前观测学生预测。
3. 将共享任务效用与具身可执行路线显式拆分，通过平台能力门控避免将合理策略分叉误判为不一致。
4. 将显式意图接入扩散轨迹生成并建立不依赖未来记录路线的目标条件闭环评价。

## 12. 预期成果与判定条件

- 一套可复现的 nuPlan/TartanGround 统一目标—拓扑数据格式；
- 一套自动标签生成与人工审计工具；
- 一个显式输出候选路线分布和局部子目标的意图模型；
- 一个兼容预训练 Diffusion-Planner 的目标条件轨迹规划器；
- 一套开环、目标闭环和执行条件干预实验；
- 证明或否定以下假设：显式 Goal–Topology 是否比无结构或 Goal-only 表征带来稳定的少样本迁移收益。

## 13. 主要研究依据

- [Diffusion-Based Planning for Autonomous Driving with Flexible Guidance](https://arxiv.org/abs/2501.15564)：基础扩散规划模型与 guidance 接口。
- [TNT: Target-driveN Trajectory Prediction](https://proceedings.mlr.press/v155/zhao21b/zhao21b.pdf)：目标状态作为可解释运动模式。
- [MTR](https://papers.neurips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html)：全局意图定位与局部运动细化。
- [PGP](https://proceedings.mlr.press/v164/deo22a.html)：图遍历表示多模态路线并条件生成轨迹。
- [Topology-Driven Parallel Trajectory Optimization](https://arxiv.org/abs/2401.06021)：用不同同伦类别区分绕障策略。
- [Trajectron++](https://arxiv.org/abs/2001.03093)：训练后验可看未来、推理先验只看历史的生成建模结构。
- [GNM](https://arxiv.org/abs/2210.03370) 与 [ViNT](https://arxiv.org/abs/2306.14846)：跨机器人统一目标条件导航和少样本适配。
- [IntentionNet](https://arxiv.org/abs/2407.03122)：高层 intention 作为规划器与低层导航器接口，并在 Spot 上验证。
- [XSkill](https://proceedings.mlr.press/v229/xu23a.html) 与 [UniAct](https://openaccess.thecvf.com/content/CVPR2025/html/Zheng_Universal_Actions_for_Enhanced_Embodied_Foundation_Models_CVPR_2025_paper.html)：共享中间行为表示与平台动作解码的跨具身依据。
- [IAIL](https://pubmed.ncbi.nlm.nih.gov/41849566/)：跨机器人对齐高层意图而非复制底层运动。
- [Toward Causal Representation Learning](https://doi.org/10.1109/JPROC.2021.3058954) 与 [Interventional Causal Representation Learning](https://proceedings.mlr.press/v202/ahuja23a.html)：通过干预发现和检验高层因果变量的理论背景。
- [Domain Generalization using Causal Matching](https://proceedings.mlr.press/v139/mahajan21b.html)：同一基础对象在域干预下进行表示匹配，而非只做粗粒度域不变。
- [TartanGround](https://tartanair.org/tartanground/)：多平台、位姿、语义占据、点云和本体数据基础。
