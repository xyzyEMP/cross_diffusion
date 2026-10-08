# Proxy Training Protocol

本文档定义两个跨 embodiment 实验。两者均使用 Omni 和 Diff 训练、ANYmal
作为完全未见的测试 embodiment；实验 B 在实验 A 的基础上加入 Diff–Omni 配对数据
和因果模块。

当前状态：**P0–P7原正式实验COMPLETE；用户授权的validation消融原因诊断已完成**。唯一执行合同见`experiment_execution_plan/00_overall_progress.md`；实施按其第3节接口、第5节门禁及第6节CLI，不从历史讨论推断设置。

## 1. 统一数据划分

| Embodiment | Train | Validation | Test |
|---|---:|---:|---:|
| Omni | 80% | 20% | 0 |
| Diff | 80% | 20% | 0 |
| ANYmal | 0 | 0 | 100% |

Omni 和 Diff 分别以完整 trajectory 为单位做确定性 80/20 划分，再合并各自的
train 和 validation。划分 seed 固定为 `20260911`。

```text
Omni-train + Diff-train -> train
Omni-val   + Diff-val   -> validation
全部 ANYmal             -> unseen-embodiment test
```

禁止按 frame、局部窗口或 segment 随机切分。必须先冻结 trajectory manifest，再从
各 split 内生成窗口或配对；同一 trajectory 的所有派生样本只能属于一个 split。

每个 embodiment 的取整规则为：

```text
n_val = max(1, round(0.2 * n_trajectories))
n_train = n_trajectories - n_val
```

现有文档记录 Diff 5 条、Omni 6 条 trajectory；若实际数据不变，对应为 Diff 4/1、
Omni 5/1。正式实验前必须按实际目录重新核验。

## 2. 实验设计

| 设置 | A：Baseline | B：Causal Pair |
|---|---|---|
| 训练/验证划分 | Omni + Diff trajectory split | 与 A 完全相同 |
| ANYmal test | 全部 trajectory | 与 A 完全相同 |
| 模型 | 带共同历史条件的 `Diffusion_Planner` | 带相同共同历史条件的 `ScoreDecompositionPlanner` |
| 训练样本 | 非配对窗口 | 相同基础样本 + Diff–Omni train pairs |
| 损失 | `L_diff` | `L_diff + λ_inv L_inv + λ_swap L_swap + λ_sep L_sep` |
| 共同历史输入 | 16维历史latent + 3项物理摘要及mask | 与A相同 |
| 额外Embodiment ID | 无 | 训练ID dropout 0.2；ANYmal ID贡献0 |
| Checkpoint 指标 | Validation Episode-macro SR | 与 A 完全相同 |

### 2.1 实验 A：Diffusion Planner Baseline

使用带共同历史条件的 [`Diffusion_Planner`](diffusion_planner/model/diffusion_planner.py)，合并
Omni-train 和 Diff-train 进行训练。模型不接收 embodiment ID，不使用配对数据或因果
辅助损失。

从 Omni-val 与 Diff-val 分别构建冻结、非重叠的导航验证任务，再合并计算
Episode-macro SR。checkpoint 按合并验证集 Episode-macro SR 最大选择，同时分别报告
Omni 和 Diff 的 Episode-macro SR、SPL、CR 与 Goal Progress。

该实验衡量带共同历史条件的 Diffusion Planner 从 Omni+Diff 到 ANYmal 的跨 embodiment baseline；不能声称仍是未修改输入的普通backbone。

### 2.2 实验 B：Diff–Omni Pair + Causal Module

实验 B 使用与 A 完全相同的基础训练样本，并加入仅由训练 trajectory 生成的
Diff–Omni 配对数据。模型分解为：

```text
total = shared + embodiment_residual
```

其中 `shared` 来自 Diffusion Planner backbone；residual 使用embodiment条件并以零初始化adapter加到模型输出。旧无来源ability[10]不再是Proxy的物理能力合同；已批准历史表示如何接入共同backbone及B residual、如何参与pair/swap，按唯一计划3.3已收口路由实施。

损失为：

```text
L = L_diff + λ_inv L_inv + λ_swap L_swap + λ_sep L_sep
```

- `L_diff`：两侧轨迹的 masked denoising loss；
- `L_inv`：配对样本 shared 输出的一致性；
- `L_swap`：交换 Diff/Omni residual 后的预测一致性；
- `L_sep`：形态分离项，包括外部 adversarial loss 与 residual 正则化。

实现接口位于
[`score_decomposition.py`](tartan/research_score/model/score_decomposition.py) 和
[`losses.py`](tartan/research_score/training/losses.py)。2026-09-30：旧 synthetic-only proxy_objective/runner 已归档，losses.py 只保留 loss 原语；不能作为真实配对门禁。当前 adversarial classifier和
真实配对训练接口、218/20数据门禁和CPU完整恢复已通过；损失权重保持批准设置，GPU从P5b开始。

## 3. Diff–Omni 配对约束

- 只允许从 `Diff-train × Omni-train` 构造训练 pair；
- validation pair 的两侧必须都来自各自 validation trajectory，且不得用于训练；
- 不允许任何 pair 跨越 train/validation；
- 每个 pair 必须保存两侧 `trajectory_key`、segment ID 和匹配指标；
- 重叠窗口不能被视为独立 trajectory 样本；
- 配对数据只给实验 B 使用，实验 A 不读取 pair manifest。

Diff–Omni pair 由几何相似轨迹构成，属于 observational matching，不等价于真实
intervention 或 counterfactual pair。

## 4. 共同训练设置

两项实验必须保持以下设置一致：

- 相同 trajectory manifest、基础窗口和 ANYmal test manifest；
- 相同 8 m / 80 点轨迹表示、坐标系、上下文特征和 normalizer；
- 相同 nuPlan checkpoint 初始化；
- 相同 batch size、优化器、初始学习率、更新预算上下限、学习率调度与早停规则，以及随机种子；
- 相同冻结导航验证任务和 checkpoint 选择标准；
- 相同基础 denoising sample 顺序。

默认沿用 [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)：batch size 64、AdamW、
learning rate `1e-4`、最多 10,000 次 target updates、每 250 次运行一次冻结导航验证、
至少训练 5,000 次，并在连续 10 次验证的 Episode-macro SR 无提升后早停。

2026-10-02 用户确认Q1：A/B各自根据自身验证集SR调度学习率及早停。
公平性要求共同预算和规则，不要求实际停止update或逐步学习率轨迹相等；必须分别报告实际更新次数与LR历史。
共同scheduler为ReduceLROnPlateau（mode=max、factor=0.5、patience=2、min_lr=5e-6）。
Q2–Q4的后续批准见下文；所有批准均不表示Proxy代码或实验已完成。

2026-10-02 Q4部分确认：导航代理圆半径Diff/Omni均0.50m、ANYmal0.35m，属于工程假设；A/B几何一致。B训练时ID贡献以概率0.2屏蔽，ANYmal推理ID贡献固定0，不使用未训练ID行。一维固定能力未获批准，已由下述历史条件方案取代。


### 4.1 已批准的历史条件（已实施并通过CPU验收）

用户确认：有效性优先，采用16维自由历史latent＋3项显式物理摘要，A/B使用相同输入、公共模块结构和初始化；不将latent各维命名为物理能力。历史为anchor之前20帧的真实位置/朝向及有效mask；3项摘要为批准观测参考系前向速度、左向速度、yaw rate在有效历史间隔内的RMS，单位m/s、m/s、rad/s，须先核实时间基准，不用路径切线代替真实参考点朝向。摘要描述近期运动，不是最大能力。缺失显式mask，不混用速度和每帧位移单位。

不增加物理辅助planner损失、VAE/KL或对比训练目标，不扩大seed/训练消融矩阵。保留有来源的静态约束记录；已批准的代理圆半径仅用于共同几何。

当前简化8m导航任务内固定任务起点历史context及摘要。ANYmal只在最终推理使用该任务起点已观测过去；编码器和normalizer冻结，不进行测试时拟合、跨测试任务统计或未来记录回填。冻结表示后的物理读出诊断仅在Omni/Diff train拟合、val评价，不参与planner训练或checkpoint选择。

本项合同已实施并通过真实CPU smoke；以下为批准接口依据；Q2/Q3随后于20261002T042210Z另行获用户确认。历史索引/单位/缺失、共同与B分支路由、pair/swap、checkpoint及cache接口以唯一计划3.1–3.4为准；禁止Proxy沿用旧ability[10]。


### 4.2 Q2/Q3及历史路由确认（20261002T042210Z）

用户同意上一轮收口方案：base64，B每步额外32对；B denoising为base加两侧pair loss均值；inv/swap/sep权重0.1/0.1/0.01，classifier与GRL具体定义按唯一计划Q2，不增加物理辅助损失或权重搜索。固定8m/80点从真实anchor重建，以Diff anchor为共同frame，无ICP；同split配对的中心距离≤3m、首尾各<1m、对应点平均<1m且最大<2m，完整有效、来源可追溯，train pair为零停止B。保留各自真实goal，属于近似observational匹配。

共同历史可进入shared，latent本身不直接做平台分类对抗；classifier仍约束shared轨迹输出。pair各用自身真实历史，在共同frame仅交换残差输出，不交换历史/目标/观测。新公共条件初始零贡献，公共结构及初始化在A/B一致；准确接口见唯一计划3.3，已实施并通过CPU连通。

### 4.3 唯一合法初始化与断点恢复

Proxy A/B分别从**同一份未经Tartan/Omni/Diff/ANYmal微调的原始nuPlan预训练backbone**独立开始；不得使用原迁移实验1/10/100%微调权重、adapter训练结果、任何旧ANYmal权重、另一Proxy任务结果或smoke checkpoint作初始化。即使A/B都从同一份旧微调权重开始，也不能消除ANYmal测试污染。

当前记录的源权重路径为`/zeron-vepfs/tjqc/cross-diffusion/checkpoints/model.pth`，配套`checkpoints/args.json`及其normalizer。路径/文件名及strict load通过仅证明位置/结构，不单独证明训练来源。执行时复用已有源下载/交接记录，保存来源、实际绝对路径及配置；若来源无法排除目标数据微调，停止初始化并补清来源，不靠重命名或默认哈希证明。

首次初始化只加载原始backbone参数；历史模块、B条件/残差/classifier按批准初始化创建，不继承旧optimizer、scheduler、AMP scaler、update或早停状态，正式update从0开始。共同新增模块用同一seed和构造顺序保证初始权重相同，条件投影和B residual初始零贡献。主干严格加载后再组装新增模块，不能用全局strict=False掩盖缺失。

仅同一Proxy任务、同一RUN_ID的真实断点恢复可加载自身last checkpoint，并恢复完整模型/optimizer/scheduler/scaler/RNG/数据游标/选模状态；它与首次预训练初始化是不同操作。A的结果不能作为B起点。本项目“全量训练数据”仅指各自80% train split内全部合格窗口，不包含validation或ANYmal。

最佳 checkpoint 首先最大化 Episode-macro SR；并列时依次比较更高的
Episode-macro SPL、更低的 Episode-macro CR、更高的 Goal Progress，以及更早的
checkpoint。Denoising loss 只作为诊断指标，不参与选模。

实验 B 必须单独报告 pair 数量、辅助损失权重和额外计算量。

## 5. ANYmal Unseen-Embodiment Test

全部 ANYmal trajectory 仅用于最终测试，不参与训练、验证、normalizer 拟合、
checkpoint 选择或阈值调整。

离线测试至少报告：

- masked denoising loss；
- 有效点 ADE/FDE；
- trajectory-macro 主指标；
- 全窗口 micro 指标和逐 trajectory 结果。

若已生成兼容的冻结闭环任务，再报告 SR、Collision Rate、SPL、Goal Progress、
Stuck Rate 和 Route-Failure Rate。实验 A 与 B 必须使用完全相同的 ANYmal test
manifest。

## 6. 结果解释

A/B 对比可以检验：加入 Diff–Omni 配对和因果模块后，ANYmal unseen-embodiment
测试是否优于具有相同历史条件的 Diffusion Planner baseline。

仅凭该实验不能声称：

- 已识别真实 embodiment 因果效应；
- 已实现物理 intervention 或真实反事实预测；
- 已验证 unseen-map 或真实机器人泛化。

本轮用户已批准`observed_lcam_front_reference`：真实前左相机pose的NED→NWU XY/yaw定义参考点，历史、RMS、目标、occupancy、配对与圆形代理导航使用相同参考系；不推断基座外参，body来源仍null。时间按已验证metadata/generator dt，不称实测时钟。圆半径和D024不变；RMS与SR/CR/SPL不能解释为真实基座能力或实体机器人安全。来源及批准为`/tj-share/cross_diffusion_workdir/runs/20261002T045353Z_proxy_cpu/reference_approval.json`、`time_source_evidence.json`、`frame_source_evidence.json`与共享trajectory_evidence旁注。

CPU_RUN_ID/DATA_ID=`20261002T045353Z_proxy_cpu`，全部CPU门禁通过；真实train/val/test cache4280/457/3071，合格pair218/20，配对侧cache396/30，ANYmal确定性导航296覆盖24条。A/B真实2步、必要梯度、B sampler、strict重载与完整停机续训比较PASS；69 tests通过/1 CUDA skip。上述为CPU阶段验收；当前GPU实际P5b已通过，P6/P7全部完成，ANYmal离线和闭环均覆盖24条轨迹。准确验收、清理与GPU续接见唯一计划第8节及run/cpu_report.json；源证据保留，不重复旧审计。CPU/GPU smoke不得作为正式初始化或科学结果。

## 已批准候选补充（2026-10-03）

在完整trajectory冻结split不变、共同frame8m/80点门禁不变的前提下，将原10m候选重建结果与现有base窗口每10帧锚点直接搜索的native8m真配对按两侧sample_id去重合并。用户批准本DATA_ID得到218train/20validation；不采用逐帧1485对密集候选。调用现有pair.run_pair_mining --mode reconstruct并传--base-window-dir；批准记录位于CPU run的pair_expansion_approval.json。20对仍来自同一对完整轨迹，窗口相关；配对loss仅诊断，选模仍为完整validation navigation SR。更多独立覆盖不靠修改split或放宽阈值。

最终完成节点：`/tj-share/cross_diffusion_workdir/runs/20261002T184606Z_proxy_seed11`，status/summary为COMPLETE，report.md包含离线与闭环指标。A/B各5000更新、best250；ANYmal每方法24条轨迹、296闭环任务。宏平均SR A0.353451/B0.121843，CR A0.285444/B0.410261，SPL A0.345613/B0.100769。运行进程已退出，无需继续启动；单地图/单seed/观测配对与批准参考系限制仍适用。


## 用户授权的追加原因诊断

原因诊断已完成（2026-10-03）：RUN_ID=`20261002T203425Z_proxy_diagnosis`，DATA_ID/CPU_RUN_ID=`20261002T045353Z_proxy_cpu`。 消融只在run配置改权重：no_swap令λswap=0；no_inv令λinv=0；no_sep令λsep=0；pair_only令三项辅助权重为0；base_only再令pair_denoising_weight=0，保留随机流/前向但不产生配对监督梯度。该字段默认1，原正式B行为不变。 五项均独立原始nuPlan初始化，使用相同冻结数据/seed及SR选模规则；no_swap/no_inv/no_sep/pair_only各5000更新、best250，base_only6000更新、best3500。validation SR分别24.25%/23.58%/24.92%/22.33%/26.92%，对照A30.25%/完整B26.83%。没有追加ANYmal模型前向。删除swap不改善；仅基础监督与完整B接近（差0.08pp）；仅配对去噪较低。B shared-only推理SR0%、ID-off21.08%，显示分支依赖；validation67个起点有效且初始连通任务中32条记录路径触发既定碰撞代理，说明模仿目标与安全评价存在冲突。报告/summary/status/configs/commands/logs及五项必要last/best checkpoint在`/tj-share/cross_diffusion_workdir/runs/20261002T203425Z_proxy_diagnosis`。状态COMPLETE，pending为空，无下一训练命令；完成项不重跑。只有故障恢复才在确认无进程后显式同RUN_ID执行continue.sh。研究限制与待修订项：2条独立val trajectory、单seed；还未修订目标/代理几何、控制器朝向及shared监督，不宣称ANYmal改进。
