# 开题方案（本地完善稿）

## 拟定题目

### 执行条件干预下的跨具身一致性决策表征、在线适配与闭环控制研究

可选的更工程化题目：

> 面向轮式与四足移动机器人的跨具身轨迹规划迁移与闭环控制方法研究

第一题更能突出因果框架和方法创新，建议作为开题主标题；第二题可作为项目或系统名称。

## 1. 研究背景与问题提出

汽车、差速轮式、全向轮式和四足机器人在导航任务中共享目标推进、避障和通道选择等高层决策规律，但在尺寸、运动学、速度、坡度和台阶通过能力上存在显著差异。直接迁移汽车规划器容易把“场景理解”和“汽车执行习惯”一并迁移；直接强制不同平台轨迹一致，又会抹去合理的具身差异。

本课题以已复现的 Diffusion-Planner 为源端规划基础，以 nuPlan 和 TartanGround 为主要数据来源，围绕三个递进问题展开：

1. 哪些轨迹分布成分应在执行条件改变时保持稳定？
2. 新平台能力未知或变化时，如何用少量数据和在线执行反馈快速适配？
3. 如何把规划轨迹可靠地转化为异构平台动作，并形成可复用的规划—控制闭环接口？

三个研究内容在科学问题上相互区分：研究内容一侧重一致性表征，研究内容二侧重具身适配，研究内容三侧重规划—控制接口与安全执行。三者分别设置离线、仿真或子系统实验。前三项研究完成后，另设第四章进行真实车辆与真实四足机器人的系统集成和全方位实车实验；第四章承担综合验证，不重复包装为新的研究内容。

论文主体暂按以下四章组织：第一章对应研究内容一及严格配对离线实验；第二章对应研究内容二及少样本/在线闭环仿真；第三章对应研究内容三及控制接口、动力学和子系统实验；第四章集成前三章，在真实车和真实狗上完成统一任务矩阵下的综合实车实验。

## 2. 总体目标

建立一套面向轮式与四足移动平台的跨具身分层决策方法。在场景和目标条件下，模型显式分离执行条件稳定的共享策略意图与平台相关修正；面对新平台或能力变化时，通过少样本数据和短时执行反馈估计当前能力状态并更新轻量适配器；再由平台控制器完成轨迹跟踪。前三项研究分别验证表征、适配和执行接口，第四章在真实车辆与真实四足机器人上综合评价目标成功、安全性、适配速度和迁移样本效率。

## 3. 总体因果与系统变量

定义：

$$
S=\text{场景},\qquad
G=\text{目标},\qquad
E_t=\text{时变执行能力},
$$

$$
Z_I=\text{共享策略意图},\qquad
Z_E=\text{具身修正},\qquad
\tau=\text{规划轨迹},
$$

$$
u_t=\text{控制命令},\qquad
x_{t+1}=\text{实际执行状态}.
$$

总体数据流为：

$$
(S,G)\rightarrow Z_I,
$$

$$
(S,G,E_t,\text{execution history})\rightarrow Z_E,
$$

$$
(Z_I,Z_E)\rightarrow \tau\rightarrow u_t\rightarrow x_{t+1}.
$$

执行反馈再用于更新 $E_t$ 的估计，形成在线闭环。课题不声称仅凭观察数据完全识别真实物理因果图，而是通过固定 $S,G$、受控改变 $E$ 的实验检验模型内部机制。

# 研究内容一：执行条件干预下的共享—具身 Score 解耦轨迹扩散

## 4.1 研究目标

研究内容一回答“跨具身决策中什么应当共享”。其核心不是对齐汽车和机器狗的完整轨迹，而是在统一轨迹空间中，将扩散去噪场分解为共享策略意图与具身修正：

$$
\hat\epsilon_e
=
\hat\epsilon_I
+
\beta(t)\hat\epsilon_E.
$$

$\hat\epsilon_I$ 只接收统一场景、目标和历史条件；$\hat\epsilon_E$ 接收平台能力、运动学与地形约束。最终轨迹由二者共同生成。

## 4.2 主要方法

1. 将各平台轨迹变换到统一机体坐标与弧长参数化空间，避免把速度差异误当共享意图；
2. 使用 nuPlan Diffusion-Planner 初始化共享去噪分支；
3. 为不同平台增加零初始化、小容量的具身残差 Adapter；
4. 在平台均衡数据上训练加性去噪目标；
5. 接入由同门构建并经本研究审计的严格 Car–Dog 反事实样本对，施加共享 score 一致性；
6. 使用目标平台轨迹作为监督执行定向 counterfactual swap；
7. 使用具身身份对抗、残差幅值和容量限制降低分支坍缩；
8. 通过分支置零、打乱、交换和 $do(E)$ sweep 验证中间分解真正参与轨迹生成。

完整公式、损失与实现边界见 [论文研究方案](./paper_scheme_score_decomposition.md)。

## 4.3 关键训练目标

$$
\mathcal L_1
=
\mathcal L_{diff}
+
\lambda_{inv}\mathcal L_{inv}
+
\lambda_{swap}\mathcal L_{swap}
+
\lambda_{sep}\mathcal L_{sep}.
$$

其中，$\mathcal L_{diff}$ 保证各平台轨迹生成，$\mathcal L_{inv}$ 保证共享分支在受控执行条件替换下稳定，$\mathcal L_{swap}$ 检验共享意图与目标具身修正能否重组，$\mathcal L_{sep}$ 降低两个分支相互吞并。

## 4.4 预期创新点

1. 提出面向跨具身轨迹扩散的共享—具身加性去噪场表征；
2. 利用严格配对数据构建目标平台有监督的定向反事实交换，避免任意样本拼接造成伪监督；
3. 将执行条件干预、分支辨识和扩散规划统一到可验证训练框架；
4. 建立从 zero-shot、少样本到目标闭环的跨平台评价协议。

## 4.5 与已有工作的区别

[NoMaD](https://general-navigation-models.github.io/nomad/index.html) 支持多机器人数据与多模态导航 diffusion，但不分解共享/具身 score；[CrossTracer](https://arxiv.org/abs/2608.06688) 已提出共享语义 trace 与具身残差，因此“共享路径＋具身修正”本身不能作为创新。本研究的差异必须落实在 score 场分解、受控 swap、分支可辨识和少样本迁移验证上。

# 研究内容二：基于执行反馈能力辨识的少样本—在线双阶段具身适配

## 5.1 研究定位

研究内容二回答“新平台或平台状态变化后怎样快速适配”。它不重新定义共享意图，也不与研究内容一竞争同一个创新点：研究内容一给出稳定决策基座，研究内容二只更新具身能力状态与轻量修正模块。

建议把“少样本初始化＋在线闭环适配”作为一个完整内容，而不是在两者之间二选一：少样本阶段解决冷启动，在线阶段解决部署后未知地形、负载、打滑和能力退化。

## 5.2 方法框架

### （1）执行历史与能力状态辨识

在时刻 $t$ 收集最近 $H$ 步执行历史：

$$
\mathcal H_t
=
\left\{
x_{t-H:t},
u_{t-H:t-1},
\tau_{t-H:t}^{plan},
x_{t-H:t}^{exec},
r_{t-H:t}
\right\},
$$

其中 $r_t$ 包含轨迹跟踪误差、速度响应、打滑/足端稳定代理、碰撞和地形反馈。能力辨识器输出时变具身状态及不确定性：

$$
(\hat z_t^E,\Sigma_t^E)
=
A_\psi(\mathcal H_t).
$$

$\hat z_t^E$ 不要求逐项等于真实质量、摩擦系数或电机参数；第一阶段可使用可解释参数与潜变量混合表示：

$$
\hat z_t^E
=
[\hat v_{max},\hat\omega_{max},\widehat{slip},
\widehat{slope\ capability},z_{latent}].
$$

最新的 Rapid Embodiment Adaptation 已证明从短时动作—状态历史在线识别关节限制和载荷变化能够改善四足闭环控制，因此“在线能力辨识”有直接依据；本课题需要进一步研究它如何调制轨迹 diffusion 的具身 score，而不是重复其低层 locomotion 设定。[Rapid Embodiment Adaptation（2026）](https://arxiv.org/abs/2608.01506)

### （2）少样本具身 Adapter 初始化

共享分支 $\theta_I$ 冻结或只开放最后少量层，目标平台只训练 LoRA/Adapter 参数 $\theta_E$：

$$
\theta_E^{(0)}
=
\operatorname{Adapt}
\left(
\theta_I,\mathcal D_{target}^{K}
\right),
$$

其中 $\mathcal D_{target}^{K}$ 为 1%、5%、10% 或固定 $K$ 条完整目标平台轨迹。与从头训练相比，该阶段只需学习具身修正。

[COMPASS](https://arxiv.org/abs/2502.16372) 使用冻结/继承的基础移动策略与 residual RL 训练平台 specialist，再进行蒸馏，直接支持“基础能力＋目标平台残差适配”；本研究不照搬完整 residual RL，而把残差限制在 trajectory score Adapter 中。

### （3）在线快—慢双时间尺度适配

快通道每个控制周期更新能力状态，不立即修改大模型参数：

$$
\hat z_t^E
\leftarrow
A_\psi(\mathcal H_t).
$$

慢通道累计高置信执行片段后，只更新具身 Adapter：

$$
\theta_E^{t+1}
=
\theta_E^t
-
\eta_t
\nabla_{\theta_E}
\mathcal L_{online}.
$$

在线损失为：

$$
\mathcal L_{online}
=
\lambda_{dyn}\mathcal L_{dyn}
+
\lambda_{track}\mathcal L_{track}
+
\lambda_{cons}\mathcal L_{intent}
+
\lambda_{safe}\mathcal L_{safe}.
$$

- $\mathcal L_{dyn}$：预测动作后的真实状态变化；
- $\mathcal L_{track}$：规划—执行轨迹误差；
- $\mathcal L_{intent}$：适配前后共享分支输出保持稳定；
- $\mathcal L_{safe}$：对碰撞、超坡度、打滑和失稳片段施加风险惩罚。

FSTTA 指出在线导航中频繁更新易漂移、更新过慢又无法跟随环境变化，并采用快—慢机制平衡可塑性与稳定性；其任务是 VLN，本课题借用更新时间尺度设计而不是语言模型部分。[FSTTA（ICML 2024）](https://proceedings.mlr.press/v235/gao24p.html)

### （4）不确定性门控与安全回退

只有满足以下条件才允许慢通道更新：

$$
\operatorname{tr}(\Sigma_t^E)<\delta_u,
\qquad
risk_t<\delta_r,
\qquad
quality_t>\delta_q.
$$

否则冻结在线更新，执行保守速度、局部安全停止或传统控制器回退。SALON 在越野导航中利用在线自监督经验快速更新 traversability cost/speed map，同时避开分布外地形，说明在线学习需要与风险意识共同设计。[SALON（2024）](https://arxiv.org/abs/2412.07826)

## 5.3 研究内容二的创新边界

已有工作分别做过 residual adaptation、在线地形学习、快慢测试时适配和短历史能力辨识，因此下列单点不能单独声称创新：

- 使用 LoRA 少样本微调；
- 从历史推断 embodiment latent；
- 在线更新 traversability；
- 基础策略加 residual policy。

建议将创新凝练为：

> 提出由执行误差驱动的能力状态辨识与不确定性门控双时间尺度适配方法，在冻结共享策略 score 的前提下，仅在线更新具身 score Adapter，使模型在保持高层决策意图稳定的同时快速响应平台能力与地形交互变化。

这一表述与研究内容一形成清晰分工，而且与只使用静态 robot ID 的 CrossTracer、只适配低层 locomotion 的 Rapid Embodiment Adaptation 均有区别。

## 5.4 评价方法

- 冷启动少样本性能：1%、5%、10%，5 个数据种子；
- 在线适配速度：达到稳定性能所需步数/秒数；
- 突变恢复：负载、摩擦、最大速度或关节能力突变后的恢复时间；
- 适配收益：适配前后 Success、SPL、Collision、Tracking Error；
- 稳定性：连续任务遗忘、错误更新率和参数漂移；
- 安全性：高不确定性下的回退触发准确率、危险动作率；
- 消融：无能力辨识、无快慢分离、无不确定性门控、全参数在线更新、只少样本不在线。

# 研究内容三：面向异构移动平台的分层轨迹跟踪与安全执行

## 6.1 研究定位

研究内容三不再提出新的共享表征，而是回答“规划轨迹怎样落到实际动作”。研究内容一和二输出统一机体级 SE(2) 轨迹及能力状态；研究内容三建立轮式、全向和四足平台的执行接口，并把跟踪误差和安全状态反馈给研究内容二。本内容以运动学/动力学仿真和控制子系统实验验证接口有效性，完整的双平台实车对照统一放在第四章。

## 6.2 统一规划—控制接口

规划器输出：

$$
\tau^{ref}
=
\{x_i,y_i,\psi_i,v_i\}_{i=1}^{N}.
$$

平台控制器为：

$$
u_t^e
=
\pi_{ctrl}^e
\left(
x_t,\tau^{ref},\hat z_t^E
\right).
$$

统一反馈包括：

$$
r_t
=
[e_{lat},e_{yaw},e_v,collision,slip,stability,risk].
$$

这样研究内容三的平台控制细节不会反向污染研究内容一的共享决策定义，但能先在仿真和控制子系统中为研究内容二提供可观测的执行证据，并为第四章实车接入固定数据接口。

## 6.3 平台实现

### 差速与全向轮式平台

- 差速平台使用非完整运动学 MPC/MPPI，约束 $v$、$\omega$、加速度和障碍距离；
- 全向平台保留横向速度 $v_y$，但设置轮速、侧向加速度和足迹碰撞约束；
- 高风险区域将地形 cost 与不确定性作为控制代价；
- 在运动学与动力学仿真中完成参数整定，同时实现 ROS2 控制接口，为第四章实车集成复用。

近年的 MPPI-IPDDP 将采样式 MPPI、无碰撞走廊和梯度优化结合，在差速移动机器人上生成平滑无碰撞轨迹，可作为轮式控制实现参考。[MPPI-IPDDP（TII 2025）](https://ieeexplore.ieee.org/document/10960717/)

### 四足平台

- 高层 SE(2) 轨迹转换为 1–3 个滚动 waypoint 或机体速度参考；
- 低层使用已训练的地形自适应 locomotion policy 跟踪；
- 若 waypoint 涉及台阶、窄通道或急转，允许低层选择不同步态/技能；
- 使用稳定性或 reach-avoid 价值触发 recovery policy；
- 足端、关节和力矩控制由低层策略处理，不再由 Diffusion-Planner 直接输出。

[Skill-Nav](https://arxiv.org/abs/2506.21853) 使用 waypoint 连接高层导航与多技能四足 locomotion，直接支持本课题的接口选择；[Agile But Safe](https://roboticsproceedings.org/rss20/p059.html) 使用 agile policy、recovery policy 和 reach-avoid value 形成安全切换，可作为四足安全执行模块参考。

## 6.4 本研究内容的验证层级

1. TartanGround 静态地图上的运动学闭环；
2. Gazebo/Isaac Sim 中的轮式与四足动力学闭环；
3. hardware-in-the-loop 或控制器在环的延迟、噪声和接口验证；
4. 可安排单平台、短路线的功能性小实验，用于排查接口与安全问题，但不替代第四章综合实车实验。

研究内容三的验收重点是轨迹—动作转换、跟踪稳定性和安全回退接口。真实车与真实狗在统一任务矩阵下的完整对照、统计检验和系统结论归入第四章。

## 6.5 评价指标

- Goal Success Rate、SPL；
- Collision、Terrain Failure、Stuck Rate；
- 横向/航向/速度跟踪误差；
- 控制平滑性、能耗代理和指令饱和率；
- 四足 recovery 触发次数与恢复成功率；
- 规划频率、控制频率、端到端时延；
- 仿真到实机或不同动力学配置下的性能下降。

# 7. 第四章（综合验证）：跨具身系统集成与全方位实车实验

## 7.1 章节定位

第四章不是第四项研究内容，也不额外制造新的算法创新点，而是将前三项研究的模型和接口部署到真实车辆与真实四足机器人上，回答“整套方法在真实执行条件下是否成立”。本章冻结前三章已经确定的方法、超参数和验收阈值，避免根据实车测试集反向调节方法后再报告结果。

## 7.2 实车系统与任务矩阵

- **平台**：一套真实轮式车辆和一套真实四足机器人，分别接入统一的 SE(2) 规划接口、平台控制器和安全急停；
- **任务**：在可对齐的起点、目标和场景语义下设置开放道路、窄通道、绕障、坡地及能力分叉任务；
- **迁移设置**：zero-shot、固定预算 few-shot、few-shot＋online adaptation；
- **对照方法**：原始 Diffusion-Planner、普通微调/LoRA、无 score 解耦、无在线适配、完整方法和平台专属上界；
- **重复规则**：每个平台—场景—方法组合进行多次独立试验，保留失败试验、人工接管和急停记录，不只展示成功案例。

能力分叉任务应允许平台产生不同的合理轨迹。例如面对可跨越台阶时，四足可以直接通过而轮式车辆可以绕行；评价共享意图时比较目标和通道层面的任务一致性，评价执行结果时分别使用平台可行性标准，不能把轨迹像素级一致当作成功条件。

## 7.3 综合评价指标

1. 任务层：Goal Success Rate、SPL、Goal Progress、完成时间；
2. 安全层：Collision、Terrain Failure、Stuck、人工接管和急停次数；
3. 迁移层：不同样本预算下的性能、适配时间、能力突变恢复时间；
4. 控制层：横向/航向/速度误差、指令饱和、四足 recovery 成功率；
5. 表征层：严格配对上的 shared-score distance、swap error、具身/任务 probe；
6. 系统层：规划与控制频率、端到端时延、连续任务稳定性。

统计上以“同一任务条件下的方法差值”为基本单位，报告均值、置信区间和逐任务成功/失败；成功率采用配对比例检验或 bootstrap 区间，连续指标采用配对 bootstrap。这样第四章既给出直观实车案例，也能形成支持前三章结论的统计证据。

## 7.4 安全与复现要求

- 先仿真、再控制器在环、最后低速实车，逐级通过后才开放下一层；
- 预设速度、坡度、台阶、姿态和通信超时阈值，并保留独立硬件急停；
- 每次试验记录模型版本、地图/目标、平台参数、随机种子、在线更新日志和 rosbag；
- 严格反事实训练集、少样本适配集和最终实车测试路线相互隔离。

# 8. 三项研究内容与第四章的关系

研究内容一与研究内容二的科学问题并列，但接口互补：

- 内容一提供“在执行条件变化下什么保持稳定”的共享决策表征；
- 内容二提供“平台当前能做什么、如何快速修正”的具身能力状态和 Adapter；
- 内容三将二者共同生成的轨迹转为动作，并产生新的执行反馈；
- 第四章不新增方法，而是在真实车辆与真实四足机器人上集成前三项成果并进行统一对照。

因此不是三个彼此无关的任务，也不是把同一个少样本实验重复写三次，而是：

$$
\text{一致性表征}
+
\text{具身在线适配}
+
\text{安全执行接口}
=
\text{跨具身迁移方法},
$$

并由：

$$
\text{前三项研究成果}
+
\text{真实 Car/Dog 综合实验}
=
\text{完整系统证据链}.
$$

总体框架图见 [research_framework.svg](./figures/research_framework.svg)。

# 9. 总体实验设计

## 9.1 数据和平台

- 源端：nuPlan 汽车轨迹与 Diffusion-Planner checkpoint；
- 目标端：TartanGround `diff`、`omni`、`anymal`；
- 主要环境：ModernCityDowntown、OldTownFall；
- 重点地形：高差、窄通道、台阶、陡坡和支撑稀疏区域；
- 严格配对数据：使用同门对不同 Car/Dog 数据集处理得到的严格反事实样本对；本研究负责数据接口、坐标/目标一致性、切分与泄漏审计。平台约束规划器生成的弱配对只进入单独消融，不支撑主结论；
- 实机平台：最终使用真实车辆和真实四足机器人完成第四章综合实验。

## 9.2 评价协议

1. zero-shot：不更新目标平台权重；
2. few-shot：按完整轨迹抽取 1%、5%、10%；
3. held-out trajectory：测试新轨迹；
4. held-out environment：测试新环境；
5. capability intervention：固定场景和目标，改变宽度、转弯、坡度或台阶能力；
6. online shift：执行中改变摩擦、负载、速度上限或局部执行能力；
7. closed loop：固定全局目标滚动规划，不读取日志未来路线；
8. real-world integrated：在真实 Car/Dog 的统一任务矩阵上比较前三项研究的单独收益与组合收益。

## 9.3 核心对比

- 原始 Diffusion-Planner zero-shot；
- 普通全参数微调；
- 普通 LoRA/Adapter；
- CrossTracer 类静态具身 residual；
- 研究内容一完整 score decomposition；
- 内容一＋只少样本适配；
- 内容一＋内容二在线适配；
- Oracle 能力参数/Oracle 平台规划器上界。

# 10. 预期成果

1. 一套跨汽车、轮式和四足平台的统一轨迹扩散接口；
2. 一个可检查、可交换、可干预的共享—具身 score 分解模型；
3. 一个少样本初始化与执行反馈驱动的安全在线适配器；
4. 一套从轨迹到异构平台动作的闭环执行系统；
5. zero-shot、few-shot、online adaptation、动力学闭环和真实 Car/Dog 综合实验；
6. 对“哪些决策可共享、何时必须分叉、在线适配是否保持共享意图”给出可证伪实验结论。

# 11. 已确认条件与实施风险

1. **严格反事实数据已确定**：数据由同门构建，本文以外部上游数据为前提；风险转化为数据协议和质量审计风险，必须验证坐标、目标、有效区间、配对机制和 split 隔离。
2. **实机平台已确定**：最终使用真实车辆和真实四足机器人；风险转化为 ROS2 接口、场地复现、安全审批、设备排期和有效重复次数，应在研究内容三阶段提前完成接口联调。
3. **统一轨迹定义**：建议共享分支对齐弧长参数化几何，速度由具身分支承担；拿到严格配对后用停走、转向和能力分叉样本验证该划分。
4. **研究内容二工作量**：完整在线更新应放在仿真安全验证之后，先做能力 latent 在线估计，再开放 Adapter 参数更新。
5. **因果表述边界**：全文使用“在严格配对假设与受控执行条件干预下的机制验证”，并披露配对构造中的残余域差异，避免将跨数据集相关性直接表述成真实物理因果效应。

# 12. 主要参考文献及采用说明

1. [Diffusion-Planner](https://arxiv.org/abs/2501.15564)：源端汽车轨迹扩散主干和 guidance 接口。
2. [NoMaD](https://general-navigation-models.github.io/nomad/index.html)：多机器人目标条件导航与多模态 diffusion policy；不直接支持 score 解耦。
3. [Motion Planning Diffusion](https://arxiv.org/abs/2308.01557)：轨迹分布先验、目标条件和规划约束融合。
4. [Reduce, Reuse, Recycle](https://proceedings.mlr.press/v202/du23a.html)：提醒 noisy score 不能由干净分布乘积机械推出，限定本文概率解释边界。
5. [Interventional Causal Representation Learning](https://proceedings.mlr.press/v202/ahuja23a.html)：受控干预有助于稳定潜变量辨识的理论背景。
6. [CrossTracer](https://arxiv.org/abs/2608.06688)：最新且直接的轮式/腿式共享 trace＋具身 residual 对照工作。
7. [COMPASS](https://arxiv.org/abs/2502.16372)：基础移动策略、具身 residual RL 与跨平台 specialist/generalist 训练。
8. [PEAC](https://proceedings.neurips.cc/paper_files/paper/2024/hash/62203a74e233e933b160711e791e1a02-Abstract-Conference.html)：通过在线交互学习 embodiment-aware、task-agnostic 知识；主要用于跨具身强化学习预训练。
9. [FSTTA](https://proceedings.mlr.press/v235/gao24p.html)：在线导航快—慢测试时适配与稳定性—可塑性权衡。
10. [SALON](https://arxiv.org/abs/2412.07826)：基于自身经验在线学习地形可通行代价和速度，并考虑分布外风险。
11. [Rapid Embodiment Adaptation](https://arxiv.org/abs/2608.01506)：从短时交互历史显式识别平台变化并调制四足控制。
12. [Skill-Nav](https://arxiv.org/abs/2506.21853)：以 waypoint 连接高层导航和四足多技能 locomotion。
13. [Agile But Safe](https://roboticsproceedings.org/rss20/p059.html)：敏捷策略、恢复策略和 reach-avoid 安全价值网络。
14. [MPPI-IPDDP](https://ieeexplore.ieee.org/document/10960717/)：差速移动机器人的平滑、无碰撞采样—优化控制参考。
