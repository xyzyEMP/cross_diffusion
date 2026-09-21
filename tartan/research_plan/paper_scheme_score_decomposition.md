# 论文研究方案：执行条件干预下的共享—具身 Score 解耦扩散规划

> 本文档按“可直接作为论文方法方案”的标准整理。核心思路保持为：将跨具身轨迹扩散模型分解为共享策略意图分支与具身修正分支，并利用受控配对和反事实交换提高二者的可辨识性。

## 1. 研究问题与基本定义

给定场景 $S$、导航目标 $G$ 和执行平台 $e$，规划器生成未来轨迹：

$$
\tau_0 \sim p_e(\tau_0\mid S,G),
\qquad e\in\{\mathrm{car},\mathrm{diff},\mathrm{omni},\mathrm{dog}\}.
$$

其中，$\tau_0$ 为未来机体级轨迹。NoMaD 说明 diffusion policy 能表示多模态导航行为分布，并能利用多机器人数据训练统一导航模型；本研究借用的是“导航行为可以用条件轨迹分布表示”这一点，而不是直接照搬 NoMaD 的 goal masking 结构。[NoMaD（ICRA 2024）](https://general-navigation-models.github.io/nomad/index.html)

不同平台面对相同任务时，既有共同的目标推进和绕障倾向，也有由尺寸、运动学和地形能力引起的差异。因此研究目标不是强迫各平台输出相同轨迹，而是分离：

- 共享策略意图：在执行条件变化下应尽量稳定的决策成分；
- 具身修正：使共享意图适应具体平台可执行域的变化成分。

## 2. 统一轨迹空间与配对条件

### 2.1 为什么不能直接对齐原始轨迹

汽车和四足机器人速度、采样时域和可执行曲率不同。若直接在原始 8 秒轨迹上约束共享分支一致，模型会被迫把平台速度也学成“共享意图”，与研究目标矛盾。

因此先使用确定性变换 $T_e$ 将轨迹映射到统一规划空间：

$$
\bar\tau_0^e=T_e(\tau_0^e).
$$

第一版采用如下约定：

1. 坐标统一到当前机体坐标系，单位保持为米；
2. 轨迹几何部分按归一化弧长重采样，削弱速度差异；
3. 航向表示为 $(\cos\psi,\sin\psi)$，避免角度跳变；
4. 速度和时间参数作为具身条件或辅助输出，不进入严格共享一致性项；
5. 轨迹超出共同局部范围的部分使用有效掩码，不通过裁剪制造伪一致。

模型最终仍输出带时间的 SE(2) 轨迹；统一规划空间只用于共享意图监督、score 对齐和反事实交换。

### 2.2 严格反事实配对数据前提

本研究后续使用由同门完成构建的严格 Car–Dog 反事实配对数据集。该数据集由不同车/狗数据源经过场景、坐标、目标和任务语义对齐后形成；数据集构建本身不作为本文创新点，本文负责定义接入协议、完成质量审计，并将合格样本对用于一致性与交换训练。

进入主实验的配对样本必须满足：场景结构已映射到同一参考系，起始状态与导航目标可比，任务语义相同，主要干预变量为执行平台或能力参数：

$$
\mathcal D_{\mathrm{cf}}
=
\left\{
(id,S,G,e_a,e_b,\tau_0^{a},\tau_0^{b},T_a,T_b,m_{valid})
\right\}.
$$

其中，$T_a,T_b$ 将两个平台的数据映射到共同坐标和轨迹定义，$m_{valid}$ 表示该对是否通过质量审计。最低接入字段为：`pair_id`、场景/目标标识、两个平台类型、两条轨迹、共同坐标变换、有效区间、数据来源和审计状态。

接入时必须检查：坐标变换误差、起点与目标一致性、轨迹有效区间、时间/弧长采样规则、pair/scene 级数据切分、重复样本和未来信息泄漏。只有 $m_{valid}=1$ 的严格配对进入 $\mathcal L_{inv}$ 与 $\mathcal L_{swap}$。固定地图和目标后由平台约束规划器补充的样本只作为**弱配对消融**，不得与严格配对混合后共同支撑主结论；语义近邻样本只可用于无配对预训练或检索对照。

原始 nuPlan 汽车样本和 TartanGround 机器狗样本仍不是天然反事实对。只有经过上述处理并通过审计的数据，才能称为本文训练所用的严格反事实样本对，不能仅因场景“看起来相似”就用于第 5、6 节的严格一致性与交换监督。

## 3. Score Decomposition

### 3.1 概率层的建模动机

在干净轨迹层，可以将平台分布直观地理解为共享意图先验与具身能量修正的乘积：

$$
p_e(\bar\tau_0\mid S,G)
=
\frac{1}{Z_e(S,G)}
p_I(\bar\tau_0\mid S,G)
\phi_E(\bar\tau_0\mid S,G,e),
$$

其中 $\phi_E>0$，$Z_e$ 是归一化常数。由此在 $t=0$ 有：

$$
\nabla_{\bar\tau_0}\log p_e
=
\nabla_{\bar\tau_0}\log p_I
+
\nabla_{\bar\tau_0}\log\phi_E.
$$

但该等式不能直接推广成“两个独立扩散模型在所有 $t>0$ 的 noisy score 简单相加”。已有研究指出，干净分布乘积经过前向扩散后的边缘分布一般不等于各自 noisy 边缘分布的乘积。[Reduce, Reuse, Recycle（ICML 2023）](https://proceedings.mlr.press/v202/du23a.html)

因此，本文把概率乘积作为建模解释，而把下面的**联合加性去噪器**作为真正的训练定义，不宣称两个分支分别对应独立、完整且已归一化的扩散分布。

### 3.2 可训练的加性去噪定义

扩散前向过程为：

$$
\bar\tau_t^e
=
\sqrt{\bar\alpha_t}\,\bar\tau_0^e
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

模型在同一噪声时刻输出两个分量：

$$
\hat\epsilon_e
=
\hat\epsilon_I(\bar\tau_t,t,c_I)
+
\beta(t)\hat\epsilon_E(\bar\tau_t,t,c_E,e),
$$

其中：

- $c_I$ 仅包含统一后的场景、目标和历史意图信息；
- $c_E$ 包含地形能力、尺寸、运动学和执行反馈；
- $\beta(t)$ 第一版取 $1$，也可以作为有界的时间调制系数；
- 具身分支最后一层零初始化，使模型起点退化为预训练共享分支。

在 $\epsilon$-prediction 参数化下，对应 score 为：

$$
\hat s_e(\bar\tau_t,t)
=
-\frac{
\hat\epsilon_I+\beta(t)\hat\epsilon_E
}{\sqrt{1-\bar\alpha_t}}.
$$

Motion Planning Diffusion 证明 diffusion 可以学习多模态轨迹先验并结合目标或可微约束进行规划；本文进一步研究该轨迹去噪场如何在执行条件维度上分解。[Motion Planning Diffusion（IROS 2023）](https://arxiv.org/abs/2308.01557)

## 4. 基础 Diffusion Loss

对平台 $e$ 的样本，基础损失为：

$$
\mathcal L_{\mathrm{diff}}^e
=
\mathbb E_{\bar\tau_0^e,t,\epsilon}
\left[
\left\|
\epsilon-
\left(
\hat\epsilon_I^e+
\beta(t)\hat\epsilon_E^e
\right)
\right\|_2^2
\right].
$$

为避免 nuPlan 样本量远大于 TartanGround 后淹没目标平台，训练采用平台均衡采样：

$$
\mathcal L_{\mathrm{diff}}
=
\sum_{e\in\mathcal E}\pi_e
\mathcal L_{\mathrm{diff}}^e,
\qquad
\pi_e=\frac{1}{|\mathcal E|}.
$$

若某阶段只比较 Car 与 Dog，则 $\pi_{car}=\pi_{dog}=0.5$。这比直接把两个数据集损失相加更稳妥，因为后者会隐含依赖数据量和 batch 组成。

## 5. 受控执行条件下的共享一致性

对 $m_{valid}=1$ 的严格配对，先在两条规范化轨迹中等概率选一条作为查询来源，并加入同一噪声：

$$
\bar\tau_0^r
\sim
\tfrac12\delta(\bar\tau_0^a)
+
\tfrac12\delta(\bar\tau_0^b),
$$

$$
q_t
=
\sqrt{\bar\alpha_t}\,\bar\tau_0^r
+
\sqrt{1-\bar\alpha_t}\,\epsilon.
$$

然后让两个平台的共享分支在**同一个查询点**上比较：

$$
\mathcal L_{\mathrm{inv}}
=
\mathbb E_{q_t,t}
\left[
w_{pair}
\left\|
\hat\epsilon_I(q_t,t,c_I^a)
-
\hat\epsilon_I(q_t,t,c_I^b)
\right\|_2^2
\right].
$$

主实验中，通过审计的严格配对取 $w_{pair}=1$，未通过审计的样本不参与该损失。若另做弱配对消融，可单独设置 $0<w_{pair}<1$，但结果必须与严格配对主实验分表报告。该损失表达的是“在受控执行条件替换下，共享去噪场应稳定”，而不是声称已经从任意跨数据集相关性中识别出唯一真实因果变量。干预数据有助于辨识稳定潜变量的理论背景可参考 [Interventional Causal Representation Learning（ICML 2023）](https://proceedings.mlr.press/v202/ahuja23a.html)。

## 6. 定向 Counterfactual Swap

原始写法中只写了一个 $\epsilon$，但没有说明它对应哪条干净轨迹。正确的定向交换必须使用**目标平台轨迹的加噪样本和噪声监督**。

### 6.1 Car Intent → Dog Embodiment

先对 Dog 目标轨迹加噪：

$$
q_t^{dog}
=
\sqrt{\bar\alpha_t}\,\bar\tau_0^{dog}
+
\sqrt{1-\bar\alpha_t}\,\epsilon^{dog}.
$$

使用 Car 的共享上下文与 Dog 的具身上下文组合：

$$
\hat\epsilon^{car\rightarrow dog}
=
\hat\epsilon_I(q_t^{dog},t,c_I^{car})
+
\beta(t)
\hat\epsilon_E(q_t^{dog},t,c_E^{dog},dog).
$$

对应损失为：

$$
\mathcal L_{\mathrm{swap}}^{car\rightarrow dog}
=
\mathbb E
\left[
w_{pair}
\left\|
\epsilon^{dog}
-
\hat\epsilon^{car\rightarrow dog}
\right\|_2^2
\right].
$$

### 6.2 Dog Intent → Car Embodiment

对称地：

$$
q_t^{car}
=
\sqrt{\bar\alpha_t}\,\bar\tau_0^{car}
+
\sqrt{1-\bar\alpha_t}\,\epsilon^{car},
$$

$$
\hat\epsilon^{dog\rightarrow car}
=
\hat\epsilon_I(q_t^{car},t,c_I^{dog})
+
\beta(t)
\hat\epsilon_E(q_t^{car},t,c_E^{car},car),
$$

$$
\mathcal L_{\mathrm{swap}}^{dog\rightarrow car}
=
\mathbb E
\left[
w_{pair}
\left\|
\epsilon^{car}
-
\hat\epsilon^{dog\rightarrow car}
\right\|_2^2
\right].
$$

最终：

$$
\mathcal L_{\mathrm{swap}}
=
\mathcal L_{\mathrm{swap}}^{car\rightarrow dog}
+
\mathcal L_{\mathrm{swap}}^{dog\rightarrow car}.
$$

这个 loss 检验：共享上下文从一个平台取得时，目标平台具身分支能否仍重建目标平台轨迹分布。它只对通过审计的严格反事实配对成立，不能用于任意 Car–Dog 样本拼接；弱配对如需使用，只能作为明确标注的附加消融。

## 7. 分支可辨识性与解耦约束

仅使用 $\mathcal L_{diff}$、$\mathcal L_{inv}$ 和 $\mathcal L_{swap}$，仍可能出现：

$$
\hat\epsilon_I=\hat\epsilon_e,
\qquad
\hat\epsilon_E=0,
$$

或者具身分支吞掉全部预测。因此需要把 $\mathcal L_{sep}$ 写成可实现的约束，而不是一个未定义符号。

### 7.1 具身身份对抗约束

在共享分支中间表示 $z_I$ 后接具身分类器 $C_\psi$，通过梯度反转层 GRL 训练：

$$
\mathcal L_{adv}
=
\operatorname{CE}
\left(
C_\psi(\operatorname{GRL}(z_I)),e
\right).
$$

分类器学习识别具身，共享编码器收到反向梯度以删除可轻易识别的平台信息。该约束必须在平台均衡、场景尽量配对的 batch 中使用，否则分类器可能识别的是 nuPlan/Tartan 数据域，而不是具身本身。

### 7.2 具身残差容量与幅值约束

具身分支使用参数量明显小于共享分支的 Adapter/LoRA，并加入：

$$
\mathcal L_{res}
=
\mathbb E
\left[
\omega(t)
\left\|
\hat\epsilon_E
\right\|_2^2
\right].
$$

该项鼓励“能由共享分支解释的部分不重复进入具身分支”，但权重不能过大，否则真实的平台差异会被抹掉。

### 7.3 解耦损失

$$
\mathcal L_{sep}
=
\lambda_{adv}\mathcal L_{adv}
+
\lambda_{res}\mathcal L_{res}.
$$

第一版不强制 $\hat\epsilon_I$ 与 $\hat\epsilon_E$ 正交，因为共享意图与具身修正在轨迹坐标上可能方向一致；未经验证的正交假设反而会限制合理修正。

## 8. 最终训练目标

$$
\boxed{
\mathcal L
=
\mathcal L_{diff}
+
\lambda_{inv}\mathcal L_{inv}
+
\lambda_{swap}\mathcal L_{swap}
+
\lambda_{sep}\mathcal L_{sep}
}
$$

其中：

- $\mathcal L_{diff}$：保证各平台正常轨迹去噪与生成；
- $\mathcal L_{inv}$：约束受控执行条件替换下的共享去噪场稳定；
- $\mathcal L_{swap}$：验证共享上下文和目标具身修正可以重新组合；
- $\mathcal L_{sep}$：减少分支坍缩并提高分解可辨识性。

初始权重只作为验证集搜索起点：

```yaml
lambda_inv: 0.2
lambda_swap: 0.5
lambda_sep: 0.1
lambda_adv: 1.0
lambda_res: 0.01
```

权重必须在 validation 上选择，并报告对 $\lambda_{inv}$ 和 $\lambda_{swap}$ 的敏感性；测试集不得用于调参。

## 9. 模型和代码实现

### 9.1 模块结构

```text
场景/目标/历史
      │
      ├── 统一场景编码器 ── shared denoiser ε_I
      │
执行平台参数、地形能力、执行反馈
      │
      └── Embodiment Adapter ── residual denoiser ε_E
                                      │
                         ε̂ = ε̂_I + β(t) ε̂_E
                                      │
                         Diffusion reverse process
                                      │
                         平台级 SE(2) 轨迹
```

### 9.2 对 Diffusion-Planner 的最小改造

1. 原 DiT 主干作为共享分支初始化；
2. 在每个选定 DiT block 的 cross-attention 或 adaLN 后增加零初始化 LoRA/Adapter；
3. `embodiment_encoder` 编码平台尺寸、运动学、坡度/台阶能力和平台 ID；
4. 共享分支不接收平台 ID；
5. 总输出头显式返回 `eps_shared`、`eps_embodiment` 和 `eps_total`；
6. 训练和评测文件保存三个输出，支持 swap 和探针实验；
7. 原 checkpoint 在 Adapter 关闭时保持数值兼容。

建议目录：

```text
tartan/research_score/
├── configs/
├── data/{canonicalize,pairs,dataset}.py
├── model/{shared_denoiser,embodiment_adapter,score_planner}.py
├── training/{losses,train_stage1,train_paired}.py
├── evaluation/{open_loop,closed_loop,swap_eval,probes}.py
└── tests/
```

## 10. 训练流程

1. **共享初始化**：加载 nuPlan Diffusion-Planner，训练/校准统一轨迹表示；
2. **具身残差预适配**：冻结共享主干，使用平台均衡数据训练零初始化 Adapter；
3. **严格配对解耦训练**：接入同门提供的 Car–Dog 严格反事实样本对，完成坐标、目标、切分与泄漏审计后，只在 $m_{valid}=1$ 的数据上启用 $\mathcal L_{inv}$ 和 $\mathcal L_{swap}$；
4. **低学习率联合训练**：仅解冻共享主干后部，防止目标小数据破坏汽车先验；
5. **少样本评价**：按完整轨迹抽取 1%、5%、10% 目标平台数据，至少 5 个随机种子；
6. **目标闭环评价**：固定目标滚动规划，不读取日志未来路线。

Transfer Guided Diffusion Process 从理论上研究了预训练源扩散模型加目标域修正用于有限目标数据迁移的形式，可作为“预训练共享分支＋目标具身 correction”的方法依据，但其验证领域不是机器人规划。[Transfer Learning for Diffusion Models（2024）](https://arxiv.org/abs/2405.16876)

## 11. 评价指标

### 11.1 规划性能

- ADE、FDE；
- Goal Success Rate、SPL、Goal Progress；
- Number/Rate of Collisions；
- slope、step、support-availability、clearance violation；
- 速度、角速度、曲率和加速度违规率；
- 推理时间与轨迹采样有效率。

仅报告 Success Rate 和 Number of Collisions 不足以说明迁移质量：安全停止可能降低碰撞却不完成目标，因此至少同时报告 Success、SPL、Collision、Stuck 和 Terrain Violation。

### 11.2 解耦是否真实成立

- Shared-score distance：配对样本上的 $\mathcal L_{inv}$；
- Swap denoising error：两个定向交换损失；
- Embodiment probe accuracy：从 $z_I$ 预测平台，目标为接近均衡随机水平；
- Task probe accuracy：从 $z_I$ 预测目标方向/路线，必须显著高于随机；
- Residual norm ratio：$\|\hat\epsilon_E\|/(\|\hat\epsilon_I\|+\epsilon)$；
- Branch ablation：置零、打乱或交换任一分支后性能变化；
- `do(E)` sweep：固定场景、目标和 noisy query，只改变能力参数，检查共享输出稳定且总输出在能力边界处变化。

### 11.3 必要对比

1. 原始汽车 checkpoint zero-shot；
2. 全参数目标平台微调；
3. 普通 LoRA/Adapter，不做 score 分解；
4. 加性双分支，仅 $\mathcal L_{diff}$；
5. 加 $\mathcal L_{inv}$；
6. 加 $\mathcal L_{swap}$；
7. 完整模型；
8. Oracle platform-specific planner 上界。

## 12. 直接相关工作与本研究边界

- [NoMaD](https://general-navigation-models.github.io/nomad/index.html)：支持多机器人数据、目标条件和多模态 diffusion navigation；不提供共享/具身 score 解耦。
- [Diffusion-Planner](https://arxiv.org/abs/2501.15564)：提供汽车轨迹扩散主干与 guidance 接口，是源模型基础。
- [Motion Planning Diffusion](https://arxiv.org/abs/2308.01557)：支持将 diffusion 作为轨迹先验并结合规划约束；不研究跨具身一致性。
- [Reduce, Reuse, Recycle](https://proceedings.mlr.press/v202/du23a.html)：说明 diffusion score 的朴素乘积组合在 $t>0$ 存在理论偏差，因此本文不把加性网络误写成两个独立扩散密度的精确乘积。
- [CrossTracer](https://arxiv.org/abs/2608.06688)：使用共享语义 trace 与具身残差修正，在轮式/腿式导航上与本研究非常接近；它是必须比较的最新并行工作。本文需要突出 score 场解耦、受控 swap 和少样本迁移，而不能只声称“共享路线＋具身残差”。
- [Cross-robot behavior adaptation through intention alignment](https://pubmed.ncbi.nlm.nih.gov/41849566/)：支持跨机器人应对齐高层意图而非直接复制动作；它使用语言锚和候选行为检索，本文使用轨迹 score 与受控执行条件配对。
- [Interventional Causal Representation Learning](https://proceedings.mlr.press/v202/ahuja23a.html)：提供利用干预数据辨识潜变量的理论背景；本文的 `do(E)` 是规划系统内的受控能力替换，不等同于从观察数据完全识别真实物理因果图。

## 13. 已确认前提与实施检查点

1. **严格配对来源已确认**：后续将接入由同门对不同 Car/Dog 数据集处理得到的严格反事实样本对。本研究不把配对数据构建作为创新贡献，但必须保留版本、来源、变换、有效掩码和审计记录，并按 `pair_id/scene_id` 整体切分，防止同一对或同一场景跨训练、验证和测试集。
2. **实机条件已确认**：最终具备真实车辆和真实四足机器人。本文方法先完成离线、仿真和分模块验证，随后纳入开题第四章的跨平台综合实车实验。
3. **统一轨迹定义仍需在数据落地时固化**：当前采用弧长参数化几何作为共享空间，速度与时间尺度归入具身分支；拿到严格配对数据后，需要用配对轨迹统计验证该定义不会删除任务相关的停走与进度信息。
4. **因果术语的使用条件**：严格配对使受控执行条件比较具备数据基础，但仍需报告配对构造机制和可能残余差异；主张限定为“在既定配对假设下的执行条件干预与机制验证”。
