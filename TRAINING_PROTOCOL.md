# 当前执行：四组统一 Diffusion Planner 实验

批准日期：2026-10-08。RUN_ID=`20261008T132918Z_four_groups`。本节为本轮执行合同；后面的旧迁移矩阵只保留作协议背景，不自动执行。

| 组 | 完整trajectory训练/验证/测试 | 当前可用轨迹的预计数量 | 最终测试 |
|---|---|---|---|
| g1 | ANYmal 70/10/20 | 17/2/5 | 冻结5条ANYmal |
| g2 | Omni 70/10/20 | 4/1/1 | 冻结1条Omni |
| g3 | Diff 70/10/20 | 3/1/1 | 冻结1条Diff |
| g4 | Diff与Omni分别80/20 | Diff 4/1、Omni 5/1 | 全部24条ANYmal |

先按完整trajectory排序，以split seed=20260911确定性shuffle，再冻结membership，随后派生窗口。小平台按整数取整并保证train/val/test非空，不按表现调整split。训练seed=11；验证和最终推理seed=11/23/47；这三次推理不等于三次独立训练。g4与g1测试范围不同，额外从g4已经算出的逐任务结果提取g1同5条轨迹指标，不增加模型前向。新窗口数以实际manifest为准，不沿用原A的4280或原实验1的2048。

**共同模型与训练。** 四组都使用原实验1的普通Diffusion Planner backbone，独立从原始nuPlan EMA初始化，全主干可训练。源为`/zeron-vepfs/tjqc/cross-diffusion/checkpoints/model.pth`，来源记录为`/tj-share/cross_diffusion_workdir/runs/20261002T045353Z_proxy_cpu/source_provenance.json`。不加载旧ANYmal、A/B或smoke权重。无history latent、RMS、ID、adapter、classifier、配对或辅助loss；`--disable-history`为必要参数。使用原checkpoint normalizer，邻居/静态物体槽位为空，不增加感知标签处理。

**共同数据与几何。** 从anchor向后读到实际走满8米，按0.1、0.2、…、8.0米得到80个未来空间站点；current state单独提供，不重复作首个预测点。训练只保留完整8米监督，短尾记录为`incomplete_8m`，不补造轨迹；anchor从20开始、步长10。监督为XY/cos(yaw)/sin(yaw)，不表达固定8秒、速度或原地转向。采用已有证据支持的前相机观测参考系，不声称已知真实机体外参。姿态、goal、occupancy统一NWU；native相机局部0.2米体素按bounds/完整quaternion解码，转世界再转anchor局部，形成101×101@0.5米BEV，约50米范围。

**验证与选模。** AdamW，lr=1e-4，weight_decay=1e-4，batch=64，clip=5，GPU AMP；四维去噪SSE/有效点。每5个完整epoch结束做一次完整validation loss和导航验证（第5/10/15…epoch），日志同时记录epoch与update；工程smoke按第2/4更新验证，不能作为正式结果。选模仍按validation trajectory-macro SR→SPL→低CR→progress→更早update，g4等权平均Diff/Omni；只使用本组val。共同预算仍最少5000、最多10000次更新，连续10次导航检查SR未提升早停，仅在验证节点触发；最大更新数可能截断最后epoch，不能把半个epoch写成完整epoch。LR按每次验证SR调整，mode=max/factor=.5/patience=2/min_lr=5e-6。此次用户更改的是验证频率，没有把选模、scheduler和预算恢复成旧实验1。

**共同评价。** 使用各自`navigation_best.pt`；离线ADE/FDE和闭环SR/CR/SPL按trajectory宏平均，保存逐窗口/逐任务结果。每次规划使用当前朝向局部frame，预测转回任务坐标执行前1米、0.1米插值，最多16次replan，成功距离0.75米。圆形半径ANYmal=.35米、Diff/Omni=.5米为既定代理；本轮控制器以执行空间站点的预测yaw续接观测参考系；位移方向不再替代相机/姿态heading，允许侧移与朝向不同。旧实验1/Proxy控制器按位移方向更新heading，保留旧行为和结果。D024连通失败统一安全停车，记route_failure、保留分母、不计collision；invalid-map统一剔除并单独列数。首次训练前检查真实val坐标往返、完整8米、未来监督不能改变地图/路线输入，并保存地图/路径叠图及记录路径碰撞计数；发现坐标错误应修复后续接，不筛任务或放宽阈值提高分数。

**与最新原实验1的区别及选择。** 对照必须使用2026-09-29总结对应的`finetune_navigation_selection_earlystop_v1`，实际保留结果为`results/transfer/20261001T154600Z_retained_spl_seed11`，而不是更早loss选模实验。

| 项目 | 原实验1实际设置 | 本轮选择与理由 |
|---|---|---|
| 模型 | 普通pretrain_finetune，无新增history模块 | 保持同backbone与原始nuPlan独立起点 |
| 未来监督 | 先取之后80帧约8秒，再重采样8米容器；2048中仅692个窗口完整80点 | 读取实际完整8米，80未来站点；落实用户固定空间长度意图 |
| 首点 | 来自下一帧记录，在空间重采样起点附近 | 从0.1米预测，current token单独提供；是明确表示选择，不称旧首点错误 |
| 姿态坐标 | 原记录XY/yaw，NED约定 | 所有输入统一NWU；NED本身不是错误，关键是全流程一致 |
| occupancy | 直接以0.5米解释同源0.2米体素索引 | 按实际体素尺度、bounds和完整姿态显式转换，修正已核实的2.5倍尺度假设差异 |
| 网格 | 至少250格；native物理范围约50米，不能据旧错误尺度称125米 | 101格@0.5米，显式转换后仍约50米范围 |
| 闭环frame | 固定任务坐标；当前state包含yaw | 每次当前朝向局部规划，再转回任务frame；对齐训练局部输入，未证实旧固定frame必然错误 |
| split/任务 | ANYmal16/3/5；val35×3seed，test63任务、默认seed10000 | 各平台完整trajectory冻结；g1 17/2/5，其他如上；统一3个最终推理seed，任务数按新manifest记录 |
| 验证 | 每5epoch | 保留每5epoch |
| 选模/预算/LR | SPL优先；最少30/最多100epoch；patience8；每10epoch按val loss(min)调LR | 统一SR优先、5000–10000update、patience10和SR(max)调LR；避免不同数据量的epoch等于不同训练更新预算 |

原实验1实际100%结果：停止85epoch/2720update，best45epoch/1440update；val macro SR=.7051767677、CR=.2114898990、SPL=.6506459323；test macro SR=.6797827173、CR=.1587662338、SPL=.6232292192。g1完成后报告对应新指标、差值、loss趋势和实际窗口/预算，同时说明split、几何与选模都变了，不能仅用差值归因模型好坏。当前几何代理与记录路径安全目标冲突仍作为限制报告；不迎合“必然接近旧分数”的预期。

**服务器路径与入口。** 正式源码`/zeron-vepfs/tjqc/cross-diffusion`；Python`/root/miniconda3/envs/diffusion-planner/bin/python`；所有产物仅`/tj-share/cross_diffusion_workdir`。本轮说明保存为`runs/20261008T132918Z_four_groups/experiment_description.md`；归档说明`archive/experiment1/README.md`、旧代码`archive/experiment1/code/`、结果`archive/experiment1/results/`、用户阶段总结`archive/experiment1/stage_summary_20260929.md`。归档从本机已保留的旧入口和依赖提取，不部署到正式源码；旧结果从服务器原保留目录复制，不删除原件、不作为当前训练起点。用户明确授权此例外，不扩大同步其他历史。

同RUN_ID启动/恢复（runner自动跳过已经完成的组；只从本组last完整恢复）：

```bash
cd /zeron-vepfs/tjqc/cross-diffusion
PROFILE=four_groups RUN_ID=20261008T132918Z_four_groups bash tartan/research_score/scripts/run_transfer_training.sh
```

g4物化可复用本轮g1/g2/g3相同sample_id的确定性物理特征，先核对冻结80点target/mask完全相同，再按g4冻结manifest更新split/study_group；cache记录reused_feature_sources，不复用latent、统计量或模型checkpoint，避免重复路线计算。

共享数据分别为`data/20261008T132918Z_four_groups_g1`至`_g4`，cache同DATA_ID；主run保存actual config/commands/logs、preflight、每组smoke/train/eval、last/best、summary/report/main_table。2张L20按g1+g2、g3+g4分两批，每GPU一个任务。原实验和已完成消融不重跑。

<!-- HISTORICAL_TRANSFER_PROTOCOL -->

# Cross-Diffusion 训练与测试协议

适用范围：本文仅原实验一transfer。当前Proxy A/B使用根`proxy-training-protocol.md`及唯一实施计划，不沿用本文件ANYmal训练划分/四方法预算。

本文档定义当前正式训练与测试的目标协议。训练协议为
`score-decomp-transfer-v1.2.3`，当前闭环测试任务为 v1.2.4；尚未与目标协议一致的
脚本列在第 8 节。项目范围和完成状态分别见 [README.md](README.md) 与
[PROJECT_STATUS.md](PROJECT_STATUS.md)。

## 1. 实验目标与边界

实验研究将 nuPlan 预训练的 Diffusion Planner 迁移到 TartanGround ANYmal
局部导航。模型输出是机体级局部轨迹，不是关节或电机控制信号。

当前正式实验比较四种迁移方法，验证场景限定为同一地图上的未见 episode。
当前结果不是多随机种子结果，也不支持未见地图、真实机器人或因果解耦结论。

## 2. 数据与轨迹表示

### 2.1 数据域

- 源域：nuPlan / Diffusion Planner car 数据与预训练权重
  `checkpoints/model.pth`。
- 目标域：TartanGround ANYmal 轨迹和 occupancy map。
- 正式训练读取预生成缓存：
  `target_train_features.pt`、`target_val_features.pt` 和
  `source_car_features_4096.pt`。

源域缓存默认从 manifest 确定性抽取 4,096 个样本，默认抽样种子为
`20260914`。当前仓库没有记录正式缓存的文件哈希，也没有一条完整命令可证明
现有缓存由这些默认参数生成，因此复现实验前必须单独核验缓存来源。

### 2.2 目标轨迹

- 历史：20 帧，10 Hz。
- 未来表示：固定 8 m 弧长，重采样为 80 点。
- 单点表示：`[x, y, cos(yaw), sin(yaw)]`。
- 对不足 8 m 的有效部分使用 validity mask。
- 候选 anchor 默认每 10 帧提取一次。

源域原始目标是固定时间范围，目标域是固定空间弧长。两者经过表示桥接后进入
同一训练接口，但语义并不完全等价；涉及联合训练或共享机制的结论必须保留此限制。

相关实现：

- [`configs/transfer_data.yaml`](tartan/research_score/configs/transfer_data.yaml)
- [`data/core.py`](tartan/research_score/data/core.py)
- [`scripts/build_transfer_manifests.py`](tartan/research_score/scripts/build_transfer_manifests.py)
- [`scripts/materialize_target_features.py`](tartan/research_score/scripts/materialize_target_features.py)
- [`scripts/materialize_source_features.py`](tartan/research_score/scripts/materialize_source_features.py)

## 3. 数据划分与预算

目标域按完整 episode 划分，目标比例为 70/15/15，划分种子为
`20260911`。当前文档记录的数据规模如下：

| 划分 | Episode 数 | 用途 |
|---|---:|---|
| Train | 16 | 2,048 个 dense windows |
| Validation | 3 | loss 监控与 35 个冻结导航任务；按 Episode-macro SR 选模 |
| Test | 5 | 63 个冻结、非重叠 8 m 任务 |

嵌套预算定义为：

| 预算 | 1% | 10% | 20% | 50% | 100% |
|---:|---:|---:|---:|---:|---:|
| 训练窗口 | 20 | 205 | 410 | 1,024 | 2,048 |

当前正式 v3 runner 只运行 1%、10% 和 100%；20% 与 50% 虽被训练器接受，
但不在该 runner 的任务矩阵中。所有预算应覆盖全部 16 个训练 episode，验证集和
测试集保持 episode 隔离。

## 4. 比较方法

| 方法 | 初始化 | 目标数据 | 源数据 | 可训练部分 |
|---|---|---:|---:|---|
| `pretrain_finetune` | nuPlan checkpoint | 是 | 否 | 完整 backbone |
| `pretrain_adapter` | nuPlan checkpoint | 是 | 否 | 冻结 backbone，仅训练 embodiment residual wrapper |
| `joint_train` | nuPlan checkpoint | 是 | 是 | 完整 backbone |
| `emb_cond_diffusion` | nuPlan checkpoint | 是 | 是 | backbone 与 embodiment residual wrapper |

wrapper 使用平台 ID 和 10 维能力向量，目标域 ID 为 1，源域 ID 为 0；residual
末层零初始化。

原迁移正式矩阵仅包含上述四方法；已实施的Proxy B使用独立proxy_ab profile，见根Proxy协议。

相关实现：

- [`training/methods.py`](tartan/research_score/training/methods.py)
- [`model/score_decomposition.py`](tartan/research_score/model/score_decomposition.py)
- [`configs/transfer_methods.yaml`](tartan/research_score/configs/transfer_methods.yaml)

## 5. 正式训练设置

### 5.1 损失与更新

正式训练只使用归一化未来轨迹上的 masked denoising MSE：

```text
L = sum(mask * ||prediction - normalized_target||^2) / sum(mask)
```

正式训练不使用 `L_inv`、`L_swap` 或 `L_sep`。虽然输出字段名为 `score`，当前
训练目标是归一化轨迹，因此本文不把它解释为已经识别的因果 score。

对 `joint_train` 和 `emb_cond_diffusion`，每次 target optimizer update 后，再从
源域缓存有放回抽取一个同 batch 大小的数据批次，执行一次独立 source update。

### 5.2 超参数

| 项目 | 设置 |
|---|---|
| Batch size | 64 |
| 最大 target updates | 10,000 |
| 最小 target updates | 5,000 |
| 导航验证间隔 | 每 250 个 target updates |
| Early-stopping patience | 连续 10 次导航验证的 Episode-macro SR 无提升 |
| 主选模指标 | Episode-macro SR，越高越好 |
| Optimizer | AdamW |
| Adapter-only learning rate | `2e-4` |
| 其他方法 learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Gradient clipping | 5.0 |
| AMP | 启用 |
| 正式训练 seed | 11 |
| 固定验证 seed | `20260914` |

学习率调度器为 `ReduceLROnPlateau`，factor 为 0.5，patience 为 2，最小学习率为
`5e-6`。当前统一实现监视 validation Episode-macro SR（mode=max）。

### 5.3 验证与 checkpoint

使用冻结的 validation 导航任务评估 checkpoint，以 Episode-macro SR 最大者作为
最佳模型。并列时依次比较更高的 Episode-macro SPL、更低的 Episode-macro CR、
更高的 Episode-macro Goal Progress，最后选择更早的 checkpoint。

训练同时输出：

- `config.json`：冻结CLI、模型args、优化器与选模规则；

- `navigation_best.pt`：验证集 Episode-macro SR 最优的权重；
- `last.pt`：最终模型及优化器、scaler、scheduler 状态；
- `metrics.json`：训练设置、更新次数、验证导航指标、验证损失和训练历史。

导航验证与 early-stopping 的现有参考实现见
[`scripts/train_transfer.py`](tartan/research_score/scripts/train_transfer.py)，
2026-09-30 已收敛为四方法共同入口，以 Episode-macro SR 为首要指标；并列规则为 SPL、低 CR、Goal Progress、早 update。早停只按 SR 是否提升计数。GPU端到端验证尚未执行。

## 6. 正式测试协议

当前测试入口为
[`run_transfer_evaluation.sh`](tartan/research_score/scripts/run_transfer_evaluation.sh)。
它应加载正式训练目录中的 `navigation_best.pt`，对四种方法和 1%、10%、100% 三档预算进行
评估，共 12 个组合。

### 6.1 测试任务

- 测试任务来自完整隔离的 test episode。
- 每个任务沿记录轨迹向前取 8 m 终点作为固定目标。
- 相邻任务从不早于上一段终点的 anchor 开始，以避免局部任务重叠。
- 协议文档记录 63 个任务，但生成后的正式 manifest 未提交到仓库，因此实际运行
  时必须以 manifest 条目数为准。
- 路线特征构造不直接读取未来轨迹，但目标点来自记录轨迹的未来 8 m，因此仍包含
  goal-selection oracle，不应称为完全无 oracle 的自主导航测试。

### 6.2 闭环 rollout

| 项目 | 设置 |
|---|---|
| 起始状态 | 局部原点、yaw 为 0 |
| 候选路线 | 每次最多 6 条 |
| 单次执行轨迹 | 最多 1 m |
| 插值间隔 | 0.1 m |
| 最大 replans | 16 |
| 成功阈值 | 到目标距离不超过 0.75 m |
| Stuck 判定 | 连续 3 次移动不足 0.05 m |
| Collision | 发生即终止 |

每次 rollout 使用确定性推理种子
`10000 + episode_index * 100 + replan_index`，没有对多次 diffusion draw 取平均。

### 6.3 指标与汇总

每个 segment 记录：

- success rate（SR）；
- collision rate（CR）；
- SPL；
- travelled path 与 shortest path；
- goal progress；
- stuck rate；
- route-failure rate；
- replans、终止原因和平均推理时间。

SPL 定义为：成功时
`shortest_path / max(shortest_path, travelled_path)`，失败时为 0。Goal progress
定义为 `(初始目标距离 - 最终目标距离) / 初始目标距离`。

汇总同时报告 segment 微平均和 episode-macro 平均。无效地图 episode 不进入指标
分母，其余失败（包括 route failure）保留在分母中。

相关实现：

- [`scripts/build_navigation_tasks.py`](tartan/research_score/scripts/build_navigation_tasks.py)
- [`scripts/evaluate_navigation.py`](tartan/research_score/scripts/evaluate_navigation.py)
- [`evaluation/metrics_navigation.py`](tartan/research_score/evaluation/metrics_navigation.py)
- [`scripts/summarize_navigation.py`](tartan/research_score/scripts/summarize_navigation.py)

## 7. 执行入口

```bash
# 新训练自动产生 RUN_ID；打印在启动输出并保存在 runs/<RUN_ID>/run_id.txt
bash tartan/research_score/scripts/run_transfer_training.sh
# 替换为该次训练实际 RUN_ID；同一 run 显式恢复也使用此值
export RUN_ID=20261001T120000Z_transfer_seed11
bash tartan/research_score/scripts/run_transfer_evaluation.sh
```

默认代码 /zeron-vepfs/tjqc/cross-diffusion，Python /root/miniconda3/envs/diffusion-planner/bin/python，输出 /tj-share/cross_diffusion_workdir。PROJECT_ROOT/PYTHON_BIN/OUTPUT_ROOT 可按环境覆盖；TRAIN_CACHE 默认 cache/target_train_features_d029_extended.pt，VAL_NAVIGATION_MANIFEST 默认 data/navigation/val.jsonl，测试默认 data/navigation/test.jsonl。新产物写 runs/<UTC_RUN_ID>/train 与 eval，ASCII 名称；历史复现结果保留在 results/。不得覆盖旧实验或把搬迁时间解释为训练时间。

## 8. 当前实现状态与复现缺口（2026-10-01 更新）

- 正式入口已收敛为 train_transfer.py；四方法共用 update训练、冻结导航验证、SR选模与早停，不再使用按loss保存的best.pt。
- runner为4方法×1/10/100% = 12个run，完成数按实际任务数核对；加载navigation_best.pt，新输出目录runs/<UTC_RUN_ID>，避免误用旧结果。
- transfer_methods.yaml 的repeat_id为11；runner只执行seed11。数据构建只冻结完整episode split，预算由build_training_budgets按seed11生成，不能声称多seed结果。
- 启动训练使用指定 TRAIN_CACHE（seed11精确窗口budget cache）与 VAL_NAVIGATION_MANIFEST（冻结val导航任务）；禁止默用旧episode-budget cache。各预算精确样本数与episode覆盖仍须在服务器核对。
- 缓存生成命令/来源、最终split和63个测试任务manifest不在仓库内；服务器运行前需核验实际manifest，使用轻量来源/shape/count/sample ID检查，不默认生成hash清单。
- source_representation.py 保留既有源缓存短轨迹全valid语义，和target固定8m padding mask不等价。不能静默重建源cache并与旧结果混比；后续改此语义必须单独冻结修订。
- 代码单元测试通过不等于新协议实验已完成；代码同步与清理不启动正式GPU训练和测试。历史SPL-first/epoch结果保留为历史，不追写为SR-first结果。

## 9. 结果解释范围

当前协议可以支持：

- 同地图、未见 episode 上的跨域迁移比较；
- 四种正式迁移方法在固定数据预算下的性能比较；
- segment 和 episode-macro 粒度的闭环导航指标。

当前协议不能支持：

- 多 seed 稳定性；
- 未见地图泛化；
- 严格跨形态因果解耦或反事实预测；
- 真实机器人或物理可执行性；
- 完全无 oracle 的自主导航结论。

2026-10-08真实预检后的最小接口修正：four_groups闭环以预测轨迹在实际执行站点的yaw更新观测参考系，允许侧移朝向与位移方向不同；proxy_ab及旧实验1保留原位移heading行为。新g1四更新AMP smoke已通过（工程产物，不作性能结果）；smoke验证时钟明确为updates2/4，正式每5完整epoch。新修正由数学侧移/转向控制器测试覆盖，不需要机体外参猜测、不改loss或监督。
