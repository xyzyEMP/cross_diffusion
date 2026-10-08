# 项目架构与文件清单

更新：2026-10-08。此清单覆盖全部正式代码、测试、配置和维护文档。每个文件对应当前任务；未登记文件视为可疑。实验一以 TRAINING_PROTOCOL.md 为准，独立 Proxy A/B 以 proxy-training-protocol.md 为准，执行规则见 AGENTS.md。

g4物化可复用本轮g1/g2/g3相同sample_id的确定性物理特征，先核对冻结80点target/mask完全相同，再按g4冻结manifest更新split/study_group；cache记录reused_feature_sources，不复用latent、统计量或模型checkpoint，避免重复路线计算。

当前四组研究复用现有build_transfer_manifests、materialize_target_features、build_navigation_tasks、audit_navigation_routes、train_transfer、run_transfer_training和evaluate_navigation、summarize_navigation；新增four_groups配置，不新建源码文件/入口。core负责完整trajectory划分与0.1…8m未来站点；cache携带显式experiment_profile隔离新表示与旧Proxy；trainer仅该配置允许ANYmal域内训练，正式每5完整epoch验证，旧proxy_ab仍250updates且ANYmal禁训。run脚本协调同一UTC RUN_ID下真实数据→preflight→GPU smoke→两GPU分批四组→离线/闭环→旧实验1比较。正式说明复用TRAINING_PROTOCOL，run内说明是产物。

用户明确授权原实验1必要旧代码/结果放输出盘archive/experiment1，属于历史参考，不属于正式源码清单/执行入口；仅此归档例外，其他本机项目外历史不部署。目录/精确差异和当前阶段见PROJECT_STATUS及TRAINING_PROTOCOL。

## 目录与职责

```text
cross_diffusion/
├── AGENTS.md / ARCHITECTURE.md / README.md / PROJECT_STATUS.md
├── TRAINING_PROTOCOL.md / proxy-training-protocol.md
├── checkpoints/               源模型args；server保留预训练model.pth
├── normalization.json         源checkpoint normalizer
├── diffusion_planner/
│   ├── model/                 checkpoint兼容Encoder/DiT/Decoder/采样与guidance接口；Proxy共同history在既有encoder内显式启用
│   └── utils/                 模型配置、normalizer及JSON读取
├── tartan/
│   ├── data/                  pose/SE2与lane/route特征公共实现
│   ├── research_score/
│   │   ├── data/              fixed-arc、源桥、窗口预算、occupancy路线和schema
│   │   ├── model/             embodiment条件、零初始化residual与逐去噪wrapper
│   │   ├── training/          Dataset、四方法注册表和loss原语
│   │   ├── evaluation/        D024、碰撞、最短路与导航指标
│   │   ├── scripts/           构建、缓存、训练、评价、汇总与源审计单入口
│   │   ├── configs/           数据构建、方法和输入路径配置
│   │   ├── preflight/         当前transfer运行前检查
│   │   └── tests/             当前接口/协议回归
│   └── tests/                 数据发现与坐标转换回归
├── pair/                      Diff–Omni真实候选挖掘及可视化
├── scripts/check_architecture.py
└── experiment_execution_plan/ 当前唯一计划、合同、Preflight、交接和模板
```

数据流：冻结manifest → data表示/预算 → Dataset/cache → train_transfer → backbone/residual wrapper → evaluate_navigation → metrics/summary。数据层不依赖训练入口；评价不依赖trainer。训练器单向复用evaluate_navigation的导航rollout。新增功能先归入既有职责层，避免平行入口。

Proxy调用（真实CPU及正式GPU全链通过）：Dataset/cache物理history→既有encoder.py的ProxyHistoryEncoder(138→64→16)及零投影(22→192)→A主干/B wrapper；B embodiment.py消费22维context+ID，classifier仍约束shared输出。学习latent不得写进共享cache，严格源加载后才启用新增模块。完整shape/初始化/恢复见唯一计划3.3；不增加源码文件。

追加原因诊断复用现有train_transfer和评价模块；trainer支持默认值1的pair_denoising_weight，仅base_only对照设0；不新增源码文件或平行入口。每项配置/复现脚本仅是输出盘run产物，ANYmal模型前向禁止；实际节点见PROJECT_STATUS。

## 当前任务与入口

- 迁移数据：build_transfer_manifests构建8m/80点与完整episode split；materialize_target_features/source_features构建cache；build_training_budgets写精确嵌套窗口membership。训练不得直接使用未写窗口membership的cache。
- 迁移训练/测试：run_transfer_training.sh / run_transfer_evaluation.sh；唯一train_transfer.py / evaluate_navigation.py；四方法×1/10/100%×seed11共12任务，20/50%是可选预算。
- 选模：validation episode-macro SR → SPL → 低CR → progress → 早update。每250 target updates验证，最少5000/最多10000，SR无提升patience10。测试使用navigation_best.pt；loss只诊断。
- Proxy A/B：Omni+Diff完整trajectory分别80/20，ANYmal全test；B只允许同split真实Diff–Omni pair。三平台manifest冻结；8m重建门禁、B训练/classifier接口已实施，已批准观测参考系门禁及真实CPU全链通过，原正式GPU阶段已完成，追加validation消融及原因报告已完成。
- 2026-10-02 Proxy历史条件已批准并实施：A/B共同使用16维历史latent与3项运动RMS摘要/mask。归属现有Dataset/cache、模型encoder/embodiment/wrapper、训练/评价入口；准确注入与pair/swap接口已在唯一计划3.3收口，已按合同落实。旧ability[10]不代表已批准物理能力表；不新增平行入口。
- pair/重建legacy10m/64点候选并补充native8m冻结窗口，218/20同split配对通过共同frame门禁后才训练B。
- source_representation保持已冻结源cache的短轨迹80有效点语义；target使用固定弧长padding mask。不能静默统一后与现有结果比较。
- D024：astar_disconnected统一安全停车，route_failure保留分母、不计碰撞。固定goal来自记录路径、同地图/单seed/简化运动学仍限制结论。

## 输出与部署

```text
/tj-share/cross_diffusion_workdir/
├── data/             冻结源基准、split/index、窗口预算、导航val/test
├── cache/            共享source/target特征
├── results/transfer/ 保留的唯一可复现结果及checkpoint
├── pairs/            真实候选；未通过配对门禁不可训练B
├── environment/      环境信息
└── runs/<UTC_RUN_ID>/ 新运行train/eval、配置、命令、日志、指标与checkpoint
```

新运行命名 YYYYMMDDTHHMMSSZ_label，仅使用ASCII字母、数字、点、下划线和连字符。训练runner自动生成时间戳；评价显式指定同一RUN_ID。重用RUN_ID只跳过已完成子任务，不完整子任务明确失败；Proxy已实现完整last恢复；CPU完整恢复已验证，GPU AMP/恢复已通过P5b。直接CLI的--output必须由调用者放入时间戳run目录。输出根OUTPUTS.md记录实际产物与搬迁映射；冻结输入和指标内容不改写。

服务器代码 /zeron-vepfs/tjqc/cross-diffusion 与本机正式源码一致。Git/工具私有目录和明确登记为“本机文档交付”的开题稿不部署；checkpoints/model.pth是服务器保留的源权重，不纳入源码清单。除用户批准的输出盘archive/experiment1参考归档外，其他归档只存本机独立目录 /home/yzy/文档/ChatGPT/cross_diffusion_local_history/2026-10-01_server_cleanup，不部署服务器。

## 维护规则

先读AGENTS、两份协议、本清单和唯一计划。优先修改原模块、原文档、同一个runner；不新建Git分支、平行版本或计划副本。新增文件必须说明必要性并同步登记职责和任务。移动/删除同步imports、启动器、测试和文档。原始数据、源checkpoint与唯一结果受保护；清理按精确路径、大小和保留来源执行。

运行 python scripts/check_architecture.py 检查清单；python -m pytest -q 检查当前协议回归。代码测试通过不代表正式GPU矩阵或Proxy A/B已完成。

## 逐文件清单

| 文件 | 职责 | 对应任务 |
|---|---|---|
| `.gitignore` | 本地治理文档/数据/checkpoint及缓存的Git排除边界 | 项目维护 |
| `AGENTS.md` | 项目执行约束与架构/协议维护指令，禁止另建平行版本 | 项目维护 |
| `ARCHITECTURE.md` | 当前架构、职责边界、逐文件清单及维护门禁 | 项目维护 |
| `PROJECT_STATUS.md` | 当前实现、部署、验证结果与可续接状态 | 项目维护 |
| `README.md` | 仓库总入口、安装与两类实验范围 | 项目维护 |
| `TRAINING_PROTOCOL.md` | 实验一冻结目标协议与当前实现/服务器复现缺口 | 项目维护 |
| `thesis_proposal.md` | 当前教师交流用硕士开题阶段稿；依据现有协议、状态与历史结果撰写，非执行入口，不部署服务器，不改变实验协议 | 本机文档交付 |
| `baseline_alignment.md` | 当前PLUTO/NoMaD三组70/10/20域内baseline共同设定；说明数据处理、训练与指标口径，输入表示由各模型负责；独立于Proxy80/20合同，不作为已有运行完成证据，不部署服务器 | 本机文档交付 |
| `checkpoints/args.json` | 原checkpoint输入shape、模型参数和normalizer配置 | 预训练模型加载/迁移 |
| `diffusion_planner/__init__.py` | 包导入边界 | 预训练模型加载/迁移 |
| `diffusion_planner/model/__init__.py` | 包导入边界 | 预训练模型加载/迁移 |
| `diffusion_planner/model/diffusion_planner.py` | checkpoint 兼容的 Encoder/Decoder 主模型；Proxy strict源加载后显式启用history | 预训练模型加载/迁移 |
| `diffusion_planner/model/diffusion_utils/__init__.py` | 包导入边界 | 预训练模型加载/迁移 |
| `diffusion_planner/model/diffusion_utils/dpm_solver_pytorch.py` | 上游 DPM-Solver ODE 数值采样器及 noise schedule | 预训练模型加载/迁移 |
| `diffusion_planner/model/diffusion_utils/sampling.py` | DPM-Solver 的 x_start/noise 模型包装与采样入口 | 预训练模型加载/迁移 |
| `diffusion_planner/model/diffusion_utils/sde.py` | VP/subVP 随机微分方程与边缘噪声分布 | 预训练模型加载/迁移 |
| `diffusion_planner/model/guidance/collision.py` | nuPlan 汽车矩形几何碰撞 guidance | 预训练模型加载/迁移 |
| `diffusion_planner/model/guidance/documentation_guidance.md` | 上游guidance接口与示例reward说明 | 预训练模型加载/迁移 |
| `diffusion_planner/model/guidance/guidance_wrapper.py` | 按 reward 梯度改正扩散 score 的 guidance wrapper | 预训练模型加载/迁移 |
| `diffusion_planner/model/module/__init__.py` | 包导入边界 | 预训练模型加载/迁移 |
| `diffusion_planner/model/module/decoder.py` | 路线编码、DiT 去噪与每步 residual correction 的采样路径 | 预训练模型加载/迁移 |
| `diffusion_planner/model/module/dit.py` | 扩散时间 embedding、条件 DiT block 与最终输出层 | 预训练模型加载/迁移 |
| `diffusion_planner/model/module/encoder.py` | 邻居、静态物体、车道与融合场景编码器 | 预训练模型加载/迁移 |
| `diffusion_planner/model/module/mixer.py` | Encoder/RouteEncoder 使用的 token/channel MLP Mixer block | 预训练模型加载/迁移 |
| `diffusion_planner/utils/__init__.py` | 包导入边界 | 预训练模型加载/迁移 |
| `diffusion_planner/utils/config.py` | 读取 checkpoint args 并创建冻结 normalizer | 预训练模型加载/迁移 |
| `diffusion_planner/utils/normalizer.py` | 观测和四维轨迹归一化及逆归一化 | 预训练模型加载/迁移 |
| `diffusion_planner/utils/train_utils.py` | checkpoint normalizer JSON读取，兼容mmengine存储 | 预训练模型加载/迁移 |
| `experiment_execution_plan/00_overall_progress.md` | 唯一当前执行计划与可续接状态 | 项目维护 |
| `experiment_execution_plan/01_preflight_framework.md` | 当前环境、冻结输入、协议与输出的运行前检查要求 | 项目维护 |
| `experiment_execution_plan/01_shared_execution_contract.md` | 当前执行合同：数据隔离、复现、存储与维护约束 | 项目维护 |
| `experiment_execution_plan/SERVER_HANDOFF.md` | 当前服务器路径、部署状态与可续接信息 | 项目维护 |
| `experiment_execution_plan/agent_handoff_template.md` | 记录完成内容、状态、产物、检查与下一步的交接模板 | 项目维护 |
| `experiment_execution_plan/decisions.md` | D024、当前方法矩阵、配对范围和维护决策 | 项目维护 |
| `experiment_execution_plan/project_AGENTS_template.md` | 可复用的先读计划/架构、更新原文件、不新建分支提示词 | 项目维护 |
| `experiment_execution_plan/protocol_changelog.md` | 当前协议修订与部署变更记录 | 项目维护 |
| `experiment_execution_plan/review_packet_template.md` | 简洁问题、证据和验证结论模板 | 项目维护 |
| `experiment_execution_plan/risk_register.md` | 当前真实实现/数据/科学结论缺口 | 项目维护 |
| `local.env` | 本机/服务器资源路径私有配置；保留用户文件，不发布 | 项目维护 |
| `normalization.json` | 源预训练观测/轨迹normalizer参数，保持checkpoint兼容 | 预训练模型加载/迁移 |
| `pair/README.md` | 真实pair挖掘接口与产物；新B还需split门禁和表示物化 | Proxy真实Diff–Omni候选 |
| `pair/config.yaml` | 真实候选挖掘长度/stride/几何阈值；10m64点候选不等同训练8m80点 | Proxy真实Diff–Omni候选 |
| `pair/matching.py` | 同地图跨平台空间邻近、entry/exit 和路径距离的真实 pair 候选挖掘 | Proxy真实Diff–Omni候选 |
| `pair/run_pair_mining.py` | Diff–Omni真实候选挖掘、时间戳输出、配置和来源记录 | Proxy真实Diff–Omni候选 |
| `pair/test_matching.py` | 回归：test_compare_identical_paths, test_grid_search_across_cell_boundary, test_visualization | 对应模块验证 |
| `pair/test_trajectory.py` | 回归：test_load_trajectory, test_extract_local_segments_uses_half_length_stride, test_extract_local_segments_interpolates_sparse_frames | 对应模块验证 |
| `pair/trajectory.py` | 读取真实 Tartan pose、metadata 时间并按弧长提取完整局部片段 | Proxy真实Diff–Omni候选 |
| `pair/visualization.py` | 绘制候选两侧轨迹及 entry/exit 的人工核验图 | Proxy真实Diff–Omni候选 |
| `proxy-training-protocol.md` | Omni+Diff训练、ANYmal仅test的proposed A/B协议 | 项目维护 |
| `pytest.ini` | 当前测试收集范围 | 项目维护 |
| `requirements_torch.txt` | 原始模型Torch与配套依赖版本 | 项目维护 |
| `scripts/check_architecture.py` | 检查所有维护文件均列入架构清单；无hash/额外依赖 | 项目维护 |
| `setup.py` | Python 项目安装及 diffusion_planner config 包资源配置 | 项目维护 |
| `tartan/.gitignore` | Tartan子包缓存/结果的Git排除规则 | 迁移数据/训练/导航 |
| `tartan/README.md` | 共享数据层与迁移实现导航 | 迁移数据/训练/导航 |
| `tartan/__init__.py` | 包导入边界 | 迁移数据/训练/导航 |
| `tartan/data/__init__.py` | 包导入边界 | 迁移数据/训练/导航 |
| `tartan/data/features.py` | occupancy路线的polyline与checkpoint lane/route特征编码 | 迁移数据/训练/导航 |
| `tartan/data/pose_utils.py` | 三平台数据发现、pose/时间来源、固定历史RMS、已批准观测reference或有证据的SE2/SE3外参、NED/NWU、三维occupancy反量化与证据旁注读取 | 迁移数据/训练/导航 |
| `tartan/research_score/README.md` | 当前职责层与执行入口导航 | 迁移数据/训练/导航 |
| `tartan/research_score/__init__.py` | 包导入边界 | 迁移数据/训练/导航 |
| `tartan/research_score/artifacts.py` | 训练/checkpoint与特征物化共用原子写出；临时文件位于目标数据卷，避免系统盘副本 | 迁移数据/训练/导航 |
| `tartan/research_score/configs/transfer_data.yaml` | 冻结8m/80点、完整episode split与原生数据构建参数 | 迁移数据/训练/导航 |
| `tartan/research_score/configs/transfer_methods.yaml` | 四方法、8m/80点公平性与当前Preflight输入路径 | 迁移数据/训练/导航 |
| `tartan/research_score/data/__init__.py` | 包导入边界 | 迁移数据/训练/导航 |
| `tartan/research_score/data/budget_sampler.py` | 完整episode覆盖、branch分层的精确嵌套窗口预算 | 迁移数据/训练/导航 |
| `tartan/research_score/data/core.py` | fixed-arc/mask、路线重采样、数据标识和隔离检查公共实现 | 迁移数据/训练/导航 |
| `tartan/research_score/data/route_builder.py` | 观测 occupancy 的四邻域带 unknown 惩罚 A* 路线候选与去重 | 迁移数据/训练/导航 |
| `tartan/research_score/data/schema.py` | 观测、轨迹、样本、配对、运行类型及真实源输入schema | 迁移数据/训练/导航 |
| `tartan/research_score/data/source_contract.py` | 源输入逐 tensor identity 校验与缓存版本常量；不是 Dataset | 迁移数据/训练/导航 |
| `tartan/research_score/data/source_representation.py` | 源训练使用的截断弧长桥接，当前短轨迹被重新拉满 80 点且全 valid | 迁移数据/训练/导航 |
| `tartan/research_score/evaluation/collision.py` | 离散路径点的 occupancy 碰撞与越界检测 | 迁移数据/训练/导航 |
| `tartan/research_score/evaluation/fairness_audit.py` | 迁移方法 manifest/预算/seed/轨迹/控制器一致性检查 | 迁移数据/训练/导航 |
| `tartan/research_score/evaluation/goal_benchmark.py` | 观测map-goal导航基准、D024停车和future突变负控 | 迁移数据/训练/导航 |
| `tartan/research_score/evaluation/metrics_navigation.py` | 逐任务SR/CR/SPL/进度聚合及统一SR优先checkpoint排序 | 迁移数据/训练/导航 |
| `tartan/research_score/evaluation/shortest_path.py` | 圆形 footprint 膨胀、八邻域最短路、sparse grid 与 nearest-free | 对应模块验证 |
| `tartan/research_score/model/adapters.py` | context 条件的末层零初始化四维轨迹 residual MLP | 迁移/Proxy模型与损失 |
| `tartan/research_score/model/embodiment.py` | legacy能力条件与Proxy22维history/IDmask编码；未知ANYmal ID屏蔽 | 迁移/Proxy模型与损失 |
| `tartan/research_score/model/outputs.py` | shared/residual/total 的 x0、epsilon 与 latent 输出 dataclass | 迁移/Proxy模型与损失 |
| `tartan/research_score/model/score_decomposition.py` | checkpoint backbone residual wrapper，训练及采样逐步修正 | 迁移/Proxy模型与损失 |
| `tartan/research_score/preflight/__init__.py` | 包导入边界 | 迁移数据/训练/导航 |
| `tartan/research_score/preflight/__main__.py` | Preflight模块命令入口 | 迁移数据/训练/导航 |
| `tartan/research_score/preflight/cli.py` | transfer/Proxy分阶段Preflight CLI、resolved config与门禁证据 | 迁移数据/训练/导航 |
| `tartan/research_score/preflight/registry.py` | 当前输入路径、资源、8m/80点、无future输入与输出冲突检查 | 迁移数据/训练/导航 |
| `tartan/research_score/preflight/result.py` | 检查结果ID、等级、状态、证据和修复建议数据类型 | 迁移数据/训练/导航 |
| `tartan/research_score/preflight/tests/__init__.py` | 测试包导入边界 | 迁移数据/训练/导航 |
| `tartan/research_score/preflight/tests/test_preflight.py` | 回归：test_missing_path_fails, test_bad_protocol_fails, test_unimplemented_profile_fails, test_existing_output_fails, test_future_input_rejected, test_sentinel_disk_capacity_is_unknown_warn | 对应模块验证 |
| `tartan/research_score/scripts/__init__.py` | 包导入边界 | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/audit_native_loader_identity.py` | 核对mmengine与原生NPZ读取tensor identity | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/audit_navigation_routes.py` | 对真实occupancy map-goal路线做future突变负控及路线审计 | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/audit_source_model.py` | 严格加载源checkpoint、导出normalizer/schema；CUDA使用真实源cache | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/build_navigation_tasks.py` | 唯一非重叠8m固定目标任务生成器；从已冻结split派生 | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/build_training_budgets.py` | 构造1/10/20/50/100%精确嵌套窗口预算并同步cache membership | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/build_transfer_manifests.py` | legacy源审计与Proxy三平台冻结trajectory、来源window/task索引及已批准reference门禁 | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/evaluate_navigation.py` | 唯一正式闭环评估及rollout/controller/episode-macro共享实现 | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/materialize_source_features.py` | 确定性选择源NPZ并按既有source桥语义生成4096特征缓存 | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/materialize_target_features.py` | 唯一manifest到目标特征cache物化；复用TartanTargetDataset | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/run_transfer_evaluation.sh` | 唯一正式测试调度；加载navigation_best并消费冻结test任务 | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/run_transfer_training.sh` | legacy矩阵与Proxy两任务CPU/GPU/formal调度；同run状态与完整恢复 | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/summarize_navigation.py` | 合并正式训练/导航结果为主表；报告选中checkpoint验证SR | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/train_transfer.py` | 唯一四方法target-update训练、SR导航验证选模/早停与checkpoint写出 | 迁移数据/训练/导航 |
| `tartan/research_score/scripts/validate_transfer_manifests.py` | 冻结源审计、训练长度覆盖、完整episode隔离和80点mask校验 | 迁移数据/训练/导航 |
| `tartan/research_score/tests/__init__.py` | 测试包导入边界 | 迁移数据/训练/导航 |
| `tartan/research_score/tests/test_canonicalize.py` | 回归：test_canonical_fields | 对应模块验证 |
| `tartan/research_score/tests/test_checkpoint_selection.py` | 回归：test_checkpoint_selection_matches_protocol, test_navigation_validation_restores_rng_and_training_mode, test_checkpoint_and_metadata_publication, test_generated_run_id_and_output_name_rules | 对应模块验证 |
| `tartan/research_score/tests/test_collision_geometry.py` | 回归：test_collision_and_outside | 对应模块验证 |
| `tartan/research_score/tests/test_data_contract.py` | 回归：test_cache_key_all_dependencies, test_source_adapter_identity, test_deterministic_worker_order | 对应模块验证 |
| `tartan/research_score/tests/test_goal_no_future.py` | 回归：test_future_mutation_invariant_and_goal_sensitive | 对应模块验证 |
| `tartan/research_score/tests/test_losses.py` | 回归：test_losses_targets_and_invalid_pair, test_grl_direction | 对应模块验证 |
| `tartan/research_score/tests/test_methods.py` | 回归：test_profiles_and_fairness | 对应模块验证 |
| `tartan/research_score/tests/test_navigation_metrics.py` | 回归：test_spl_exact, test_failed_episode_spl_zero_and_denominator | 对应模块验证 |
| `tartan/research_score/tests/test_representation_bridge.py` | 回归：test_bridge_and_padding_mask | 对应模块验证 |
| `tartan/research_score/tests/test_route_no_future.py` | 回归：test_route_api_accepts_only_observed_map_and_fixed_goal | 对应模块验证 |
| `tartan/research_score/tests/test_route_replanning.py` | 回归：test_replanning_changes_with_goal_and_is_deterministic, test_disconnected_is_explicit, test_rollout_replans_and_reaches_fixed_goal | 对应模块验证 |
| `tartan/research_score/tests/test_route_set.py` | 回归：test_occupancy_builder_avoids_static_and_no_future_dependency | 对应模块验证 |
| `tartan/research_score/tests/test_score_decomposition.py` | 回归：test_zero_residual_and_source_regression, test_parameterization_and_condition_isolation, test_bridge_line, test_reinitialize_only_new_diff_row, test_epsilon_diagnostic_uses_backbone_sde | 对应模块验证 |
| `tartan/research_score/tests/test_shortest_path.py` | 回归：test_straight_and_disconnected, test_footprint_inflation | 对应模块验证 |
| `tartan/research_score/tests/test_source_regression.py` | 回归：test_source_schema_observed_inputs, test_canonical_dataclasses_construct, test_checkpoint_strict_cuda_load | 对应模块验证 |
| `tartan/research_score/tests/test_splits.py` | 回归：test_d029_exact_nested_episode_coverage_and_order_invariance, test_d029_rejects_cross_split_rows, test_no_leak, test_budget_builder_publishes_matching_cache_and_manifest | 对应模块验证 |
| `tartan/research_score/tests/test_window_representation.py` | 回归：test_single_point_short_window_uses_xy_cos_sin_and_mask, test_empty_short_window_is_finite_xy_cos_sin_padding | 对应模块验证 |
| `tartan/research_score/training/cached_target_dataset.py` | 正式目标特征缓存 Dataset，按 budget/seed 筛选并重建零填充输入 | 迁移数据/训练/导航 |
| `tartan/research_score/training/losses.py` | masked L_diff/L_inv/L_swap/residual与GRL原语；trainer在真实pair上调用，classifier属于B wrapper | 迁移/Proxy模型与损失 |
| `tartan/research_score/training/methods.py` | 唯一四种正式迁移方法静态注册表；train_transfer复用 | 迁移数据/训练/导航 |
| `tartan/research_score/training/tartan_target_dataset.py` | 目标manifest特征物化Dataset；由materialize_target_features实际调用，保留唯一动态特征路径；Proxy共用camera→world→批准observed_reference occupancy loader | 迁移数据/训练/导航 |
| `tartan/tests/__init__.py` | 测试包导入边界 | 迁移数据/训练/导航 |
| `tartan/tests/test_core.py` | 回归：test_discover_trajectories_accepts_dataset_or_environment_root, test_local_transform及Proxy三维voxel/SE3/空grid回归 | 对应模块验证 |

当前CPU节点`20261002T045353Z_proxy_cpu`已完成，包括218/20增量、396/30配对侧缓存及新增B配对的真实smoke/完整恢复。来源reference/time/approval从旁注和metadata传入共享loader；cache只保存物理history/RMS，latent在线编码。原子文本发布处理实际FSX空页，恢复比较和已完成任务支持跳过。

pair.run_pair_mining的--base-window-dir复用现有数据/共同frame门禁；materialize_target_features的--reuse-cache复用同normalizer/身份物理特征；CPU smoke优先验证新增native配对。GPU续接script与报告是run产物，不是平行源码入口。来源证据、批准记录、冻结输入、必要checkpoint及实际config/command/metrics保留；cleanup_record记录精确清理。当前协议和状态无重复历史节点；独立transfer_primary接口保留但不作为本轮入口。

2026-10-08：原始backbone对照复用train_transfer.py的A-only --disable-history；默认历史A/B及原checkpoint配置保持兼容。evaluate_navigation.load_model从冻结cli选择是否启用历史再strict加载；无历史模型不可执行latent诊断。无新增源码文件。run内runner为实际命令编排产物，不是新训练入口。
