# 当前执行计划：Proxy A/B

## 2026-10-08四组实验当前节点

用户批准g1 ANYmal70/10/20、g2 Omni70/10/20、g3 Diff70/10/20、g4 Diff+Omni分别80/20→全部24条ANYmal；四组统一普通backbone无history/RMS，每5完整epoch验证。RUN_ID=`20261008T132918Z_four_groups`，DATA_ID为该ID分别加_g1…_g4，catalog来自已验收20261002T045353Z_proxy_cpu。来源、表示、预算、评价及最新原实验1区别见TRAINING_PROTOCOL开头。

本机实现与CPU验证已完成：72 passed、1 skipped（本机无GPU，跳过现有CUDA source regression）；5epoch完整遍历/恢复边界、70/10/20和80/20完整trajectory隔离、0.1…8m完整80站点、原proxy ANYmal禁训、缓存表示及旧恢复接口覆盖。shell语法与模块编译通过。未把测试通过当作实验完成。服务器已恢复，2张空闲L20；正在部署并物化真实数据，真实preflight/GPU smoke/正式训练及最终评价尚待实际产物验证。

说明位置：正式源码TRAINING_PROTOCOL.md；主run/experiment_description.md；历史归档archive/experiment1/{README.md,code/,results/,stage_summary_20260929.md,source_record.json}。归档仅最新原实验1入口/依赖43个文件和保留SPL-first结果，原目录不删；用户明确授权输出盘归档例外。g1完成后报告与旧100%指标差值；g4同时导出g1同5条测试子集，无额外模型前向。

下一命令：`PROFILE=four_groups RUN_ID=20261008T132918Z_four_groups bash tartan/research_score/scripts/run_transfer_training.sh`；由同一runner完成冻结、cache、一次几何preflight、各组4更新AMP smoke、原始nuPlan独立初始化训练及三seed离线/导航，最后汇总旧实验1对照。实际status/log/config/checkpoint位于/tj-share/cross_diffusion_workdir/runs/20261008T132918Z_four_groups。恢复同RUN_ID；不重跑完成组、不使用smoke起点。训练GPU0/1按两组一批，原Proxy结果和消融不改写。


更新：2026-10-03（Asia/Shanghai）。P0–P4/P5a已通过，218/20配对增量已通过；P5b(20261002T184329Z_proxy_gpu_smoke)已通过，P6/P7正式运行(20261002T184606Z_proxy_seed11)全部完成，状态COMPLETE。追加五项validation原因消融已完成，实际结果见第10节。

本文件是唯一实施计划；科学边界见proxy-training-protocol.md。以下为当前已实现接口与后续GPU合同。原迁移实验的ANYmal训练划分/四方法预算属于独立transfer_primary profile，不用于本轮Proxy。

## 1. 当前输入与已完成节点

CPU_RUN_ID=DATA_ID=`20261002T045353Z_proxy_cpu`。服务器源码`/zeron-vepfs/tjqc/cross-diffusion`，Python`/root/miniconda3/envs/diffusion-planner/bin/python`，输出`/tj-share/cross_diffusion_workdir`，原始数据`/tj-share/tartanground/ModularNeighborhood`。

- Diff5条按4train/1val，Omni6条按5train/1val；ANYmal24条全部test。窗口4280/457/3071，navigation140val/296ANYmal。
- 正式同split真配对218train/20val，pair侧缓存396/30；legacy10m重建与冻结10帧网格native8m候选去重并集。8m/80点共同frame门禁未变。
- 时间及observed_lcam_front_reference/occupancy来源门禁通过。真实前左相机pose的NED→NWU XY/yaw定义观测参考系；真实body外参未知，body来源为null，不猜零或继续作为本合同阻塞。
- A/B各2次真实CPU更新，B已使用新增native配对；完整checkpoint恢复、有限loss、必要梯度、B sampler、strict reload通过。baseline69passed/1CUDA skip，pair增量4passed。
- 原始nuPlan来源已由用户确认，A/B独立从checkpoints/model.pth加载；CPU/GPU smoke不得作为正式起点。
- 本轮只读确认GPU可见：4张NVIDIA L20，每卡47810936832bytes。P5b实际AMP、有效batch、参数更新、峰值显存和完整B恢复已通过，证据见对应run/gpu_acceptance.json。

实际证据与必要续接见第8节及run/status.json、cpu_report.json。保留源数据、源checkpoint、冻结输入与必要来源记录；已完成CPU步骤不重跑，不生成哈希清单。

## 2. 已批准的科学设置

已固定：Omni/Diff分别完整trajectory 80/20、seed20260911；先trajectory后window/pair；ANYmal只最终test；相同基础数据、nuPlan初始化、normalizer、验证任务、base denoising顺序；B仅同split真实Diff–Omni observational pairs；8m/80点；禁止合成训练fixture。A为共同历史条件backbone，B增加residual/辅助目标；不增加nuPlan source update。正式仅全量base数据×A/B×seed11，无默认消融/预算子矩阵。D024保持。

Q1–Q4均已获用户确认；自动按阶段推进，不再请求重复批准。接口按第3节落实；只有真实数据/科学冲突才停止依赖步骤，不能自行放宽规则。

### Q1：共同预算和规则、各自早停（2026-10-02已确认）

原协议同时要求相同更新次数与各自SR patience10，可能不相容；ReduceLROnPlateau也可能产生不同实际LR轨迹。

用户接受按此前建议推进，Q1记录为：相同初始LR、预算上限/下限、验证时点、scheduler及早停规则，A/B各自按SR停止，报告实际updates/LR，不声称逐步相等。共同：batch64、AdamW lr1e-4/wd1e-4、clip5、AMP、seed11；min5000/max10000、每250验证、patience10；scheduler mode=max/factor0.5/patience2/min_lr5e-6。不得自行改成强制10000或双方共同停止。Q2/Q3于20261002T042210Z已批准；Q4最新结论见下节。

初始化澄清（20261002T042210Z）：A/B均独立从原始nuPlan backbone开始，源路径记录为`/zeron-vepfs/tjqc/cross-diffusion/checkpoints/model.pth`。禁止旧1/10/100%微调、旧ANYmal/adapter、其他Proxy任务或smoke checkpoint起步。路径和strict load不是来源证明；复用来源记录，来源不明则停止，不生成默认哈希清单。首次不继承optimizer/update等状态；只有同RUN_ID、同方法的断点恢复才加载自身last完整状态。详见根协议4.3。

### Q2：完整B objective、pair曝光与classifier（20261002T042210Z已确认）

原协议缺项已按下列设置收口并获批准，数字是事前工程起点，**不是已验证最优设置**：

1. A/B每update读取相同base64窗口，末尾不足64不丢；B额外均匀有放回抽32对train pairs，每update一次；合成一次model/classifier optimizer step，无source update。pair不得混入base sampler。
2. `L_diff_A=L_base`；`L_diff_B=L_base+(L_pair_diff+L_pair_omni)/2`，保留协议“两侧denoising”。B额外曝光/计算是比较差异的一部分，不能将效果全归因于辅助项。
3. base噪声独立于pair/验证，A/B base t/epsilon一致；pair两侧共享t（uniform[0.001,1]）及标准正态epsilon数组，在各自ego-local归一化空间加噪。这是随机数耦合，不是物理干预。
4. 在共同frame和共同有效mask上：`L_inv=MSE(shared_a,shared_b)`；`L_swap=(MSE(shared_a+delta_b,y_b)+MSE(shared_b+delta_a,y_a))/2`；`L_res=(MSE(delta_a,0)+MSE(delta_b,0))/2`。MSE是四维平方和/有效点数，不再除4。
5. classifier输入：共同frame下shared ego未来输出的masked mean `[2P,4]`；`Linear(4,64)→SiLU→Linear(64,2)`，Diff label0、Omni label1，两侧均衡、CE取mean。它约束shared轨迹可辨识性，不另称encoder feature不变性。
6. `L_sep=CE(classifier(GRL(pool(shared))))+L_res`；GRL=1；`lambda_inv=0.1, lambda_swap=0.1, lambda_sep=0.01`、内部residual权重1。classifier与planner同一AdamW/同一次step，GRL仅反转传回shared的梯度。无额外optimizer/warmup/权重搜索。记录原始/加权各项及CE。
7. val pairs只冻结诊断，不训练、不调权重、不选模；val pair=0允许缺此诊断，train pair=0停止B。不能根据smoke数值大小临场调整λ。

### Q3：8m表示、真实anchor重建和配对门禁（20261002T042210Z已确认）

用户已批准如下空间suffix替代旧80原始帧截断：

- base anchor=`range(20,n_pose-1,10)`。从当前pose沿后续完整记录路径寻找XY8m，不受未来80原始帧限制；站点`linspace(0,8,80)`，第一点为当前anchor，模型另有current token，loss仍只80目标槽。短尾保留真实mask，不外推。heading unwrap插值；连续重复XY弧长保留首个姿态，不把原地转动伪装成空间距离，并报告该表示限制。
- base固定goal取8m端点，短尾取真实终点；非重叠导航任务只取完整8m。context只看当前occupancy、已冻结goal与规定过去历史；goal-selection oracle限制保留。沿用当前占据图路线；自车历史按已批准Q4加入，actor槽仍按无观测处理。
- 184候选用原pose+原10m/2.5m配置回放定位，取旧段start_frame的真实观测frame作新anchor，再重建XY8m/80点；记录旧插值entry到新anchor位移。不能给插值位置伪造occupancy。pair anchor可不在base stride10中，但只进入独立pair cache。
- 共同frame取Diff观测anchor：`T_diff=I`、`T_omni=T_world_to_diff @ T_omni_to_world`。只由pose确定，禁止ICP/轨迹拟合。两侧保留各自真实8m goal，作为近似匹配，`common_goal_id=null`，不虚称同一goal。
- 已批准8m门禁：重算XY80点后center≤3m、entry/exit<1m、对应点mean<1m/max<2m；Chamfer/overlap/heading差仅诊断。两侧完整8m且80点全valid、同map/split、可回源、SE2/输入/数值有效、去完全重复pair。阈值先冻结，不因pair少调整。
- 不重新全库挖掘。固定split门禁后train=0就停止B，记录原因；需扩搜索/放阈值必须由用户决定，不移动trajectory凑pair。

### Q4：历史上下文、未知ID与代理几何（已批准）

- 共同输入：anchor前20帧真实SE(2)历史→16维学习latent；另有纵向/横向速度和yaw rate的历史RMS三项及mask。没有可靠dt不得造速度。具体时间、shape、归一化和路由见3.3。
- 16维仅是工程起点，不逐维命名为能力；三项摘要是已发生的运动强度，不是最大能力。无物理辅助planner损失、VAE、额外对比学习、维数/seed搜索。历史/摘要在每个8m导航任务内固定。
- ID保留car0、ANYmal1、Diff2、Omni3。Proxy只训练2/3；B训练每样本以0.2概率将ID贡献置零，val Omni/Diff使用ID，ANYmal推理ID贡献恒零。A无ID分支。
- 代理圆半径Diff/Omni0.50m、ANYmal0.35m；工程假设，A/B相同，仅供geometry，不冒充能力真值。旧无来源ability[10]不属于Proxy输入，静态约束只记录来源，不补数。
- ANYmal过去历史仅作为最终推理时的已观测输入；网络和normalizer冻结、不跨测试任务拟合、不读取未来记录补历史。CPU可做结构/字段/缓存构建，禁止模型前向或测试表现分析。

研究依据（不是本项目效果保证）：[GNM](https://arxiv.org/pdf/2210.03370)支持历史导航上下文；[DreamWaQ](https://arxiv.org/pdf/2301.10602)使用显式速度与16维latent；[HIM](https://arxiv.org/pdf/2312.11460)的消融支持显式/隐式表示并存。后两者属于底层控制，不能照搬其仿真监督或宣称本项目有效。[Concept-Residual研究](https://arxiv.org/abs/2312.00192)提示自由分支可能重复编码显式概念，不能声称完全解耦。[TartanGround](https://arxiv.org/html/2505.10696v1)支持Omni/Diff运动类型差异，但不提供本项目完整十维能力表；三平台时间来源已通过P1，来源记录保留在CPU run。

## 3. 当前数据和模型契约

以下为已实现合同；旧transfer接口只在其原profile保留，Proxy禁止回退旧ability/ID默认值。路径保存服务器绝对路径及dataset-relative来源，ID不依赖run时间；文件名ASCII。split统一`train|val|test`。

### 3.1 trajectory→window→cache

`trajectory_key=map_id+'/'+embodiment+'/'+trajectory_id`，Proxy的episode_id等于完整key。

```text
trajectory manifest每行：
trajectory_key,map_id,embodiment,trajectory_id,split
pose_path,metadata_path,occupancy_dir,pose_count,sample_rate_hz,time_source
frame_convention,split_seed=20260911,split_algorithm
platform_id,robot_radius_m,timestamp_path_or_null,frame_time_policy,body_heading_source
```

每平台按key排序，用各自重新初始化的`random.Random(20260911).shuffle`；前`max(1,round(.2*N))`为val，其余train（Python round）。不按长度/window/pair数量分层；ANYmal全test。保存具体membership，派生物只join、不重算。训练平台少于2条则停。`assert_no_split_leak`用trajectory_key，不能默认map_id（同图允许跨split）或裸trajectory_id。

window沿用现有sample_id/episode_id/anchor_index/domain/embodiment/split/branch/current_state/fixed_goal/route_set/trajectory/provenance，落实：

```text
trajectory_key,platform_id,history_start_frame,history_end_frame,history_policy="past20_anchor_exclusive"
sample_id=<trajectory_key>:anchor:<six_digit_frame>
trajectory.raw_reference,source_start_frame,source_end_frame
trajectory.fixed_arc_length_80:[80,4],valid_mask:[80]
trajectory.arc_metric="xy",length_m=8,stations_m:[80],measured_arc_m
current_state.anchor_world_se2:[3],frame_convention
fixed_goal.xy_local:[2],actual_distance_m,rule
route_set.map_reference,future_gt_dependency=false
provenance.trajectory_manifest,representation_policy,time_source
```

无有效点/损坏文件必须记录拒绝与trajectory覆盖，不造数据补齐。base和offline保留有有效点的短尾，pair仅完整段；ANYmal任一trajectory无可用window须明确未覆盖，不能称全量完成。

两Dataset统一返回`(inputs,target,valid_mask,metadata)`；metadata含sample_id/key/platform_id/split、history frame范围、时间与批准reference heading来源，body来源保留null。修改全部调用者；旧transfer保持旧表示政策。cache保存这些字段数组、manifest路径/sample IDs/count/表示政策/normalizer引用；Proxy缺字段不得回退ID1。

| tensor | dtype/shape（batch B） |
|---|---|
| ego_current_state | float32 `[B,10]`，简化输入`[0,0,1,0,0,0,0,0,0,0]` |
| neighbor_agents_past | float32 `[B,32,21,11]`，无actor观测零填充；21不代表已接通自车历史 |
| static_objects | float32 `[B,5,10]`，零填充 |
| lanes / route_lanes | float32 `[B,70,20,12]` / `[B,25,20,12]` |
| speed_limit / has_speed_limit | float32 / bool `[B,70或25,1]`，零填充 |
| target / valid_mask | float32 `[B,80,4]` xy_cos_sin / bool `[B,80]` |
| platform_id（metadata） | int64 `[B]`，仅B条件使用 |
| ego_history / history_mask | float32 `[B,20,4]` / bool `[B,20]` |
| history_dt / history_dt_mask | float32 `[B,19]`秒 / bool `[B,19]` |
| motion_rms / motion_mask | float32 `[B,3]`物理单位 / bool `[B,3]` |
| sampled_trajectories / diffusion_time | `[B,11,81,4]` / `[B]`，ego为0轨迹，current为0时点 |
| score / 三个x0分解 | `[B,11,81,4]`，loss取`[:,0,1:]` |
| sampler prediction | `[B,11,80,4]`，物理坐标，评估ego |

主干normalizer固定来自同一`checkpoints/args.json`的Config，不拟合三平台数据，不读旧source cache；不要误用另一个normalization.json覆盖而不记录。路线无候选原因保留。

### 3.2 pair与共同frame

扩充现有CanonicalPair：pair_id/split/map；双侧完整key、segment/sample/anchor；原候选路径/ID/配置/segment_index；raw frame范围/弧长起点；重建8m/80/XY/范围；`T_a/T_b:[3,3]`；common原点/yaw来源；两goal common坐标；metrics_8m、pair_valid、rejection_reasons；pair_level=`observational_matched`，intervention/counterfactual=false。

`CachedPairDataset`放现有`training/cached_target_dataset.py`，按sample ID引用去重pair cache，不为每对复制张量。返回a/b各自`(inputs,[80,4],[80],metadata)`及pair_id、pair_valid、T_a/T_b、质量；batch维P，`M_pair=pair_valid[:,None]&mask_a&mask_b:[P,80]`。train拒绝引用val/test。独有pair窗口不能加到共同base manifest。

变换：各自ego-local前向→取ego future→shared inverse normalizer；delta仅乘std、不加mean→shared XY做R*x+t、方向cos/sin做R*v；delta XY/方向只乘R、不加t→target同样变换→按固定`[20,20,1,1]`缩放后算Q2（等于checkpoint ego std，配置须一致）。不可直接比较各自local归一化tensor；不可在loss中归一化方向破坏`total=shared+delta`。共用core变换，SE2往返float32误差≤1e-4。

pose/occupancy采用已批准observed_lcam_front_reference，来源及批准记录通过P1；真实camera-to-base外参未知，当前合同不依赖它且不猜外参。批准参考系证据缺失或实样不一致时停相关派生。即使数学对齐也不等于环境潜变量一致或真实反事实。

### 3.3 历史数据、模型路由与初始化（唯一实现合同）

1. **索引/时间**：anchor=a，历史严格取`pose[a-20:a]`，末帧a-1，共19个相邻间隔；不得跨trajectory。base使用a≥20，pair真实anchor不足20直接拒绝`insufficient_history`，不移动anchor凑资格。20帧pose及批准参考系heading全部有效才接纳，不用未来或合成值补历史；历史mask仍显式保存。时间优先真实timestamps，其次能证明与pose同步的metadata/generator固定dt；记录来源和换算，不能把采样声明直接当实测。无时间证据保留pose、dt=0/mask=false、RMS=0/mask=false，记录缺口；可继续不依赖时间的CPU工作，**三平台时间语义未核实前不通过P1最终门禁、不启动正式训练**，不要让ANYmal未知时间成为临场调参机会。
2. **计算**：历史XY和朝向统一到anchor批准观测参考系，`x,y,cos(yaw_i-yaw_a),sin(...)`。各相邻位移在该间隔起点真实观测参考点heading下旋转，除正dt得到vx/vy；yaw差unwrap后除dt得到omega；RMS为有效间隔速度平方的算术均值再开根，三项均需19个已证实正dt方为有效。真实零运动的mask为true。本轮以真实前左相机pose定义观测参考点，不推断基座外参，不用路径切线代替朝向。
3. **归一化/缓存**：cache存上述物理值、mask与来源，不缓存学习latent。网络内部历史XY除20m，cos/sin不缩放，dt除1s，RMS分别除1m/s、1m/s、1rad/s（固定单位转换，非拟合量）；不clip、不估计ANYmal或Omni/Diff总体统计。缺失值先清零并拼mask。主干`observation_normalizer`只处理旧特征key，新字段旁路后合并，禁止未知字段掉落或二次归一化。
4. **共同history模块**：在现有`diffusion_planner/model/module/encoder.py`定义小型`ProxyHistoryEncoder`，输入flatten(history×mask)80 + mask20 + dt×mask19 + dt_mask19，总138维；`Linear(138,64)→SiLU→Linear(64,16)`产生z。`c=concat(z, motion_rms×motion_mask, motion_mask.float)`为`[B,22]`。无dropout。接`Linear(22,hidden_dim=192)`，其weight/bias全零；把投影`[:,None,:]`加到现有fusion后的`encoding[B,107,192]`，保持token数/原checkpoint结构。encoder返回额外`proxy_context[B,22]`。公共模块在严格加载原始backbone后通过显式`enable_proxy_history()`挂到`backbone.encoder.encoder`，不可再次对整模型apply初始化覆盖预训练权重。
5. **A/B调用**：A直接调用启用history的backbone。B的`score_decomposition.py`预先调用同一encoder的`encode_proxy_history(inputs)`取得c，在本次forward副本中以内部key`proxy_context`交给encoder复用，避免重复编码/断梯度；该key不可从磁盘cache读取。B `EmbodimentEncoder`的Proxy分支为`Linear(22,64)→SiLU→Linear(64,64)`加`Embedding(4,64)*id_mask`，只替换Proxy条件接口，旧transfer保持原签名。wrapper接现有ZeroResidualAdapter，仍只修正ego，不修正邻车或current token（delta[:,0,1:]有效，其余零）。classifier归wrapper，A不创建。采样每一步使用同一c/ID条件，不能只在采样结束修正一次。ID dropout独立CPU generator seed14，每侧样本各抽一次，采样过程不重复抽。
6. **初始化/恢复**：原始nuPlan strict load后，在`torch.random.fork_rng`隔离的seed11上下文创建共同history模块，A/B公共初始参数逐tensor相同；创建B专用模块不扰动base噪声流。公共投影及residual末层初始化零，先构造再置零，避免父模块初始化覆盖。history潜层首步梯度可能为零，第二步检查。新run全新optimizer/scaler/scheduler；resume/eval先构造相同完整结构再strict load完整checkpoint，不重新加载源权重、不重置投影。记录原始源路径/来源/加载选择键，来源不明则停止依赖。禁止用100%旧微调、smoke或A结果初始化B。
7. **配对/GRL**：各侧用自己的past20，先独立forward再按3.2到共同frame，仅swap delta。不得交换历史/goal或把common-frame轨迹用于生成输入。classifier只直接读取shared轨迹pool，不直接读取z；GRL经shared自然传回history编码器，不额外detach或另加z对抗。约束冲突是科学限制，不能为优化方便改路由或权重。
8. **导航**：冻结任务保存起点的同一历史数据/摘要或其明确cache引用；每次replan重复使用起点坐标下的历史上下文（行为签名，不是当前坐标下的物体位置），不随当前pose重旋转历史、不读取轨迹后续真实pose。主干当前地图/路线仍按当前pose局部化；真实 executed path仅更新导航位置/yaw。任务历史过时、切线yaw控制及没有物理时钟须进报告。
9. **解释诊断**：P7附Omni/Diff train的冻结best z线性读出RMS（带截距，numpy lstsq，无超参搜索）；val上MAE、R²及train均值基线，零方差R²写null。A/B各自拟合同一train ID集合，无梯度回planner、不做ANYmal拟合。只报告可读出性，不称因果或使用证明。shared/total的heading对位移夹角与路径转角在Omni/Diff val报告均值，零位移不计夹角并记录数量；不把空间80点推导成速度，不把残差四维命名为能力。

### 3.4 训练细节的唯一解释

- B base64、pair32是两个batch；pair项各侧按有效点SSE归一化再半和，不与base混成一个全局平均。微批累积按各项全batch分母加权，仍一次AdamW/scaler/clip/scheduler流程；不按微批均值相加。原swap_loss返回sum，Proxy须显式实现Q2的半和而不影响旧transfer。
- base target归一化后先SDE加噪；无邻车label时邻车全零且不参与loss，current token不加噪、不算loss。pair每对同t/epsilon、不同对独立。缺mask有效点是数据错误，不用clamp掩盖。
- 每250实际optimizer update验证；AMP溢出跳步不计update。scheduler监视严格改善的macro SR，threshold=0、threshold_mode=abs、cooldown=0、eps=1e-8；设置中保留Q1 factor/patience/min_lr。scheduler只在验证后step。stale从第一次验证开始累计，只有SR严格提高清零；update≥5000且stale≥10停，最多10000。SR并列的SPL等只更新best，不清零stale。所有验证任务都D024仍有效，invalid_map全无有效任务则停止。
- base sampler和t/epsilon独立CPU generator seed11；pair索引seed12、pair噪声seed13、IDdrop seed14，各流序列化。先在CPU生成再传设备保证同base前缀。公共模型dropout按每update在隔离RNG上下文播种seed11+update，A/B base相同，B pair使用独立seed100000+update；val保存/恢复模式及RNG。微批配置A/B一致并记录，不能保证任意更改微批后位级相等。
- CPU实现应去除Proxy路径的硬编码.cuda()，所有输入、采样和SDE使用模型实际device；CPU禁AMP。保留Config原normalizer数值，只把tensor迁移到实际设备。GPU正式开启AMP，不改变已批准目标。

## 4. 改动文件和职责

**目前无需新增源码/测试/配置文件**；run中的JSON/PT/CSV/报告属于产物。若确需新源码，先在本计划说明原模块为何不能承载、准确文件名/职责/依赖，并同次登记ARCHITECTURE；不得恢复历史实现或新建训练入口。

除根路径外，下表以`tartan/research_score/`为基准。

| 文件 | 修改职责/调用关系 |
|---|---|
| 根`tartan/data/pose_utils.py` | 三平台发现、metadata时间/观测anchor身份；builder/pair共同读取 |
| `data/schema.py`,`data/core.py` | manifest/pair契约、XY弧长及SE2点/差量公共函数；保留旧transfer政策 |
| `scripts/build_transfer_manifests.py`,`validate_transfer_manifests.py` | 增proxy_ab与trajectory/window阶段；不执行旧源审计；验证完整key隔离/来源/shape |
| `configs/transfer_data.yaml` | 原transfer不变，增加proxy_ab段：split/Q3/平台注册表/normalizer |
| `scripts/build_navigation_tasks.py` | 混合平台key/geometry/协议身份；冻结全val和全ANYmal非重叠8m任务 |
| `training/tartan_target_dataset.py`,`scripts/materialize_target_features.py` | base/pair共用特征物化、显式metadata和frame；分开cache |
| `training/cached_target_dataset.py` | metadata返回及同文件轻量pair引用Dataset |
| 根`pair/trajectory.py`,`matching.py`,`run_pair_mining.py` | 回放旧候选、真实anchor8m重建、同split门禁；同CLI增reconstruct |
| 根`pair/config.yaml`,`visualization.py`,`README.md` | 保留候选参数，增加明确proxy_gate；标注common frame/拒绝原因 |
| `training/methods.py`,`configs/transfer_methods.yaml` | registry增proxy_a/proxy_b、profile=proxy_ab/source=false；一个配置段定义Q1/Q2/路径 |
| 根`diffusion_planner/model/module/encoder.py` | 3.3共同history编码、22→192零投影；主干strict load后显式启用 |
| `model/embodiment.py`,`model/score_decomposition.py` | Proxy 22维context/IDmask及ego-future残差；classifier放wrapper；保持旧transfer签名 |
| `training/losses.py`,`scripts/train_transfer.py` | 真实pair objective、GRL、噪声流、各项分母、选模及完整恢复 |
| `evaluation/goal_benchmark.py`,`scripts/evaluate_navigation.py` | 按平台几何/条件；Proxy一致rollout frame；同入口增加offline |
| `evaluation/metrics_navigation.py`,`scripts/summarize_navigation.py` | 完整key宏平均、显式两任务矩阵、公平性/覆盖与报告 |
| `scripts/run_transfer_training.sh`,`run_transfer_evaluation.sh` | PROFILE分支、两任务/GPU列表、同RUN_ID恢复；不改原transfer矩阵含义 |
| `preflight/registry.py`,`preflight/cli.py` | proxy_ab只检查所需输入/门禁/环境，不能借transfer PASS |
| 现有tests | 按第5节扩展，不生成训练fixture或另建测试框架 |
| 根`ARCHITECTURE.md`,`PROJECT_STATUS.md`,`proxy-training-protocol.md`,`README.md`；`experiment_execution_plan/SERVER_HANDOFF.md`,`01_preflight_framework.md`,`decisions.md`,`protocol_changelog.md`,`risk_register.md` | 随冻结/实现同步职责、状态和限制，不复制计划 |

依赖方向不变：data不依赖trainer；model不导入训练入口；GRL在trainer/loss调用classifier前施加。trainer复用evaluator验证函数，evaluator不导入trainer，共同属性来自配置。复用artifacts.publish原子发布，不另建发布框架。

## 5. 阶段、验收、停止和续接

顺序P0配置→P1 trajectory/loader→P2 base/tasks→P3 pair门禁→P4训练/评价/恢复→P5a真实CPU smoke→P5b GPU smoke→P6正式训练→P7测试/报告。代码可先按已冻结接口实现，不因P3数据阻塞停止独立CPU工作。自动验收通过即继续；仅真实科学/数据/环境阻塞才停，不增加逐阶段审批。

### P0 冻结配置与部署节点

- 前置：Q1–Q4已批准，无需重复问。同步原Proxy协议、两个YAML/平台表，更新ARCHITECTURE状态，不改旧transfer输入。
- 输出：`runs/<RUN_ID>/config.json,command.sh,status.json`，实际配置副本、批准决定/来源、代码清单/环境；不生成默认hash清单。本机更新后只同步正式清单文件，不宽泛delete同步。
- 验收：配置没有未决超参；time/frame等数据事实可记录为P1待核实，Q4不靠ANYmal拟合；两地使用同一批准内容。检查ARCHITECTURE在本批实现完成时一次进行，不为每次文档修改反复执行。
- 停止：未决参数/协议冲突/来源不明；记录缺项及受阻阶段，不能临场选能力表或权重。

### P1 三平台loader与先冻结trajectory

- 已有目录/pose数学，缺manifest；改pose_utils/schema/builder/validator/data配置，不重做nuPlan审计。
- 输入三平台pose/metadata/occupancy及Q3/Q4。核对key唯一、pose shape/finite/count、metadata声明及3.3时间/批准reference heading/历史资格；ANYmal仅结构/身份清点，不做性能、能力拟合或阈值选择。
- 输出`data/<DATA_ID>/trajectories.jsonl,trajectory_summary.json`：数据不变则9 train/2 val/24 test，完整membership、frame/time来源及拒绝原因。冻结后所有派生物join它。
- CLI见第6节；存在manifest只允许显式恢复并读取原membership，拒绝覆盖或重新shuffle。
- 测试：在`tartan/tests/test_core.py`、`research_score/tests/test_splits.py`验证完整key/平台独立确定性/取整；数学SE2小数组可测，不作为训练fixture。每平台选1个真实pose/occupancy anchor核对坐标约定/索引。一次结果检查验证9/2/24、全key隔离及来源。
- 停止/续接：数据数量改变先记录差异、按协议取整，不能悄悄排除新trajectory；frame无法证实则停相关派生，保存目录列表、manifest、缺失来源。不得猜camera外参。

### P2 base window、cache及val/ANYmal任务

- 前置P1、Q3/Q4；改builder windows分支/core/Dataset/materializer/navigation builder。不用旧build_training_budgets给Proxy套1/10/100%；base是全部有效train窗口。
- 输出`data/<DATA_ID>/base_train.jsonl,base_val.jsonl,anymal_test.jsonl`及对应`cache/<DATA_ID>/*.pt`。保留短尾mask/branch；ANYmal cache只供最终评价，训练CLI不接受它。
- 冻结`navigation_val.jsonl`（Omni/Diff两条val）与`navigation_anymal_test.jsonl`（24条test的兼容任务）。每trajectory从真实frame取8m，下一anchor≥上一end；列无完整任务trajectory，不能复用旧test63。具体anchor起点见Q3，task builder与base使用同一历史资格规则。
- 本轮按pose数/stride计算候选上界：Omni+Diff共4,737窗口，ANYmal3,071；**尚未生成，不是通过数**。缓存sample ID索引/metadata接口见第3节。
- 测试：扩展`test_window_representation.py,test_data_contract.py,test_route_no_future.py`，覆盖8m站点/短尾mask/平台字段及target不进入context。每平台真实1个window核对cache/manifest的ID、shape、值；一次检查全部split数量/覆盖/IDs。ANYmal只结构校验，不前向模型。
- 停止：ID1默认、未来target进入route、train含ANYmal、无有效点、测试trajectory遗漏未说明。记录确切sample/路径，不插值造图、不用旧cache补齐；保存完成产物与失败点，同DATA_ID续接。

### P3 真实候选回放、8m重建和门禁

- 前置P1/P2/Q3。先用完整key join split，分train/train、val/val、跨split；跨split拒绝，ANYmal来源错误；不根据pair数改split。
- 旧段10m按XYZ弧长切、匹配64点按XY，必须从原pose回放start_frame再重建XY8m，不是64→80插值。修改根pair三模块与core/schema，所有张量真实可回源。
- 输出`pairs/<DATA_ID>/train.jsonl,val.jsonl,rejected.jsonl,gate_summary.json,train_windows.jsonl,val_windows.jsonl`及去重的`cache/<DATA_ID>/pair_train.pt,pair_val.pt`。summary追踪184输入，分split/trajectory-pair计数、拒绝原因、unique windows/trajectories、重叠度；pair数不等于独立trajectory数。
- 现有可视化器固定排序取train/val及阈值附近共≤6对，标原候选/8m/frame/goal/mask/指标；不重画原50张，不建重复审查包。
- 测试：扩展根`pair/test_trajectory.py,test_matching.py`，真实来源回放、split拒绝、T往返、80点与8m指标；错误split标签副本可测validator，但不保存为训练数据。结果验收accepted全部通过Q3、train>0；val=0仅缺辅助诊断，不阻塞导航选模。
- 停止：来源不能回放、共同frame不成立、train为空、未冻结阈值、mask错误。保存候选ID/拒绝统计，停止B依赖；不能自动放宽门禁或全库重挖。A代码可继续，正式矩阵仍未就绪。

### P4 接通A/B、评价和恢复

- 前置P2/P3接口冻结；按第4节修改training/model/evaluation/runner/preflight。METHODS增proxy_a/b，profile=proxy_ab/source=false。train_transfer内分流；Proxy不要求source-cache或budget，不用budget100过滤未赋membership的数据；传旧ANYmal训练cache须拒绝。
- 复用strict nuPlan加载、AdamW/AMP和selection key。base窗口按sample_id排序，每epoch用独立CPU generator(seed11)生成randperm，不加平台或trajectory重采样；保存排列/游标。base噪声generator(seed11)、pair索引generator(seed12)、pair噪声generator(seed13)各自独立并保存state；pair/validation不消费base流，wrapper初始化及额外forward不改变A/B base IDs/t/epsilon。B逐样本条件，A不接ID、不读pair manifest。新增classifier参数不改变base初始化，恢复不重新播种。
- 按Q2接通两侧forward、共同frame、GRL/classifier；inv/swap显式接point mask/pair_valid，只算ego future；无有效pair不能NaN或伪报成功。测试零residual等价、每步sampling correction。x_start字段叫score不构成因果score证据。
- val：先trajectory内聚合再两条trajectory等权，单列Omni/Diff；selector SR↑/SPL↑/CR↓/progress↑/早update，stale仅SR，loss仅诊断。loss累计SSE/valid点数，不能用不等batch均值冒充micro。
- rollout：manifest决定平台/geometry/条件。Proxy每次重规划输入坐标以当前pose为原点/yaw0；在原任务地图网格规划后把路线旋到当前heading，预测逆旋转/平移回任务frame再检测碰撞。route_inputs已将路线转入当前批准参考系，并通过显式Proxy分支与90度变换回归；原transfer语义保持。沿用≤1m执行/0.1m插值、最多16次、成功0.75m、3次<0.05m stuck及D024。
- evaluator同入口增`--evaluation-kind offline|navigation`；registry识别B。offline分开denoising诊断与sampler单draw，不能用teacher-forced预测算生成ADE。summary读取显式两任务矩阵，不用旧glob。
- 恢复：trainer增`--resume <last.pt>`；初始update0保存last；每250更新验证后原子写last，best改善立即写navigation_best。last保存model/classifier、optimizer/scaler/scheduler、update/base与pair曝光数、epoch、base排列/游标、pair sampler RNG、Python/NumPy/Torch CPU/CUDA RNG、noise generators、next_val/stale/best_sr/selection key、history/分子分母/冻结引用，并保存best模型状态及其update。恢复以完整last为提交节点，若独立best文件的update不一致，从last内best恢复，避免两文件发布间中断造成错配。普通区间最多重做249更新；若完成第250步但保存前中断，最多重做250更新，日志明确重放区间；恢复从保存的下一batch开始。
- 恢复核对实质配置、manifest IDs/count、schema/frame/normalizer，不默认hash；同RUN_ID，不另建retry。best只供评价，last供恢复，旧weights-only snapshot不能冒充恢复；配置不符/损坏停止且保留原文件。不每epoch留快照。
- 一次实现检查：扩展`test_methods.py,test_losses.py,test_score_decomposition.py,test_checkpoint_selection.py,test_navigation_metrics.py,test_route_replanning.py`和已有preflight测试，验证ID、GRL符号、point-mask分母、swap/SE2、selector/D024、offline最后有效点。数学小tensor可测，不建synthetic Dataset/cache。相关测试通过后本批最终跑一次全套和architecture，不重跑历史审计；真实连通性由P5a/P5b验证。
- 停止：硬编码仍存、A误用pair、B应有梯度未接通、test入val、Omni geometry缺失、frame不闭合、恢复缺失。CPU通过仅记实现检查；GPU不可用不声称smoke成功。

### P5a 最小真实CPU smoke（无GPU也必须完成）

- 前置P1–P4相关门禁；从base_train固定取2个真实窗口（Omni/Diff各1），train pairs优先取新增native候选按pair_id首个，若没有native再取legacy首个，共1对；不运行ANYmal。使用同trainer的`--smoke cpu --device cpu`，base2/pair1、2 updates、禁AMP，标purpose=engineering_cpu，非正式batch/结果。
- 检查A/B forward、各loss、backward/optimizer、第二步history/residual梯度、classifier、B sampler一次、完整checkpoint保存和strict重载输出一致。seed/索引选取与loss数值无关，保存实际ID。两个val真实任务只做route/frame输入与D024检查；不以CPU短跑选正式模型。
- 产物`runs/<CPU_RUN_ID>/cpu_smoke/{config.json,checks.json,logs/,checkpoint.pt}`；CPU验证通过后状态`CPU_READY_GPU_PENDING`。train pair为空时B smoke阻塞，继续可独立A实现/CPU检查，不能宣布CPU全链通过。
- CPU smoke失败只定位对应接口/真实样本，修复后重跑受影响项；不生成合成fixture。CPU检查不替代下述GPU AMP/显存和正式batch恢复检查。

### P5b 最小真实GPU smoke

- 前置P5a通过及GPU。仅冻结Omni/Diff train base与accepted train pairs、Omni/Diff val；不运行ANYmal。子集是固定真实ID，不新划split。
- 同trainer增`--smoke gpu --device cuda`，保持objective/shape/optimizer/effective batch，取正式base sampler的前4个batch；明确max_updates=4、min_updates=0、val_every=2、禁用smoke早停，B pair32可有放回。导航smoke只取每个val平台按sample_id排序第一条地图有效且路线非空的真实任务，共2条，选择不依赖模型表现；保存选中ID，不能改写正式val manifest。若一个平台无这样的任务，停止并诊断，不能用全D024停驶结果冒充推理连通。独立UTC smoke RUN_ID，config标purpose=engineering_smoke、eligible_for_formal_result=false；不能用smoke best替代正式nuPlan初始化。
- 恢复对照一次：B连续4步，对照第2步保存last并resume到4步；同smoke run下continuous/resumed两个明确用途检查子目录，非版本/retry。比下一batch IDs/t/noise/计数/selector及loss/参数（预先rtol1e-4、atol1e-6）；超差先定位随机流/非确定算子，不随意放宽。
- 验收：A/B base计划一致；loss有限，应训练模块有梯度（零初始化adapter第一步前层为0正常，第二步检查）；classifier更新/GRL方向正确；同split/mask门禁实际调用；两个平台val与宏平均可复算，D024一致；恢复可继续、best可加载推理。记录peak显存、step/val耗时和cache实占用。
- 产物`runs/<SMOKE_RUN_ID>/config.json,command.sh,logs/,smoke_checks.json,train/,eval/`。CPU数学单测不代替此真实smoke，smoke不形成性能/因果结论。
- 停止：环境/显存、NaN、wrong frame、pair缺失、ANYmal读取或恢复不一致。OOM可共同固定microbatch并累积保持effective base64/pair32及一次step，只重验受影响smoke；不能降低有效batch/预算冒充同条件。

### P6 正式矩阵与持续运行

P5b通过后新正式RUN_ID，共享冻结DATA_ID，从同nuPlan初始化；不续smoke权重，不启动原12任务。

| 任务 | 数据/模型 | 训练与选模 |
|---|---|---|
| proxy_a_seed11 | 全部base_train，共同history backbone，无ID/pair | Q1；固定val的navigation_best |
| proxy_b_seed11 | 相同base+合格train pairs，history/ID/residual/classifier | Q1/Q2；同val/selector，额外曝光单报 |

- 每250验证，min5000/max10000，最多40轮/方法，按批准Q1停止。单卡顺序即可，两独立GPU可并发；默认单卡，不沿用固定四卡。run保存task_matrix.json，不能仅数metrics文件判完成。
- 每次验证保存状态/last/best及原始loss、各平台val、base/pair曝光与重复抽样数、最佳update、wall time/peak memory。不中途新增消融/seed。
- 显式同RUN_ID+RESUME=1；完成且状态/checkpoint一致的任务跳过，未完从last续。目录存在但无可恢复last明确失败，从初始last或最后提交last恢复，不偷偷删除重训。
- 训练完成判据：两任务metrics.status=complete且停止原因合法，best/last/config/commands/log齐全，共同base前缀顺序/val一致，未读ANYmal。此时只记训练完成，P7未完不叫实验完成。

### P7 最终ANYmal评价与报告

- 前置两任务训练/选模结束，所有权重、阈值、历史/物理摘要定义、推理规则锁定。各自navigation_best仅最终test；manifest覆盖全部24，不能旧5条subset。A/B同任务/normalizer/种子与几何。
- offline必做：固定seed20260914的t/noise计算normalized masked MSE；生成用固定sample排序索引、seed=10000+sample_index的一次sampler draw，无best-of-N；teacher-forced x0不当生成结果。window ADE是有效xy欧氏距离均值，FDE最后有效点；heading不进入ADE/FDE。
- 每window保存SSE/valid_count/ADE/FDE/key。先trajectory内window指标均值，再跨trajectory等权为主macro；另列全window等权micro及全部有效点加权MSE/ADE，名称不混淆。列短尾/coverage；无可用窗口的trajectory单列，不从24覆盖声明中消失。
- 若P2生成兼容闭环任务，两模型必须执行全部：排序key/segment冻结，seed=10000+task_index*100+replan；同控制参数。SR/CR/SPL/Goal Progress/Stuck/Route-Failure报segment micro、trajectory macro和逐trajectory。D024保留分母不碰撞；invalid_map一致排除，公开名单/原始分母。确无兼容任务则标闭环未完成/不适用及原因，不能离线冒充闭环。
- 输出`eval/<method>/offline/{per_window.csv,per_trajectory.csv,summary.json}`及`navigation/{per_segment.csv,per_trajectory.csv,summary.json}`；共同summary.json/report.md列A/B/B−A、pair覆盖/λ、额外denoising与辅助计算、wall time/显存/推理时延、停止/best updates。
- 完成3.3规定的冻结物理读出/运动诊断，不反向影响选模。一次结果检查：明细重算macro/micro/D024，核对两个checkpoint/同manifest/全24覆盖/选模来源；不重训多seed确认。真实评价bug可用同checkpoint重算并记录替代来源，不依据ANYmal结果调模型。
- 实验完成：训练2/2、offline2/2且全24覆盖；确有数据原因无法覆盖时标PARTIAL并列限制，不冒充COMPLETE；有兼容任务则navigation2/2；报告/配置/命令/checkpoint/核心明细齐全，缺项明确，更新计划/PROJECT_STATUS/ARCHITECTURE/HANDOFF。存在未完成必需项时不得标COMPLETE。

## 6. CLI、产物和资源契约

### 6.1 当前可执行CLI与续接

所有Proxy入口显式传`--profile proxy_ab`或`PROFILE=proxy_ab`；配置读取两个YAML的proxy_ab段及pair/config.yaml的proxy_gate。禁止落入transfer_primary预算/ANYmal训练或source-cache路径。CPU已完成，实际构建命令在run/command.sh，已完成cpu_resume.sh与pair_expansion_resume.sh直接退出。

```bash
# 用户要求继续GPU后，首次P5b由该脚本创建UTC GPU-smoke RUN_ID。
ssh cross-diff 'bash /tj-share/cross_diffusion_workdir/runs/20261002T045353Z_proxy_cpu/gpu_resume.sh'
```

P5b恢复必须显式读取已保存RUN_ID并设RESUME=1，不重新date。P5b通过后才创建新的正式RUN_ID；A/B分别独立从原始nuPlan初始化，使用同DATA_ID。GPU shell环境必须保留PROJECT_ROOT、OUTPUT_ROOT、PYTHON_BIN、PROFILE、PROXY_CONFIG、SOURCE_CKPT及DATA_ID，不能把smoke目录作为正式输出。

```bash
# 只在P5b通过之后首次执行；每个新shell显式设置Proxy环境：
export PROJECT_ROOT=/zeron-vepfs/tjqc/cross-diffusion
export OUTPUT_ROOT=/tj-share/cross_diffusion_workdir
export PYTHON_BIN=/root/miniconda3/envs/diffusion-planner/bin/python
export PROFILE=proxy_ab
export DATA_ID=20261002T045353Z_proxy_cpu
export PROXY_CONFIG=$PROJECT_ROOT/tartan/research_score/configs/transfer_methods.yaml
export SOURCE_CKPT=$PROJECT_ROOT/checkpoints/model.pth
cd "$PROJECT_ROOT"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_proxy_seed11"
RUN_KIND=formal DEVICE=cuda GPU_IDS=0 bash tartan/research_score/scripts/run_transfer_training.sh
# 中断恢复：保持原RUN_ID，显式RESUME=1。
# RUN_KIND=formal RESUME=1 DEVICE=cuda GPU_IDS=0 bash tartan/research_score/scripts/run_transfer_training.sh
# 两任务正式训练完成后执行最终ANYmal评价：
bash tartan/research_score/scripts/run_transfer_evaluation.sh
```

runner从DATA_ID构造manifest/cache路径，method=proxy_a/proxy_b，seed11，A拒绝pair参数，B必须传train/val pair manifest/cache。同run恢复对照完整model/optimizer/scheduler/scaler/sampler/RNG/noise/selector及冻结输入身份；不静默重新开始。离线和navigation评价共用evaluate_navigation入口，只有正式checkpoint可最终测试。新增候选复现仍使用pair.run_pair_mining --mode reconstruct并传--base-window-dir；现有冻结产物拒绝不同内容覆盖。

### 6.2 目录和配置字段

```text
/tj-share/cross_diffusion_workdir/
  data/<DATA_ID>/     trajectories/base_train/base_val/anymal_test,
                     navigation_val/navigation_anymal_test, summary/config
  cache/<DATA_ID>/    base_train/base_val/anymal_test/pair_train/pair_val.pt
  pairs/<DATA_ID>/    train/val/rejected, *_windows, gate_summary, selected_plots/
  runs/<CPU_RUN_ID>/  cpu_smoke/, data_checks.json, status.json, config.json, command.sh
  runs/<RUN_ID>/      run_id.txt,config.json,command.sh,status.json,task_matrix.json,
                     environment.json,preflight/,logs/,train/,eval/,summary.json,report.md
  results/transfer/   唯一历史结果，保留不覆盖
```

RUN_ID为UTC `YYYYMMDDTHHMMSSZ_proxy_seed11`，CPU后缀proxy_cpu，GPU smoke后缀proxy_gpu_smoke，恢复同ID；不使用本地时间、乱码/空格、v2/retry6。DATA_ID可与训练ID不同，显式记录。所有run配置至少包含profile/purpose、methods、seed、DATA_ID、全部manifest/cache/源权重/args路径，representation/frame/normalizer政策，platform表，pair gate/objective完整参数，optimizer/batch/microbatch/噪声、验证/选模/earlystop、GPU/环境、resume路径及实际命令。共享输入配置在data下冻结一份，run引用并保存实际使用值。

### 6.3 资源和完成预算

- 当前4张L20已可用；P5b及正式训练已实测每任务峰值约1.3–1.5GB显存。单卡顺序即可，2卡可并发；CPU先≤8 workers，内存预算16–32GiB，现有可见128GiB。
- cache约93.3KB/window（新增history/dt/mask/RMS不足1KB）（lane67.2KB+route24KB+target1.28KB等），7,808候选约0.73GB tensor，metadata后预留1GB。184候选最多368独有pair侧窗口约34MB；用ID去重，不复制每对张量。JSON/图/日志预留≤0.5GB，P2/P3按实际count/size更新。
- 源权重约48.5MB，完整AdamW last含恢复所需best副本粗估200MB、独立best50MB/模型，A/B约0.5–0.7GB；原子替换临时文件留约0.3GB，位于同数据盘。新增常驻估计2–3GB（含必要smoke），运行前确认至少10GiB实际可写余量，不含受保护原始数据/历史结果；不存每250步全部快照。
- GPU时间已实测，A约1791秒、B约2231秒（各5000更新，包含验证）；追加消融按实际负载记录时间。原预算估计使用P5b实测t_A_step/t_B_step/t_A_val/t_B_val、任务数和测试速度；上限估计`10000*(t_A_step+t_B_step)+40*(t_A_val+t_B_val)+两个最终test耗时`，另计一次cache构建。记录B额外forward/backward/classifier耗时与曝光，不因耗时长擅减预算。
- 本机仅代码/小配置/简洁状态，不复制数据/cache/checkpoint；root盘只环境工具。清理限明确路径/size/正式替代来源、可重建的临时物；不删原始数据/用户文件/checkpoint/唯一结果或未被替代的失败诊断，不宽泛递归删除。

## 7. 结果解释边界

主比较是同地图、单seed、Omni+Diff到ANYmal未见embodiment。B同时增加配对曝光、结构和辅助目标，无消融不能隔离各因素贡献。匹配和输出一致性不能证明因果识别/干预/真实反事实。代理能力/几何、记录路径选goal、零动态actor、空间弧长不表达时间或原地转向、简化闭环运动等限制必须进入报告。不默认多seed或消融，不测试后选阈值；无统计设计不称显著。

## 8. 当前节点与GPU续接

原正式实验COMPLETE；CPU_RUN_ID=DATA_ID=`20261002T045353Z_proxy_cpu`。第1节为唯一当前数量与验收摘要，不保留重复进度节点。P5b run=20261002T184329Z_proxy_gpu_smoke已通过，A/B各4次实际AMP更新，B完整恢复PASS；正式run=20261002T184606Z_proxy_seed11已完成P6/P7，A/B各5000更新、best均250，全部ANYmal离线/闭环覆盖24条、各296任务。

run路径`/tj-share/cross_diffusion_workdir/runs/20261002T045353Z_proxy_cpu`。cpu_report/status/config/metrics为最终CPU记录；pair_expansion_approval/checks为218/20批准和验收；source_provenance/reference_approval/time/frame evidence为来源；canonical CPU checkpoint与完整resume_checks保留。原114/3小型manifest/config/source/metrics仅作来源证据，不是当前执行输入。cleanup_record记录精确清理路径及大小。

20个validation配对仍仅来自Diff/P1003与Omni/P0005，相关窗口不等于独立样本；配对loss仅诊断，选模使用完整validation navigation SR。逐帧密集候选没有采用，禁止通过事后换split或放宽门禁凑数量。

来源证据和当前checkpoint保留。已清理被替代的缓存/恢复副本和重复日志。FSX发生过陈旧空页，artifacts通过原子发布、POSIX_FADV_DONTNEED及一次读取核对处理；不重复存储探针。保留dirty main，不新分支，不同步项目外历史。

原持久化进程PID873已退出，正式run=20261002T184606Z_proxy_seed11，已从原始nuPlan完成A/B和全部24条ANYmal最终评价/报告。当前不再启动或重跑；保留恢复命令RESUME=1 bash /tj-share/cross_diffusion_workdir/runs/20261002T184606Z_proxy_seed11/continue.sh恢复同RUN_ID。最终完成以该run/status.json、summary.json及report.md为准。

## 9. 原迁移实验状态（本轮不执行）

原实验一仍为四方法×1/10/100%×seed11共12任务，当前SR-first/update正式GPU矩阵未运行；20/50%不自动加入。results/transfer中的SPL-first/epoch历史不能改写为SR-first或Proxy结果。source短轨迹全valid与target padding mask差异仍属原协议限制；Proxy不用source cache，不能顺便重建后混比。历史仅在本机项目外独立归档，绝不恢复执行入口。

最终完成节点：`/tj-share/cross_diffusion_workdir/runs/20261002T184606Z_proxy_seed11`，status/summary为COMPLETE，report.md包含离线与闭环指标。A/B各5000更新、best250；ANYmal每方法24条轨迹、296闭环任务。宏平均SR A0.353451/B0.121843，CR A0.285444/B0.410261，SPL A0.345613/B0.100769。运行进程已退出，无需继续启动；单地图/单seed/观测配对与批准参考系限制仍适用。


## 10. 追加原因诊断（用户授权）

原因诊断已完成（2026-10-03）：RUN_ID=`20261002T203425Z_proxy_diagnosis`，DATA_ID/CPU_RUN_ID=`20261002T045353Z_proxy_cpu`。 消融只在run配置改权重：no_swap令λswap=0；no_inv令λinv=0；no_sep令λsep=0；pair_only令三项辅助权重为0；base_only再令pair_denoising_weight=0，保留随机流/前向但不产生配对监督梯度。该字段默认1，原正式B行为不变。 五项均独立原始nuPlan初始化，使用相同冻结数据/seed及SR选模规则；no_swap/no_inv/no_sep/pair_only各5000更新、best250，base_only6000更新、best3500。validation SR分别24.25%/23.58%/24.92%/22.33%/26.92%，对照A30.25%/完整B26.83%。没有追加ANYmal模型前向。删除swap不改善；仅基础监督与完整B接近（差0.08pp）；仅配对去噪较低。B shared-only推理SR0%、ID-off21.08%，显示分支依赖；validation67个起点有效且初始连通任务中32条记录路径触发既定碰撞代理，说明模仿目标与安全评价存在冲突。报告/summary/status/configs/commands/logs及五项必要last/best checkpoint在`/tj-share/cross_diffusion_workdir/runs/20261002T203425Z_proxy_diagnosis`。状态COMPLETE，pending为空，无下一训练命令；完成项不重跑。只有故障恢复才在确认无进程后显式同RUN_ID执行continue.sh。研究限制与待修订项：2条独立val trajectory、单seed；还未修订目标/代理几何、控制器朝向及shared监督，不宣称ANYmal改进。

2026-10-08原始backbone无历史对照完成：RUN_ID=`20261008T090250Z_proxy_a_no_history`，DATA_ID/CPU_RUN_ID=`20261002T045353Z_proxy_cpu`，状态COMPLETE，pending为空。复用原A冻结数据、normalizer、样本顺序/噪声随机流、seed11、原始nuPlan EMA独立初始化及训练/评价规则，仅--disable-history移除latent/RMS条件。真实4次AMP smoke、新旧checkpoint strict加载通过；正式5000更新、best250、基础曝光319408，原A样本顺序一致。全部24条ANYmal、3071离线窗口、296导航任务评价完成。

| 指标 | 原A | 无历史A |
|---|---:|---:|
| validation SR | 30.25% | 32.1667% |
| validation CR | 27.5833% | 25.6667% |
| validation SPL | 29.7442% | 31.7978% |
| ANYmal ADE/FDE(m) | 0.669893/1.501356 | 0.652690/1.450710 |
| ANYmal SR/CR/SPL | 35.3451%/28.5444%/34.5613% | 36.7433%/28.4892%/35.7453% |

无历史best验证Diff SR/CR/SPL=35/30/35%，Omni=29.3333/21.3333/28.5955%。best验证loss原A0.095083→无历史0.096257；末段训练loss0.040238→0.051718，末验证loss0.084650→0.097450。无历史第500验证SR25.6667%，原A1.3333%，早期骤降缓解，但最终best只改善1.9167pp；ANYmal改善1.3982pp。原A成功/碰撞/route_failure/timeout=103/84/102/7，无历史108/84/102/2。paired新增8个成功、丢失3个成功，净增加5个。历史分支可能影响早期优化，但单seed不能认定普遍有害，也不足以解释整体低成功率；没有接近旧ANYmal域内微调测试SR67.9783%（旧SPL优先/epoch协议及任务不同，非等价对照）。监督与碰撞代理冲突、简化朝向执行仍未修订；不追加未授权训练。

正式产物位于`/tj-share/cross_diffusion_workdir/runs/20261008T090250Z_proxy_a_no_history`：config.yaml/experiment.json/command.sh/runner.py/status.json/summary.json/report.md、logs、smoke_acceptance.json、train/proxy_a_no_history/{last.pt,navigation_best.pt}及eval。必要权重全部保留。完成分析后删除heartbeat；无下一训练命令，不重跑。


2026-10-08最新实验1对照基准复核：以用户提供2026-09-29阶段总结对应的finetune_navigation_selection_earlystop_v1为准，保留run results/transfer/20261001T154600Z_retained_spl_seed11。实际100% metrics：2720 updates/85 epochs，best1440 updates/45 epochs；val宏SR70.5177%、SPL65.0646%，35任务×seed11/23/47；按SPL优先选模，LR每10epoch按val loss(min)调整。当前无历史A为5000/best250，SR选模，LR按SR(max)，不能将差距仅归为平台数据。真实target_train_features及d029_extended缓存均2048，完整80点692、平均有效61.5493；真实最终manifest raw_points80、timestamps至8秒，旧先截时间再空间重采样的区别成立。extended只扩充预算档位，没有延长轨迹。用最终manifest首条真实样本重建旧direct-grid特征，lanes/route_lanes/target/mask与缓存完全一致。CPU source evidence确认同源occupancy为camera-local、0.2m体素、XY bounds[-25,25]；旧直接按0.5m索引解释，尺度假设不一致。新101×101@0.5m是显式转换后BEV，保留约50m尺度；不能将旧250格解释为真实125m视野。NED→NWU仅坐标约定，NED本身不是错误；当前ego-local闭环用于对齐训练，但旧fixed-frame并非已证实错误。起点纳入是表示合同选择，未证明旧下一帧错误。平台/split改变为用户目标，不作为待修复问题。未改模型、缓存、原始数据或训练结果，未启动新训练；旧高分受上述几何假设限制，未量化其影响。
