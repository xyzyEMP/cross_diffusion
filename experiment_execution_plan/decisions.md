---
document: decision-log
plan_version: "1.2.3"
status: ACTIVE
owner: project_lead
---

# 决策日志

| ID | 状态 | 决策 | 理由 | 影响阶段 |
|---|---|---|---|---|
| D001 | SUPERSEDED | 当前使用 Tartan 两智能体 `proxy pairs` | 已由 D020 替代：现有真实 Diff–ANYmal 不再作为正式 Proxy 配对，只保留诊断与 synthetic/interface fixture | 03–10 |
| D002 | FROZEN | Proxy 不支持正式 Car–Dog 科学结论 | 防止证据越界 | 07–10 |
| D003 | FROZEN | Strict 数据未来通过同一 adapter/schema 接入 | 避免重写模型和评测 | 03–10 |
| D004 | FROZEN | 源 checkpoint 保持 `x_start` | 当前权重原生参数化 | 02、04、05 |
| D005 | FROZEN | `L_inv/L_swap` 第一版在 x0 空间计算 | 避免 epsilon 换算的低噪声权重爆炸 | 05、08、09 |
| D006 | FROZEN | 源 observation/state normalizer 冻结 | 保持 checkpoint 语义和零残差回归 | 03–06 |
| D007 | FROZEN | 数据抽样 seed 与训练 seed 一一绑定，共 5 个联合重复 | 避免 5×5 运行数歧义 | 03、07–10 |
| D008 | FROZEN | 轨迹采用固定物理弧长并重采样为 80 点；弧长仅由 Stage 03 训练集审计从 `{8,10,12,15,20} m` 中选择满足至少 90% 有效覆盖的最大值 | 保持输出 token 数，同时消除跨平台速度尺度差异；禁止 test 调参 | 03–10 |
| D009 | FROZEN | 输入为固定目标与当前地图生成的 `route_set`，最多 6 条候选；非 oracle 路线不得读取 future GT | 支撑多路径分布与 Stage 09 mode coverage | 03、03B、07–09 |
| D010 | SUPERSEDED | Proxy 配对固定为 Tartan `diff ↔ anymal`，目标平台为 `anymal` | 已由 D020 替代：缺少共同任务与坐标标定，现有真实轨迹不可用于配对 loss | 03、05、08、09 |
| D011 | FROZEN | `pretrain_finetune` 固定为全参数微调，另设 `pretrain_adapter` | 消除“原结构或 adapter”歧义 | 06、07 |
| D012 | FROZEN | 加性双分支仅 `L_diff` 为正式主对照 | 区分结构收益与因果约束收益 | 06–08 |
| D013 | FROZEN | 实验一主迁移线为 `nuPlan Car checkpoint → TartanGround ANYmal`；Proxy 配对辅助线单独输出 | 不把 Tartan Diff 错当预训练源 | 03、05–10 |
| D014 | FROZEN | 原 checkpoint 的 8 s/80 帧时间语义只用于源回归；新增表示桥接器将轨迹转换到固定弧长/80 点空间，二者不宣称天然等价 | 防止预训练输出语义被静默改变 | 03、04、06、07 |
| D015 | FROZEN | 所有独立运行使用不可变 `run_id`；成功、失败和中断记录均不可覆盖 | 保留潜在消融和失败证据 | 01、05–10 |
| D016 | FROZEN | 正式消融候选保留 checkpoint 与逐样本结果；纯调试至少保留配置、命令、测试和失败原因 | 控制磁盘同时保证可追溯 | 01、05–10 |
| D017 | FROZEN | Stage 03 后增加无泄漏目标导航 Benchmark Gate（Stage 03B），未通过不得训练研究模型 | 先验证 SR/CR/SPL 与闭环协议 | 03B–07 |
| D018 | FROZEN | Stage 03 数据按 `moving_planning` 与 `stop_or_short` 分流；二者全部保留，前者参与固定弧长训练和覆盖率分母，后者进入独立停止/短程分支与 mask 统计 | 修复整 episode 长度替代真实 anchor/window 长度及 stationary 标签替代几何长度的问题 | 03–10 |
| D019 | FROZEN | 实验一目标域评测限定为 `same-map unseen-episode transfer`；按完整 episode 切分，禁止 trajectory/window 跨 split，不主张 unseen-map 泛化 | 当前仅有 ModularNeighborhood 一张 TartanGround 地图，不能伪造地图外测试 | 03–10 |
| D020 | FROZEN | 现有 Tartan Diff–ANYmal 真实轨迹退出正式实验一及固定弧长选择，仅保留非科学配对诊断和 synthetic/interface fixture；Strict 仍等待严格 Car–Dog 数据 | 现有真实轨迹起点最小误差 73.16m，不能构成同任务反事实 pair | 03–10 |
| D021 | FROZEN | nuPlan Car 正式源数据使用原生 Diffusion-Planner NPZ 缓存及其 100 万条训练索引；Pluto `(8,3)` trajectory cache 仅保留为无效源诊断，不进入正式 manifest、覆盖率或训练 | 原生 NPZ 的 `ego_agent_future` 为源模型实际使用的 `(80,3)`、8 s/10 Hz 轨迹，并同时保留完整 observation/map/route 张量语义 | 03–10 |
| D022 | FROZEN | 源 Car 弧长覆盖率使用训练索引中按路径 SHA256 固定抽取的 50,000 条 NPZ；以单侧 95% Wilson 下界达到 90% 判断候选合格。Tartan ANYmal 对全部 train moving windows 做精确审计 | 避免对共享盘 100 万个小文件进行无界全扫，同时用预先冻结、可复现的保守统计判据控制抽样不确定性 | 03–10 |
| D023 | FROZEN | Tartan ANYmal 仍以完整 episode 为不可分割单位，但按“是否含至少 10 个 moving window”分层后再用固定 seed/SHA256 分配 train/val/test | 修复原随机 episode split 将 8 个有效规划 episode 分成 train=6/val=2/test=0，使每个 split 都可评价 moving 和 stop，同时继续禁止 window 泄漏 | 03–07 |
| D024 | FROZEN | Stage 03B 将 `astar_disconnected` 统一处理为安全停止并记为 `route_failure`；episode保留在评测分母中，所有方法使用同一规则 | 防止静默删除困难样本或按方法差别处理，保证SR/CR/SPL可比 | 03B–07 |

### 2026-09-12 Stage 03 批准与 Stage 03B 边界

- 负责人明确批准 Stage 03 v1.2.3 并允许进入 Stage 03B。
- 负责人同时冻结 D024：4/30 `astar_disconnected` 作为 Stage 03B 显式边界，不得静默丢弃或按方法区别处理。

### 2026-09-12 Stage 03 v1.2.3 兼容修订

- 负责人已允许执行中为解决实证问题对计划做合理微调，但实验一目标不变且不得产生逻辑冲突；据此冻结 D023。
- 修订只使用轨迹的 moving/stop 标签构建分层，不读取任何模型输出、成功率或测试指标。
- split 仍按完整 episode 切分，同一 episode 的 window 不得跨 split。

### 2026-09-11 Stage 03 v1.2.2 批准记录

- 负责人明确批准 D021–D022，协议修订标识为 `score-decomp-transfer-v1.2.2`。
- 源样本集合、50,000 条固定抽样 ID、抽样算法版本与 hash 必须在查看覆盖率之前冻结；不得因审计结果追加或替换样本。
- Wilson 判据只改变源 Car 覆盖率的审计方式，不改变实验一的模型、损失、对比方法、迁移方向或科学目标。
- Tartan ANYmal 仍执行 train moving windows 全量精确覆盖统计；val/test 不参与弧长选择。
- 触发证据为 v1.2.1 remediation 中发现 Pluto cache 实际轨迹形状为 `(8,3)`；该失败 run 保留，不覆盖。
- Stage 03 v1.2.2 完成后仍须独立审查和负责人确认，不能自动进入 Stage 03B。

### 2026-09-11 Stage 03 remediation 批准记录

- 负责人明确批准 D018–D020，协议兼容修订标识为 `score-decomp-transfer-v1.2.1`。
- 触发证据：`/tj-share/cross_diffusion_workdir/research_score_v1_2/01_data_audit/stage03_index_20260911T100714Z_nogit/`。
- 旧 run 保留为 v1.2 阻塞证据，不覆盖、不删除；修订只对新的 remediation run 生效。
- 影响阶段：03、03B、04–10。进入 Stage 03B 仍需 Stage 03 remediation 独立审查和负责人再次确认。

状态：`PENDING/PROPOSED/FROZEN/SUPERSEDED`。只有负责人可以将决策改为 `FROZEN`。

新增决策必须记录日期、负责人、替代方案和被替代 ID；影响已运行结果时同步更新 [协议变更](./protocol_changelog.md)。
