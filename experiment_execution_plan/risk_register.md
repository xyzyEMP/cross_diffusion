---
document: risk-register
plan_version: "1.2.3"
status: ACTIVE
owner: project_lead
---

# 风险登记

| ID | 等级 | 风险 | 触发信号 | 缓解措施 | 责任阶段 | 状态 |
|---|---|---|---|---|---|---|
| R001 | P0 | 固定弧长选择不满足训练覆盖 | 所有候选长度覆盖率均低于 90% | Stage 03 停止并提交数据诊断；不得查看 test 后改阈值 | 03 | MONITOR |
| R002 | P0 | Proxy 被误写成 Strict 结论 | 图表出现 Car/Dog claim 但 pair_level=proxy | Preflight 失败；Proxy/Strict 输出隔离 | 07–10 | OPEN |
| R003 | P0 | Future leakage | 输入随 future GT 改变 | 科学不变量负对照；非 oracle run 禁止继续 | 03、07 | OPEN |
| R004 | P0 | 数据域与具身身份混杂 | domain probe 很高，GRL 只识别数据集 | 对抗 loss 限匹配 pair；记录 domain_id；加入 domain probe | 03、05、08 | OPEN |
| R005 | P1 | x0→epsilon 辅助 loss 低噪声爆炸 | loss 随 t→0 急剧增大 | 第一版 x0 辅助 loss；若改 epsilon 必须 clamp/weight 消融 | 05 | MITIGATED |
| R006 | P1 | 基线不公平 | 输入/预算/controller hash 不同 | Stage 06 fairness audit | 06、07 | OPEN |
| R007 | P1 | seed 层级伪重复 | 把 5 data × 5 train 当 25 独立重复 | 采用一一绑定 5 次；层级统计 | 03、07、10 | MITIGATED |
| R008 | P1 | 源 normalizer 被重算 | 零残差输出漂移 | 冻结源 normalizer；回归测试 | 03、04 | OPEN |
| R009 | P1 | route-set 候选退化或携带 future leakage | 候选高度重合或改变 future GT 后输入改变 | 地图—目标独立构造、拓扑去重、Stage 03B 负对照 | 03、03B、09 | OPEN |
| R010 | P1 | OOM 后协议被静默修改 | batch/采样数跨方法不同 | 资源 Preflight；偏差申请；不得自动降级 | 06–09 | OPEN |
| R011 | P2 | 严格 Car–Dog 数据延迟 | CF_PAIR_ROOT 不可用 | 完成 Proxy 管线；Strict 状态保持 BLOCKED | 03、07–10 | ACCEPTED |
| R012 | P0 | 把 Diff–ANYmal 诊断当作 nuPlan→ANYmal 主迁移证据 | 主表将 Proxy fixture 与主迁移方法混合解释 | Proxy 禁止正式训练/主表；Preflight 检查 claim scope | 05–10 | MITIGATED_V1.2.1 |
| R013 | P0 | 固定弧长表示破坏源 checkpoint 语义 | 新空间零残差被错误声称为原模型复现 | 源 8 s 回归与表示桥接测试分开报告 | 03、04、06 | OPEN |
| R014 | P1 | 小实验被覆盖或清理后无法用于消融 | 相同路径重新运行、缺 run manifest | 不可变 run_id、默认拒绝 overwrite、清理需负责人批准 | 01、05–10 | OPEN |
| R015 | P0 | 在指标实现错误时开始训练 | 手工成功/碰撞轨迹无法得到预期 SR/CR/SPL | Stage 03B Benchmark Gate 先于研究训练 | 03B–07 | OPEN |

每个阶段开始和结束时更新相关风险。P0 未关闭不得 formal run。
