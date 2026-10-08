# 当前项目状态

2026-10-03（Asia/Shanghai）：Proxy A/B状态COMPLETE。正式RUN_ID=`20261002T184606Z_proxy_seed11`，DATA_ID/CPU_RUN_ID=`20261002T045353Z_proxy_cpu`，GPU smoke RUN_ID=`20261002T184329Z_proxy_gpu_smoke`。P0–P7全部完成，原正式进程873已退出，原矩阵无遗漏评价。

原因诊断已完成（2026-10-03）：RUN_ID=`20261002T203425Z_proxy_diagnosis`，DATA_ID/CPU_RUN_ID=`20261002T045353Z_proxy_cpu`。 消融只在run配置改权重：no_swap令λswap=0；no_inv令λinv=0；no_sep令λsep=0；pair_only令三项辅助权重为0；base_only再令pair_denoising_weight=0，保留随机流/前向但不产生配对监督梯度。该字段默认1，原正式B行为不变。 五项均独立原始nuPlan初始化，使用相同冻结数据/seed及SR选模规则；no_swap/no_inv/no_sep/pair_only各5000更新、best250，base_only6000更新、best3500。validation SR分别24.25%/23.58%/24.92%/22.33%/26.92%，对照A30.25%/完整B26.83%。没有追加ANYmal模型前向。删除swap不改善；仅基础监督与完整B接近（差0.08pp）；仅配对去噪较低。B shared-only推理SR0%、ID-off21.08%，显示分支依赖；validation67个起点有效且初始连通任务中32条记录路径触发既定碰撞代理，说明模仿目标与安全评价存在冲突。报告/summary/status/configs/commands/logs及五项必要last/best checkpoint在`/tj-share/cross_diffusion_workdir/runs/20261002T203425Z_proxy_diagnosis`。状态COMPLETE，pending为空，无下一训练命令；完成项不重跑。只有故障恢复才在确认无进程后显式同RUN_ID执行continue.sh。研究限制与待修订项：2条独立val trajectory、单seed；还未修订目标/代理几何、控制器朝向及shared监督，不宣称ANYmal改进。

A/B各5000更新，均按SR patience在最小预算后停止，navigation_best均选自update250。两任务独立原始nuPlan初始化，基础曝光各319408；B额外pair侧曝光320000。ANYmal离线和闭环均覆盖全部24条轨迹，每方法296个闭环任务。

| 方法 | ADE宏平均 | FDE宏平均 | SR宏平均 | CR宏平均 | SPL宏平均 |
|---|---:|---:|---:|---:|---:|
| A | 0.669893 | 1.501356 | 0.353451 | 0.285444 | 0.345613 |
| B | 0.614328 | 1.307719 | 0.121843 | 0.410261 | 0.100769 |

B离线轨迹误差降低，但闭环成功率/SPL下降、碰撞率上升；本轮不支持B整体优于A的结论。单地图/单seed、observational pair、额外曝光、固定记录路径目标、冻结历史和批准观测参考点限制仍适用。20个val pair仅来自一组相关轨迹，RMS和圆形代理不证明真实基座能力或实体安全。

正式产物：`/tj-share/cross_diffusion_workdir/runs/20261002T184606Z_proxy_seed11`的status.json、summary.json、report.md、config.json、command.sh、logs、eval、train/{proxy_a_seed11,proxy_b_seed11}/last.pt与navigation_best.pt。完整原始来源和批准记录留在CPU run；GPU验收在smoke run/gpu_acceptance.json、smoke_checks.json。原正式实验勿重跑；若确需故障恢复，显式同RUN_ID/RESUME=1。

冻结数据：完整trajectory train9/val2/ANYmaltest24；基础cache4280/457/3071；真配对218/20，pair侧cache396/30。8m/80点门禁、split、D024和原始nuPlan初始化不变。CPU69passed/1CUDA skip、增量4passed、整理相关2passed，GPU实际AMP与完整恢复通过。必要证据/数据/权重保留，清理实录在CPU run/cleanup_record.json。

源码：本机/home/yzy/文档/ChatGPT/cross_diffusion，服务器/zeron-vepfs/tjqc/cross-diffusion，SSH cross-diff，Python/root/miniconda3/envs/diffusion-planner/bin/python。dirty main保留，两地正式源码一致；历史不部署。独立transfer_primary实验未执行，不混作Proxy结果。
