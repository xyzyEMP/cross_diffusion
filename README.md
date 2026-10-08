# Cross-Diffusion

当前推进四组统一普通Diffusion Planner实验（2026-10-08），配置`four_groups`，每5个完整epoch验证。g1 ANYmal70/10/20，g2 Omni70/10/20，g3 Diff70/10/20，g4 Diff+Omni80/20训练/验证、全部24条ANYmal测试。执行合同与旧实验1精确差异见[TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)开头，实际状态见[PROJECT_STATUS.md](PROJECT_STATUS.md)。既有Proxy A/B结果保留，不是这四组的新结果。


当前执行Proxy A/B：真实Omni+Diff完整trajectory分别80/20训练/验证，全部ANYmal仅最终test；8m/80点路径，A/B共同历史16维latent与3项运动RMS及mask。B另使用同split真配对和辅助损失。

CPU已全部通过，状态CPU_READY_GPU_PENDING，DATA_ID=`20261002T045353Z_proxy_cpu`。基础窗口4280train/457val/3071test，配对218train/20val，pair缓存396/30侧窗口。P5b GPU验收已通过，正式运行20261002T184606Z_proxy_seed11已完成A/B训练、全部ANYmal评价及报告。运行实况见PROJECT_STATUS及正式run/status.json。

- [执行计划](experiment_execution_plan/00_overall_progress.md)：唯一接口和实施顺序。
- [Proxy协议](proxy-training-protocol.md)：科学边界与批准设置。
- [状态](PROJECT_STATUS.md)及[交接](experiment_execution_plan/SERVER_HANDOFF.md)：实际节点和必要续接。
- [架构](ARCHITECTURE.md)：正式文件清单及调用关系。
- [TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)：独立原迁移实验transfer_primary，不是本轮Proxy入口。

服务器源码`/zeron-vepfs/tjqc/cross-diffusion`，Python`/root/miniconda3/envs/diffusion-planner/bin/python`，输出`/tj-share/cross_diffusion_workdir`。实际CPU报告位于runs/20261002T045353Z_proxy_cpu/cpu_report.json；GPU从同run/gpu_resume.sh启动，正式A/B独立从原始nuPlan开始。来源证据及可复现记录保留，历史不作为执行规则。

安装依赖见requirements_torch.txt，使用pip install -e .。按改动运行相关测试；文件清单检查为python scripts/check_architecture.py。优先修改现有模块，不新分支或平行入口。

最终完成节点：`/tj-share/cross_diffusion_workdir/runs/20261002T184606Z_proxy_seed11`，status/summary为COMPLETE，report.md包含离线与闭环指标。A/B各5000更新、best250；ANYmal每方法24条轨迹、296闭环任务。宏平均SR A0.353451/B0.121843，CR A0.285444/B0.410261，SPL A0.345613/B0.100769。运行进程已退出，无需继续启动；单地图/单seed/观测配对与批准参考系限制仍适用。
