# Cross-Diffusion

当前四组统一普通Diffusion Planner实验全部完成：RUN_ID=`20261008T132918Z_four_groups`，状态`COMPLETE`。四组独立原始nuPlan初始化；完整8米、每5完整epoch验证、SR选模和三seed推理。详细结果、loss趋势和旧实验1对照见[PROJECT_STATUS.md](PROJECT_STATUS.md)，服务器完整报告为本run的report.md。

| 组 | 完整trajectory train/val/test | 最终测试 |
|---|---|---|
| g1 | ANYmal17/2/5 | 5条ANYmal |
| g2 | Omni4/1/1 | 1条Omni |
| g3 | Diff3/1/1 | 1条Diff |
| g4 | Diff4/1、Omni5/1 | 全部24条ANYmal |

四组独立原始nuPlan初始化，无history/RMS/adapter/pair；真实完整8米，80未来站点0.1…8.0米。每5完整epoch验证，使用预测yaw续接观测参考系；共同SR选模/5000–10000update预算。完整说明与最新原实验1差异见[TRAINING_PROTOCOL.md](TRAINING_PROTOCOL.md)开头。

- [PROJECT_STATUS.md](PROJECT_STATUS.md)：当前阶段、实际产物与已完成历史指标。
- [唯一计划](experiment_execution_plan/00_overall_progress.md)及[服务器交接](experiment_execution_plan/SERVER_HANDOFF.md)：接口、实际状态和同RUN_ID恢复。
- [ARCHITECTURE.md](ARCHITECTURE.md)：正式模块职责和文件清单。
- [proxy-training-protocol.md](proxy-training-protocol.md)：已完成旧Proxy A/B协议，不是本轮四组结果；旧profile仍ANYmal禁训。

服务器源码`/zeron-vepfs/tjqc/cross-diffusion`，Python`/root/miniconda3/envs/diffusion-planner/bin/python`，输出`/tj-share/cross_diffusion_workdir`。本轮产物为`runs/20261008T132918Z_four_groups`，说明为`experiment_description.md`，CPU验收为`cpu_report.json`，实际状态以`status.json`为准。原实验1必要代码、最终结果及精确差异说明在输出盘`archive/experiment1/`，仅历史参考；其他项目外归档不部署。GitHub保存当前源码、协议和状态，数据/cache/checkpoint/完整结果留服务器。

故障恢复先确认无活动runner，再显式同RUN_ID执行既有run/continue.sh。当前run已完成，无需重启；后续研究需另行冻结合同。原A/B、validation消融和无历史A已完成，指标/产物链接见PROJECT_STATUS，不能作为新四组完成证据。安装依赖见requirements_torch.txt，使用pip install -e .；架构检查python scripts/check_architecture.py。保护手工文件，不新建分支或平行入口。

最终test trajectory宏SR：g1 32.2955%、g2 37.2396%、g3 16.3399%、g4全24条35.6727%；g4与g1相同五条仅30.4848%，未超过g1。新g1未接近最新原实验1，split/几何/任务/选模等共同变化，不能单因素归因。数据/checkpoint/完整结果保留输出盘，GitHub只发布源码和简洁说明。
