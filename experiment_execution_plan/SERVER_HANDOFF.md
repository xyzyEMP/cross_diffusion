# 服务器交接

更新2026-10-09（Asia/Shanghai）。RUN_ID=`20261008T132918Z_four_groups`，阶段COMPLETE；四组训练、全部最终测试及分析已经完成，无待续训练命令。不要重启或重跑完成组、CPU缓存、旧A/B/消融/无历史A。

服务器源码`/zeron-vepfs/tjqc/cross-diffusion`，Python`/root/miniconda3/envs/diffusion-planner/bin/python`，SSH cross-diff；全部正式产物`/tj-share/cross_diffusion_workdir`。主run为runs/20261008T132918Z_four_groups，DATA_IDS加_g1…g4，catalog为20261002T045353Z_proxy_cpu。说明/config/status/命令/log、CPU报告、train/eval结果保留。

| 组 | updates/完整epoch/best update | test SR/CR/SPL |
|---|---|---|
| g1 | 5120/160/320 | 32.2955/27.0141/31.1675% |
| g2 | 5120/160/640 | 37.2396/36.4583/37.1911% |
| g3 | 5040/240/1260 | 16.3399/47.7124/16.3399% |
| g4 | 5025/75/335 | 35.6727/29.2401/35.6481% |

g4同g1五条子集SR30.4848%，未超过g1；新g1也未接近旧实验1SR67.9783%，split/监督/几何/控制器/选模预算/推理seed共同改变，不能单因素归因。后续目标需用户批准；可基于现有叠图和逐任务输出先定位几何/记录路径冲突，不直接增加预算或放宽阈值。

完整结果/分析为run/report.md、summary.json、main_table.csv、training_curves.png；逐行输出eval/g1…g4/{offline,navigation}/，唯一checkpoint为train/g1…g4/{last.pt,navigation_best.pt}。冻结配置config.yaml，说明experiment_description.md。旧实验1为archive/experiment1/{README.md,code/,results/,inputs/,stage_summary_20260929.md}，43必要旧代码和462MiB结果、2048train/35val/63test输入，原件保留，不是当前执行入口或起点。其他项目外历史不部署。

g3曾在smoke来源JSON发布读到288字节零页，一次fadvise恢复完整EMA来源；artifact修复为实际读错后一次刷新，持续错误仍失败，相关10测试通过。原runner1136退出后确认无活动任务，同RUN_ID续接1987完成g3；smoke未初始化正式模型，完成组/CPU阶段未重跑。必要故障日志和原始来源记录保留。最终汇总只读已完成输出，无新增模型前向；新摘要与原metrics一致、子集覆盖完整，曲线一次检查通过。

D024 route_failure留分母不计碰撞；ANYmal全部test禁训仍适用于旧proxy_ab，本轮g1域内训练仅four_groups例外。保护原数据/checkpoint/手工baseline_alignment.md和thesis_proposal.md，不提交或覆盖手工文件。GitHub只发布代码和简洁状态，实际发布版本见run/status.json；自动跟进交付后删除。
