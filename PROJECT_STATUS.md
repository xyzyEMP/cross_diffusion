# 当前项目状态

更新：2026-10-09（Asia/Shanghai）。四组统一普通Diffusion Planner研究已完成全部训练、离线/闭环测试及分析；RUN_ID=`20261008T132918Z_four_groups`，阶段`COMPLETE`。没有待续训练任务。原Proxy A/B、消融、无历史A属于已完成历史对照，不重跑。

## 本轮结果

四组独立原始nuPlan EMA初始化、训练seed11，无history/RMS/pair/adapter；完整8米80未来站点，原始源normalizer。完整trajectory先冻结（split seed20260911），正式每5完整epoch验证，SR优先选模，5000–10000有效更新/10次无改善早停。最终推理11/23/47，三个推理seed不是三个训练seed。数据/缓存DATA_IDS=RUN_ID加_g1…_g4，catalog DATA_ID=`20261002T045353Z_proxy_cpu`。

| 组/训测 | trajectory train/val/test | window train/val/test | 最终更新/epoch | best更新/epoch | test SR/CR/SPL | ADE/FDE（米） |
|---|---|---|---|---|---|---|
| g1 ANYmal→ANYmal | 17/2/5 | 2007/229/590 | 5120/160 | 320/10 | 32.2955/27.0141/31.1675% | 0.695163/1.492737 |
| g2 Omni→Omni | 4/1/1 | 2032/301/625 | 5120/160 | 640/20 | 37.2396/36.4583/37.1911% | 0.502557/0.959891 |
| g3 Diff→Diff | 3/1/1 | 1328/152/270 | 5040/240 | 1260/60 | 16.3399/47.7124/16.3399% | 0.526659/1.183645 |
| g4 Diff+Omni→ANYmal | Diff4/1+Omni5/1→24 | 4255/453/2826 | 5025/75 | 335/5 | 35.6727/29.2401/35.6481% | 0.693694/1.582624 |

所有指标为trajectory宏平均。g4同g1五条测试子集：SR30.4848%、CR29.9307%、SPL30.4563%、ADE0.709125米/FDE1.594194米；相对g1 SR−1.8106pp、CR+2.9167pp、SPL−0.7112pp。本seed下跨平台g4没有超过域内g1；g4全部24条与g1五条不能直接排名。

| best验证平台 | SR/CR/SPL |
|---|---|
| g1/ANYmal | 37.1795/24.3590/36.7504% |
| g2/Omni | 30.6667/23.1111/30.5162% |
| g3/Diff | 33.3333/30.8333/33.1752% |
| g4/Diff | 35.0000/30.0000/34.9797% |
| g4/Omni | 26.6667/24.0000/26.5568% |

g4选模Diff/Omni等权，best SR30.8333%。四组均因SR patience停止；前250→后250训练loss均值分别0.140658→0.067198、0.061461→0.040280、0.086681→0.060686、0.078054→0.052261，训练目标有拟合。g1验证MSE在best为0.163912、末期0.238590；g2/g3/g4末期0.056258/0.144341/0.098923，导航最佳均早于终点，更多更新没有提升最佳SR。

导航raw/included/invalid-map剔除：g1 177/177/0，g2 480/384/96，g3 198/153/45，g4 888/888/0。D024 route_failure统一留分母、不计碰撞，宏比例37.5844/23.1771/32.0261/33.9119%；g1与g4同五条该比例相同。未筛失败任务或放宽阈值。

## 第一组与最新原实验1

对照archive/experiment1/results中的100% SPL-first结果：旧停止85epoch/2720update，best45epoch/1440update；旧val SR70.5177%、CR21.1490%、SPL65.0646%；旧test SR67.9783%、CR15.8766%、SPL62.3229%。新g1 test SR−35.6828pp、CR+11.1374pp、SPL−31.1554pp，没有接近旧实验1。

新旧test各五条，仅P2003/P2012/P2015重合；共同三条描述宏SR旧51.7587%、新30.9722%，依然不是相同任务的公平单因素对照。P2015 route_failure从9.0909%变60%，显示环境表示/任务构造有实质差异。原0.2米体素按0.5解释、本轮显式几何转换；物理原始范围仍约50米，不能声称旧真实125米。完整8米监督、首点、split、闭环frame/预测yaw、SR选模/调LR、更新预算、推理seed同时改变，不能将下降归因某一项。

域内g1也低，单纯用跨平台迁移或历史latent解释不充分；Diff闭环高碰撞也不能由较低离线ADE排除。当前主要问题是共同路线/几何代理与记录路径安全目标及训练损失/闭环指标不一致，非仅训练没拟合。真实base外参仍未知，使用有证据的前相机观测参考点；圆形footprint、同地图、简化控制器、单训练seed、Omni/Diff各仅一条独立val/test限制结论。后续可先基于保存叠图/逐任务数据定位几何一致性，再批准固定split的单项几何对照；本轮不追加训练/调参。

## 实际产物与验证

正式输出根`/tj-share/cross_diffusion_workdir`，主run=`runs/20261008T132918Z_four_groups`。完整报告`report.md`，机器汇总`summary.json/main_table.csv`，训练曲线`training_curves.png`，冻结说明`experiment_description.md`、配置`config.yaml`、CPU验收`cpu_report.json`、实际命令`command.sh/continue.sh`、日志`logs/`。`train/g1…g4/`保留config/原始来源记录/last/navigation_best/metrics，`eval/g1…g4/{offline,navigation}/`保留逐窗口/轨迹及逐任务结果；输入data/和cache/均用相同DATA_IDS。数据/checkpoint不进入GitHub。

本机原完整测试72通过/1无GPU跳过；本次实际共享盘零页修复相关10测试通过。四组真实缓存、preflight、各4更新AMP smoke及正式训练/最终trajectory覆盖门禁全部通过。汇总只处理既有逐行结果，没有额外模型前向；新增字段与原metrics更新/best一致，所有离线指标有限，同五条覆盖完整；训练曲线一次视觉检查通过。必要检查未重复重跑训练或旧实验。

g3曾在来源记录发布读到288字节零页，刷新一次恢复真实EMA来源；producer改为实际不一致后一次刷新，持续不一致失败。原runner1136退出后确认无活动进程，同RUN_ID续接1987，完成第三组；smoke没有作为正式起点。既有runner恢复保留状态和冻结说明、跳过完成组；汇总保留已知状态，不覆盖配置/训练结果。诊断日志保留，必要原件/checkpoint/手工文件保护。

历史归档`archive/experiment1/{README.md,code/,results/,inputs/,stage_summary_20260929.md}`为用户授权独立参考：43个必要旧代码文件，462MiB保留结果，2048行train/35val/63test输入，原件results/transfer/20261001T154600Z_retained_spl_seed11保留；不作为当前训练起点，其他项目外历史不部署。完整设置/差异见TRAINING_PROTOCOL开头；唯一计划experiment_execution_plan/00_overall_progress.md，交接SERVER_HANDOFF.md，清单ARCHITECTURE.md。服务器源码/zeron-vepfs/tjqc/cross-diffusion，Python/root/miniconda3/envs/diffusion-planner/bin/python，SSH cross-diff。

## 已完成历史对照

| 实验/输出根相对路径 | 已完成主要结果 |
|---|---|
| 原Proxy A/B：runs/20261002T184606Z_proxy_seed11 | 各5000update/best250；ANYmal24条296任务；A SR35.3451%、CR28.5444%、SPL34.5613%；B SR12.1843%、CR41.0261%、SPL10.0769% |
| validation消融：runs/20261002T203425Z_proxy_diagnosis | no_swap/no_inv/no_sep/pair_only/base_only SR24.25/23.58/24.92/22.33/26.92%；没有新增ANYmal前向 |
| 无历史A：runs/20261008T090250Z_proxy_a_no_history | 5000update/best250；val SR32.1667%；ANYmal ADE0.652690/FDE1.450710、SR36.7433%、CR28.4892%、SPL35.7453%；相对原A SR+1.3982pp |

下一命令：无，当前运行已完成。完成节点及最终发布版本保存在run/status.json；自动跟进在交付后删除。新研究需要另行批准冻结目标，不能重启此run冒充新实验。
