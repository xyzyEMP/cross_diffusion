# 当前项目状态

更新：2026-10-08。当前执行四组统一普通Diffusion Planner研究；原Proxy A/B、消融和无历史A已经完成，不能视为新四组结果。

## 四组当前进度

RUN_ID=`20261008T132918Z_four_groups`；状态`PREPARING_DATA`，服务器进程PID353。数据/缓存DATA_ID为RUN_ID加_g1…_g4；catalog DATA_ID=`20261002T045353Z_proxy_cpu`。g1 ANYmal17/2/5，g2 Omni4/1/1，g3 Diff3/1/1，g4 Diff4/1+Omni5/1→全部24条ANYmal。完整轨迹先冻结，seed20260911；普通backbone、训练seed11、原始nuPlan独立初始化，无history/RMS/adapter/pair；每5完整epoch验证，推理seed11/23/47。准确预算、几何、控制器和旧实验1差异见TRAINING_PROTOCOL开头。

已完成：现有模块接通四组配置、完整8米未来80站点、明确cache表示隔离、每5完整epoch验证与恢复边界、最终三seed评价、g4同g1测试子集汇总、g1对旧实验1比较；本机72测试通过、1因无GPU跳过，架构清单与shell/编译通过；服务器架构/入口兼容通过。原实验1服务器独立归档已发布，43个必要源码文件，复制462MiB保留结果，2048行训练membership/35任务val/63任务test输入，原件保留，无hash。

运行实况：主run说明/config/status/command/log已保存；g1已冻结真实完整8米窗口train2007/val229/test590，正在物化。所有组真实几何预检、GPU smoke、正式训练、最终测试及结果分析尚未完成。真实数据和训练在后台推进，通过门禁后才进入GPU；不得将代码测试或已完成旧实验充作本轮成功。

路径：主run=`/tj-share/cross_diffusion_workdir/runs/20261008T132918Z_four_groups`；说明=`experiment_description.md`；实际冻结配置=`config.yaml`；缓存和manifest在输出根cache/data相同DATA_ID。train/g1…g4保留last/navigation_best，eval/g1…g4保存offline/navigation；完成后summary.json/report.md/main_table.csv给出本轮结果和原实验1差值。恢复必须同RUN_ID，在确认原runner无活动后执行run/continue.sh，不并发重复启动。

服务器：`/zeron-vepfs/tjqc/cross-diffusion`；Python=`/root/miniconda3/envs/diffusion-planner/bin/python`；SSH=cross-diff。当前2张L20，按g1+g2和g3+g4两批，每GPU一个任务。唯一计划为experiment_execution_plan/00_overall_progress.md，交接为SERVER_HANDOFF.md，文件清单见ARCHITECTURE.md。GitHub只发布正式源码、协议和状态，数据/cache/checkpoint/完整结果仍留输出盘。

## 已完成结果与对照来源

| 实验 | 实际路径（输出根相对路径） | 已完成结果 |
|---|---|---|
| 最新原实验1 | archive/experiment1/results；原件results/transfer/20261001T154600Z_retained_spl_seed11 | ANYmal16/3/5；100%2048窗口，停止85epoch/2720update，best45epoch/1440update；val SR70.5177%、SPL65.0646%，test SR67.9783%、CR15.8766%、SPL62.3229% |
| 原Proxy A/B | runs/20261002T184606Z_proxy_seed11 | 各5000update/best250；ANYmal24条、各296任务。A SR35.3451%、CR28.5444%、SPL34.5613%；B SR12.1843%、CR41.0261%、SPL10.0769%；B离线误差较低，不等于导航更好 |
| validation消融 | runs/20261002T203425Z_proxy_diagnosis | no_swap/no_inv/no_sep/pair_only/base_only SR24.25/23.58/24.92/22.33/26.92%；无新增ANYmal模型前向 |
| 无历史A | runs/20261008T090250Z_proxy_a_no_history | 5000update/best250；val SR32.1667%；ANYmal ADE.652690/FDE1.450710、SR36.7433%、CR28.4892%、SPL35.7453%，相对原A SR+1.3982pp |

原实验1归档说明：`/tj-share/cross_diffusion_workdir/archive/experiment1/README.md`；code/仅历史参考，results/保留全部1/10/20/50/100%结果，stage_summary_20260929.md为用户最新总结，source_record.json/archive_record.json记录来源；其他项目外历史不部署。当前g1完成后必须对比上述100%结果，解释新split、实际8米监督、地图尺度、闭环frame、选模及预算共同变化，不能视为仅平台或模型变化。

## 已知限制与必要续接

参考点是证据支持的前相机观测点，真实base外参未知；圆形footprint及简化heading控制器仍是代理。原occupancy直接把0.2米索引解释成0.5米，已核实2.5倍尺度假设差异；本轮按bounds/完整姿态显式转换，101×101@0.5米仍约50米范围。新旧高分不能等价因果比较。真实记录路径与碰撞代理可能冲突，预检记录并保留任务，不放宽阈值。单地图、单训练seed和少量独立validation trajectory仍限制结论。

下一阶段：完成真实cache/preflight→各组4update AMP smoke→独立原始nuPlan正式训练→各组冻结test→g1原实验1比较/g4同5条子集→分析并同步文档/GitHub。当前run/status.json为实际状态依据，报告未完成前不标COMPLETE。已授权完整执行，不重跑旧A/B或旧消融。
