# 当前项目状态

更新：2026-10-08。当前执行四组统一普通Diffusion Planner研究；原Proxy A/B、消融和无历史A已经完成，不能视为新四组结果。

## 四组当前进度

RUN_ID=`20261008T132918Z_four_groups`；状态`GPU_RUNNING`，服务器进程PID1136。数据/缓存DATA_ID为RUN_ID加_g1…_g4；catalog DATA_ID=`20261002T045353Z_proxy_cpu`。g1 ANYmal17/2/5，g2 Omni4/1/1，g3 Diff3/1/1，g4 Diff4/1+Omni5/1→全部24条ANYmal。完整轨迹先冻结，seed20260911；普通backbone、训练seed11、原始nuPlan独立初始化，无history/RMS/adapter/pair；每5完整epoch验证，推理seed11/23/47。准确预算、几何、控制器和旧实验1差异见TRAINING_PROTOCOL开头。

已完成：现有模块接通四组配置、完整8米未来80站点、明确cache表示隔离、每5完整epoch验证与恢复边界、最终三seed评价、g4同g1测试子集汇总、g1对旧实验1比较；本机72测试通过、1因无GPU跳过，架构清单与shell/编译通过；服务器架构/入口兼容通过。原实验1服务器独立归档已发布，43个必要源码文件，复制462MiB保留结果，2048行训练membership/35任务val/63任务test输入，原件保留，无hash。

运行实况：主run说明/config/status/command/log已保存；四组真实完整8米缓存及全部preflight已通过，CPU_READY=true；g1/g2各4更新AMP smoke通过，已分别进入GPU0/1正式训练，g3/g4在第二批续接。正式训练、最终测试及结果分析尚未全部完成，不将工程验收或旧实验充作本轮成功。

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

参考点是证据支持的前相机观测点，真实base外参未知；圆形footprint和路径执行仍是代理；本轮统一用预测yaw续接参考系，已修复位移方向替代观测heading的接口不一致。原occupancy直接把0.2米索引解释成0.5米，已核实2.5倍尺度假设差异；本轮按bounds/完整姿态显式转换，101×101@0.5米仍约50米范围。新旧高分不能等价因果比较。真实记录路径与碰撞代理可能冲突，预检记录并保留任务，不放宽阈值。单地图、单训练seed和少量独立validation trajectory仍限制结论。

下一阶段：前两组正式训练/评价→后两组smoke/独立原始nuPlan正式训练/评价→g1原实验1比较/g4同5条子集→分析并同步文档/GitHub。当前run/status.json为实际状态依据，报告未完成前不标COMPLETE。已授权完整执行，不重跑旧A/B或旧消融。

2026-10-08真实预检后的最小接口修正：four_groups闭环以预测轨迹在实际执行站点的yaw更新观测参考系，允许侧移朝向与位移方向不同；proxy_ab及旧实验1保留原位移heading行为。新g1四更新AMP smoke已通过（工程产物，不作性能结果）；smoke验证时钟明确为updates2/4，正式每5完整epoch。新修正由数学侧移/转向控制器测试覆盖，不需要机体外参猜测、不改loss或监督。

当前四组实际冻结窗口与导航任务（2026-10-08 CPU完成节点）：

| 组 | train/val/test窗口 | val/test导航任务（每任务3个推理seed） |
|---|---|---|
| g1 | 2007/229/590 | 19/59 |
| g2 | 2032/301/625 | 95/160 |
| g3 | 1328/152/270 | 45/66 |
| g4 | 4255/453/2826 | 140/296 |

`cpu_report.json`记录四组PASS和原始碰撞/起点/连通计数，计数存在重叠，不能相加当作排除数。共享盘g4新清单读到过陈旧页，一次fadvise后正常140条，生产端已复用artifact原子发布/刷新；同RUN_ID恢复，无重建cache，无删checkpoint。g4确定性特征已全部复用本轮前三组并逐点核对，split按自己的manifest更新；不是模型或训练统计共享。

自动续接=`analyze-four-group-diffusion-planner-study`，正常运行静默，故障最小修复同RUN_ID，全部完成后分析g1对旧实验1以及部分重合测试轨迹、g4同g1五条子集，再更新本机/服务器文档并推送GitHub，报告完成后删除自动任务。当前不得重新启动活跃runner；必要恢复命令保持同一RUN_ID。
