# 服务器交接

当前RUN_ID=`20261008T132918Z_four_groups`，阶段GPU_RUNNING，进程PID1136；已启动同RUN_ID流程，无需重复启动。完整设置和旧实验1区别见TRAINING_PROTOCOL开头；实际进度见run/status.json及logs/*_prepare.log。四组真实cache/preflight全部PASS；g1/g2四更新AMP smoke通过并正式训练中，g3/g4随后执行。CPU验收汇总见run/cpu_report.json；正式结果尚未完成。

服务器源码/zeron-vepfs/tjqc/cross-diffusion，Python/root/miniconda3/envs/diffusion-planner/bin/python，SSH cross-diff，全部产物/tj-share/cross_diffusion_workdir。主run=runs/20261008T132918Z_four_groups；DATA_ID分别为该ID加_g1…_g4。配置冻结在run/config.yaml，说明为run/experiment_description.md；来源catalog=data/20261002T045353Z_proxy_cpu/trajectories.jsonl，原始nuPlan来源record仍在对应CPU run/source_provenance.json。

流程：四组完整trajectory冻结→真实完整8米/0.1…8m80未来站点cache→一次val几何检查→各组4update独立AMP smoke→普通无历史backbone独立原始nuPlan初始化→每5完整epoch验证/SR选模→各组3seed离线和导航→g1与原实验1对比/g4同g1五条子集→最终报告。2张L20，g1+g2、g3+g4两批，每卡一个任务。旧proxy_ab仍ANYmal禁训；仅four_groups g1允许ANYmal70/10/20。严禁以smoke或另一组checkpoint初始化。

失败时先确认无活动训练/runner进程，读取短日志并修复，再恢复同RUN_ID：

```bash
cd /zeron-vepfs/tjqc/cross-diffusion
PROFILE=four_groups RUN_ID=20261008T132918Z_four_groups bash tartan/research_score/scripts/run_transfer_training.sh
```

runner跳过完成组，从本组last完整恢复optimizer/scaler/scheduler/sampler/RNG；部分评价可同目录重算，保留冻结配置和唯一结果，不新建retry目录。完成后读取summary.json/report.md/main_table.csv，并更新唯一计划、PROJECT_STATUS、协议及GitHub；自动跟进在完成报告后删除，不重复通知。

历史来源：archive/experiment1/{README.md,code/,results/,inputs/,stage_summary_20260929.md,source_record.json,archive_record.json}已发布，43必要源码、462MiB结果、训练membership2048行/val35/test63。原件results/transfer/20261001T154600Z_retained_spl_seed11保留。该目录仅用户授权历史参考，不加入当前源码入口/训练起点。其他归档留本机项目之外。

已完成旧结果不重跑：原A/B runs/20261002T184606Z_proxy_seed11；validation消融 runs/20261002T203425Z_proxy_diagnosis；无历史A runs/20261008T090250Z_proxy_a_no_history。指标见PROJECT_STATUS/各run报告。D024 route_failure保留分母、不计碰撞；当前观测参考点/假定footprint/简化heading/单seed限制不伪造解除。保护手工未提交文档，不同步项目外其他历史，不宽泛删除或生成hash。

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
