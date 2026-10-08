from pathlib import Path
import argparse,json,pandas as pd
import sys

def transfer_summary():
 p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--eval',required=True);a=p.parse_args()
 train=Path(a.train);ev=Path(a.eval);rows=[]
 for d in sorted(ev.glob('*pct_seed11')):
  s=json.loads((d/'summary.json').read_text());m=json.loads((train/d.name/'metrics.json').read_text());e=s['episode_macro']
  rows.append({'method':m['method'],'budget_pct':m['budget'],'train_windows':m['target_samples'],'segments':s['episodes'],'SR':s['sr'],'CR':s['cr'],'SPL':s['spl'],'episode_macro_SR':e['sr'],'episode_macro_CR':e['cr'],'episode_macro_SPL':e['spl'],'route_failure_rate':s['route_failure_rate'],'goal_progress':s['goal_progress'],'diagnostic_min_val_loss':m['best_validation_loss'],'selected_validation_macro_SR':m['best_navigation']['sr'],'best_target_update':m['best_target_update']})
 df=pd.DataFrame(rows).sort_values(['method','budget_pct']);df.to_csv(ev/'formal_main_table.csv',index=False)
 cols=list(df.columns);table=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
 for _,r in df.iterrows():table.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in r)+' |')
 (ev/'formal_main_table.md').write_text('\n'.join(['# Non-overlapping 8 m closed-loop results','',*table,'']))
 print(ev/'formal_main_table.md')

def proxy_summary():
 from tartan.research_score.artifacts import publish, publish_text
 p=argparse.ArgumentParser();p.add_argument('--profile',choices=['proxy_ab'],required=True);p.add_argument('--run-root',required=True);p.add_argument('--task-matrix',required=True);a=p.parse_args();root=Path(a.run_root);matrix=json.loads(Path(a.task_matrix).read_text())
 if matrix['RUN_KIND']!='formal' or {t['method'] for t in matrix['tasks']}!={'proxy_a','proxy_b'}:raise ValueError('formal A/B task matrix required')
 data=Path('/tj-share/cross_diffusion_workdir/data')/matrix['DATA_ID'];expected={r['trajectory_key'] for line in (data/'trajectories.jsonl').open() if line.strip() for r in [json.loads(line)] if r['embodiment']=='anymal'}
 summaries={};limitations=[];configs={}
 for task in matrix['tasks']:
  method=task['method'];train=Path(task['directory']);metrics=json.loads((train/'metrics.json').read_text());configs[method]=json.loads((train/'config.json').read_text())
  if metrics['status']!='complete' or not metrics['eligible_for_formal_result'] or not all((train/name).is_file() for name in ('last.pt','navigation_best.pt','config.json','command.sh')):raise ValueError('incomplete formal training '+method)
  offline=json.loads((root/'eval'/method/'offline'/'summary.json').read_text());missing=expected-set(offline['covered_trajectories']);navpath=root/'eval'/method/'navigation'/'summary.json';nav=json.loads(navpath.read_text()) if navpath.exists() else None
  if missing:limitations.append(method+' missing ANYmal trajectories: '+','.join(sorted(missing)))
  if nav is None:limitations.append(method+' closed-loop evaluation unavailable')
  else:
   nav_keys=set(nav['raw_trajectory_coverage'])
   if nav['status']!='complete':limitations.append(method+' navigation has no valid tasks')
   if expected-nav_keys:limitations.append(method+' navigation coverage missing: '+','.join(sorted(expected-nav_keys)))
  summaries[method]={'training':metrics,'offline':offline,'navigation':nav,'missing_trajectories':sorted(missing)}
 shared=('train_cache','val_cache','val_navigation_manifest','args','checkpoint','seed','device','smoke','amp')
 if any(configs['proxy_a']['cli'][k]!=configs['proxy_b']['cli'][k] for k in shared):raise ValueError('A/B frozen common inputs differ')
 if configs['proxy_a']['proxy_ab']!=configs['proxy_b']['proxy_ab']:raise ValueError('A/B frozen settings differ')
 a_metrics=summaries['proxy_a']['offline']['trajectory_macro'];b_metrics=summaries['proxy_b']['offline']['trajectory_macro'];delta={k:b_metrics[k]-a_metrics[k] for k in a_metrics}
 pairgate=Path('/tj-share/cross_diffusion_workdir/pairs')/matrix['DATA_ID']/'gate_summary.json'
 output={'status':'PARTIAL' if limitations else 'COMPLETE','profile':'proxy_ab','DATA_ID':matrix['DATA_ID'],'methods':summaries,'offline_B_minus_A':delta,'pair_gate':json.loads(pairgate.read_text()),'limitations':limitations,'scientific_limits':['single seed and map','observational matched pairs, no causal identification','B adds exposure, structure and auxiliary computation','recorded-path goals and simplified kinematics','spatial stations do not express time or stationary turns','historical context is frozen during replanning']}
 publish(output,root/'summary.json',True)
 report=['# Proxy A/B results','',f"Status: {output['status']}",'','| Method | Updates | Base exposures | Pair-side exposures | ADE macro | FDE macro | MSE macro |','|---|---:|---:|---:|---:|---:|---:|']
 for method,value in summaries.items():
  tr=value['training'];off=value['offline']['trajectory_macro'];report.append(f"| {method} | {tr['updates']} | {tr['base_exposures']} | {tr['pair_exposures']} | {off['ade']} | {off['fde']} | {off['mse']} |")
 report+=['','Selected checkpoints: '+json.dumps({m:v['training']['best_update'] for m,v in summaries.items()}),'','| Method | ANYmal trajectories | Navigation tasks | SR macro | CR macro | SPL macro | Route failure macro |','|---|---:|---:|---:|---:|---:|---:|']
 for method,value in summaries.items():
  nav=value['navigation']
  if nav is not None:
   macro=nav['episode_macro'];report.append(f"| {method} | {len(nav['raw_trajectory_coverage'])} | {nav['episodes']} | {macro['sr']} | {macro['cr']} | {macro['spl']} | {macro['route_failure_rate']} |")
 report+=['','B minus A (offline trajectory macro): '+json.dumps(delta),'','Limitations:']+['- '+x for x in limitations+output['scientific_limits']]
 publish_text('\n'.join(report)+'\n',root/'report.md')

def four_group_summary():
 from tartan.research_score.artifacts import publish,publish_text
 from tartan.research_score.scripts.evaluate_navigation import episode_macro
 p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--run-root',required=True);a=p.parse_args();root=Path(a.run_root);groups={};table=[]
 for group in ('g1','g2','g3','g4'):
  metrics=json.loads((root/'train'/group/'metrics.json').read_text());nav=json.loads((root/'eval'/group/'navigation'/'summary.json').read_text());off=json.loads((root/'eval'/group/'offline'/'summary.json').read_text())
  if metrics['status']!='complete' or nav['status']!='complete' or off['uncovered_trajectories']:raise ValueError('incomplete group '+group)
  best=next(r for r in metrics['history'] if r['target_updates']==metrics['best_update']);losses=[r['losses']['base'] for r in metrics['history']]
  frozen=json.loads((root/'train'/group/'config.json').read_text());data=Path('/tj-share/cross_diffusion_workdir/data')/frozen['proxy_ab']['paths']['data_id']
  declared={r['trajectory_key'] for line in (data/'trajectories.jsonl').read_text().splitlines() for r in [json.loads(line)] if r['split']=='test'}
  if declared!=set(nav['raw_trajectory_coverage']) or declared!=set(off['covered_trajectories']):raise ValueError('incomplete final test trajectory coverage '+group)
  groups[group]={'updates':metrics['updates'],'completed_epochs':metrics['completed_epochs'],'best_update':metrics['best_update'],'best_epoch':best['epoch'],'train_loss_first250':sum(losses[:250])/len(losses[:250]),'train_loss_last250':sum(losses[-250:])/len(losses[-250:]),'validation':best['navigation'],'validation_loss':best['val_denoising']['mse'],'test':nav['episode_macro'],'offline':off['trajectory_macro'],'test_trajectory_count':len(declared),'test_rollouts':nav['segments_selected']}
  vals=[r for r in metrics['history'] if 'val_denoising' in r]
  groups[group].update(window_counts=json.loads((data/'window_summary.json').read_text())['counts'],base_exposures=metrics['base_exposures'],stop_reason=metrics['stop_reason'],platform_validation=best['platform_navigation'],validation_nodes=len(vals),first_validation_loss=vals[0]['val_denoising']['mse'],last_validation_loss=vals[-1]['val_denoising']['mse'],min_validation_loss=min(r['val_denoising']['mse'] for r in vals),last_validation_sr=vals[-1]['navigation']['sr'],test_included_rollouts=nav['included_segments'],test_excluded_invalid_map=len(nav['excluded_invalid_map_ids']))
  table.append({'group':group,**{k:groups[group][k] for k in ('updates','completed_epochs','best_update','best_epoch')},**{k:nav['episode_macro'][k] for k in ('sr','cr','spl')},**off['trajectory_macro']})
 frame=pd.read_csv(root/'eval'/'g4'/'navigation'/'per_segment.csv');g1_data=Path('/tj-share/cross_diffusion_workdir/data')/(root.name+'_g1');subset={json.loads(x)['trajectory_key'] for x in (g1_data/'test.jsonl').read_text().splitlines()}
 matched=frame[frame.trajectory_key.isin(subset)&frame.included_in_denominator]
 shared_test=episode_macro(matched.to_dict('records'))
 offline_frame=pd.read_csv(root/'eval'/'g4'/'offline'/'per_trajectory.csv')
 offline_subset=offline_frame[offline_frame.trajectory_key.isin(subset)]
 if set(offline_subset.trajectory_key)!=subset or len(offline_subset)!=len(subset):raise ValueError('incomplete g4 offline comparison subset')
 shared_offline={k:float(offline_subset[k].mean()) for k in ('mse','ade','fde')}
 historical=Path('/tj-share/cross_diffusion_workdir/archive/experiment1/results')
 old=json.loads((historical/'eval/pretrain_finetune_100pct_seed11/summary.json').read_text());old_train=json.loads((historical/'train/pretrain_finetune_100pct_seed11/metrics.json').read_text())
 old_scenes={k.split(':')[-1]:v for k,v in old['episode_macro']['per_episode'].items()};new_scenes={k.split('/')[-1]:v for k,v in groups['g1']['test']['per_episode'].items()}
 comparison={'common_test_scenes':{k:{'original':old_scenes[k],'g1':new_scenes[k]} for k in sorted(set(old_scenes)&set(new_scenes))},'original_test_scenes':sorted(old_scenes),'g1_test_scenes':sorted(new_scenes),'comparison_scope':'descriptive; geometry/tasks/split/selection/inference seeds differ','reference':str(historical),'old_test':old['episode_macro'],'g1_minus_original_test':{k:groups['g1']['test'][k]-old['episode_macro'][k] for k in ('sr','cr','spl')},'original_best_epoch':old_train['best_navigation']['epoch'],'original_updates':old_train['target_updates'],'original_validation':{k:old_train['best_navigation']['macro_'+k] for k in ('sr','cr','spl')}}
 common=comparison['common_test_scenes'];comparison['common_scene_macro']={side:{k:sum(v[side][k] for v in common.values())/len(common) for k in ('sr','cr','spl','route_failure_rate')} for side in ('original','g1')}
 output={'status':'COMPLETE','RUN_ID':root.name,'groups':groups,'g4_on_g1_test_subset':shared_test,'experiment1_comparison':comparison,'limitations':['one training seed; inference seeds are not independent training repetitions','new complete-trajectory splits differ from original 16/3/5','original map-index scale differs from native occupancy scale; scores are not an isolated platform ablation','original SPL-first/loss scheduler/epoch budget differs from current SR-first/SR scheduler/update budget','observed camera reference and assumed circular footprints; current predicted-yaw controller differs from historical motion-heading controller','D024 route failures stay in denominator']}
 output['g4_offline_on_g1_test_subset']=shared_offline
 pd.DataFrame(table).to_csv(root/'main_table.csv',index=False);publish(output,root/'summary.json',True)
 lines=['# 四组统一 Diffusion Planner 最终报告','',f'RUN_ID: `{root.name}`。四组训练与离线/闭环评价全部完成。训练seed11，最终推理11/23/47，使用各组navigation_best.pt；下列SR/CR/SPL均为完整trajectory宏平均，百分数；ADE/FDE单位米。','', '| 组 | train/val/test窗口 | 更新/完整epoch | best更新/epoch | Test SR | CR | SPL | ADE | FDE |','|---|---|---|---|---:|---:|---:|---:|---:|']
 for group,g in groups.items():
  w=g['window_counts'];t=g['test'];o=g['offline'];lines.append(f"| {group} | {w['train']}/{w['val']}/{w['test']} | {g['updates']}/{g['completed_epochs']} | {g['best_update']}/{g['best_epoch']} | {100*t['sr']:.4f} | {100*t['cr']:.4f} | {100*t['spl']:.4f} | {o['ade']:.6f} | {o['fde']:.6f} |")
 lines+=['','g1: ANYmal→ANYmal，17/2/5；g2: Omni→Omni，4/1/1；g3: Diff→Diff，3/1/1；g4: Diff4/1+Omni5/1→全部24条ANYmal。全主干、独立原始nuPlan EMA初始化，无history/RMS/pair/adapter；完整8米80未来站点，固定源normalizer。','', '## 验证、拟合与训练预算','', '| 组/val平台 | best SR | CR | SPL |','|---|---:|---:|---:|']
 for group,g in groups.items():
  for platform,v in g['platform_validation'].items():lines.append(f"| {group}/{platform} | {100*v['sr']:.4f} | {100*v['cr']:.4f} | {100*v['spl']:.4f} |")
 lines+=['',f"g4选模对Diff/Omni等权，best SR{100*groups['g4']['validation']['sr']:.4f}%。正式每5完整epoch验证，SR→SPL→低CR→progress；5000–10000有效更新、10次SR无提升早停，四组均因sr_patience停止。best远早于最终更新，后续训练没有替换最佳导航模型。",'', '| 组 | 前250/后250训练loss均值 | 首次/best/最终val去噪MSE | 最低val MSE | base曝光/验证次数 |','|---|---|---|---:|---|']
 for group,g in groups.items():lines.append(f"| {group} | {g['train_loss_first250']:.6f}/{g['train_loss_last250']:.6f} | {g['first_validation_loss']:.6f}/{g['validation_loss']:.6f}/{g['last_validation_loss']:.6f} | {g['min_validation_loss']:.6f} | {g['base_exposures']}/{g['validation_nodes']} |")
 lines+=['','训练loss均下降，表明训练目标在拟合；不能将低SR直接解释为完全没学到。g1末期验证loss高于早期最佳，支持泛化退化/过拟合可能，但去噪loss不等于导航成功率。共同更新预算并不等于共同epoch数或相同数据量；各组独立选模。归一化、mask/有效监督长度不同，旧实验loss不能直接作为新loss的好坏阈值。','', '## 评价分母与几何限制','', '| 组 | 导航raw/included/invalid-map剔除 | test trajectory数 | route_failure宏比例 |','|---|---|---:|---:|']
 for group,g in groups.items():lines.append(f"| {group} | {g['test_rollouts']}/{g['test_included_rollouts']}/{g['test_excluded_invalid_map']} | {g['test_trajectory_count']} | {100*g['test']['route_failure_rate']:.4f}% |")
 lines+=['','raw为冻结任务数×3推理seed。invalid-map按共同协议剔除并保留逐行来源；D024的astar_disconnected留在SR/CR/SPL分母、安全停车、不计碰撞。不能剔除route_failure来改善主指标。g1和g4相同五条子集的route_failure完全相同，是共同地图/路线瓶颈；不是模型不同导致全部差距。圆形footprint、前相机观测参考点、简化控制器和记录路径goal仍是代理；预检接口通过不代表记录路径都满足碰撞代理安全约束。','', '## 第四组与第一组的相同测试集对照','', '| 模型/范围 | SR | CR | SPL | ADE | FDE |','|---|---:|---:|---:|---:|---:|']
 for label,t,o in [('g1/冻结五条',groups['g1']['test'],groups['g1']['offline']),('g4/相同五条',shared_test,shared_offline),('g4/全部24条',groups['g4']['test'],groups['g4']['offline'])]:lines.append(f"| {label} | {100*t['sr']:.4f} | {100*t['cr']:.4f} | {100*t['spl']:.4f} | {o['ade']:.6f} | {o['fde']:.6f} |")
 lines+=['',f"相同五条上g4−g1：SR {100*(shared_test['sr']-groups['g1']['test']['sr']):+.4f}pp、CR {100*(shared_test['cr']-groups['g1']['test']['cr']):+.4f}pp、SPL {100*(shared_test['spl']-groups['g1']['test']['spl']):+.4f}pp。本seed下跨平台g4未超过域内g1；g4全24条不能与g1五条直接排名，且g1没有全24条的独立测试结果。子集仅汇总已有逐行输出，没有新增ANYmal前向或调参。",'', '## 第一组与最新原实验1','', '| 项目 | 原实验1 100% SPL-first | 当前g1 |','|---|---:|---:|',f"| 更新/epoch | {comparison['original_updates']}/85 | {groups['g1']['updates']}/{groups['g1']['completed_epochs']} |",f"| best更新/epoch | 1440/45 | {groups['g1']['best_update']}/{groups['g1']['best_epoch']} |"]
 for k in ('sr','cr','spl'):lines.append(f"| val {k.upper()} | {100*comparison['original_validation'][k]:.4f}% | {100*groups['g1']['validation'][k]:.4f}% |")
 for k in ('sr','cr','spl'):lines.append(f"| test {k.upper()} | {100*comparison['old_test'][k]:.4f}% | {100*groups['g1']['test'][k]:.4f}% |")
 lines+=['', 'test描述差值（g1−旧）：'+', '.join(f"{k.upper()} {100*v:+.4f}pp" for k,v in comparison['g1_minus_original_test'].items())+'。没有接近旧实验1；不能因用户预期改善而改变评价。','', '旧test: '+', '.join(comparison['original_test_scenes'])+'；新test: '+', '.join(comparison['g1_test_scenes'])+'。仅以下三条重合，任务、几何、训练split及推理seed仍不同，属于描述对照：','', '| 共同trajectory | 旧/新SR | 旧/新route_failure |','|---|---|---|']
 for scene,v in common.items():lines.append(f"| {scene} | {100*v['original']['sr']:.4f}/{100*v['g1']['sr']:.4f}% | {100*v['original']['route_failure_rate']:.4f}/{100*v['g1']['route_failure_rate']:.4f}% |")
 lines+=['','尤其P2015路线失败从9.0909%变60%，提示环境表示/任务构造差异具有实质影响。旧按0.5米解释原生0.2米体素，新显式几何变换；两者物理原始范围均约50米，不能把旧错误尺度称真实125米。新监督读足8米并用0.1…8米80未来站点，旧先截未来80帧约8秒后填8米容器、2048窗口仅692个完整；split从16/3/5变17/2/5，选模从SPL变SR，预算/调LR和闭环预测yaw续接/推理seed也同时改变。详细逐项选择见TRAINING_PROTOCOL及archive/experiment1/README.md。没有单因素因果对照，不能据此断言某一个改动造成全部下降。','', '## 解释与下一步','', f"域内g1也低、且g4同五条没有更好，因此单纯解释为跨平台域差异或历史latent问题不足。g3碰撞率{100*groups['g3']['test']['cr']:.4f}%明显高，离线ADE{groups['g3']['offline']['ade']:.6f}米仍不能保证闭环安全；去噪目标、route输入及几何代理之间的冲突更值得优先处理。成功案例的SPL接近SR，当前主要瓶颈是失败/碰撞，而非成功后的路径效率。",'', '下一项可落地研究应先用已保存的共同场景、preflight叠图和逐任务输出定位真实占据图/相机参考点/footprint与记录路径的一致性，再冻结修正后的共同评价合同；必要时用户批准后做一项固定split的几何对照。不要先增加epoch、用测试集选模或自动放宽阈值。本轮不追加训练，不把代理校验通过当作已知真实base外参。','', '## 产物与边界','', '配置config.yaml；说明experiment_description.md；输入data/<RUN_ID>_g1…g4与cache/；训练train/g1…g4/{config.json,source_initialization.json,command.sh,last.pt,navigation_best.pt,metrics.json}；逐窗口/轨迹及逐任务结果eval/g1…g4/{offline,navigation}/；机器汇总summary.json/main_table.csv；拟合图training_curves.png。原实验1归档archive/experiment1/{code,results,inputs,README.md,stage_summary_20260929.md}，原件保留；小元数据故障诊断g3_smoke.log，后续成功smoke/正式初始化记录均保留。','', '单训练seed；三次推理不是三次独立训练。Omni/Diff各仅一条独立val和test轨迹，无法估计跨地图泛化或稳定性；同地图、相机观测参考系和圆形控制代理限制科学结论。必要数据/checkpoint/唯一结果和手工文件全部保留，无默认哈希清单。']
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figure,axes=plt.subplots(4,3,figsize=(13,11),constrained_layout=True)
 for i,group in enumerate(groups):
  history=json.loads((root/'train'/group/'metrics.json').read_text())['history'];vals=[x for x in history if 'val_denoising' in x]
  chunks=[history[j:j+250] for j in range(0,len(history),250)]
  axes[i,0].plot([c[-1]['target_updates'] for c in chunks],[sum(x['losses']['base'] for x in c)/len(c) for c in chunks]);axes[i,0].set_ylabel(group)
  axes[i,1].plot([x['target_updates'] for x in vals],[x['val_denoising']['mse'] for x in vals])
  axes[i,2].plot([x['target_updates'] for x in vals],[100*x['navigation']['sr'] for x in vals])
  for axis in axes[i]:axis.axvline(groups[group]['best_update'],color='black',linestyle='--',alpha=.5);axis.set_xlabel('Updates');axis.grid(alpha=.2)
 for axis,title in zip(axes[0],('Train loss (250-update means)','Validation denoising MSE','Validation SR (%)')):axis.set_title(title)
 figure.savefig(root/'training_curves.png',dpi=160);plt.close(figure)
 publish_text('\n'.join(lines)+'\n',root/'report.md')
 status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
 status.update(status='COMPLETE',RUN_ID=root.name,pending=[],active_groups=[],completed_groups=list(groups),blocked_groups={},next_command=None,report=str(root/'report.md'))
 publish(status,root/'status.json',True)

if __name__=='__main__':
 if '--profile' in sys.argv and sys.argv[sys.argv.index('--profile')+1]=='four_groups':four_group_summary()
 elif '--profile' in sys.argv:proxy_summary()
 else:transfer_summary()
