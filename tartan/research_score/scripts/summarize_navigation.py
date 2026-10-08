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
  table.append({'group':group,**{k:groups[group][k] for k in ('updates','completed_epochs','best_update','best_epoch')},**{k:nav['episode_macro'][k] for k in ('sr','cr','spl')},**off['trajectory_macro']})
 frame=pd.read_csv(root/'eval'/'g4'/'navigation'/'per_segment.csv');g1_data=Path('/tj-share/cross_diffusion_workdir/data')/(root.name+'_g1');subset={json.loads(x)['trajectory_key'] for x in (g1_data/'test.jsonl').read_text().splitlines()}
 matched=frame[frame.trajectory_key.isin(subset)&frame.included_in_denominator]
 shared_test=episode_macro(matched.to_dict('records'))
 historical=Path('/tj-share/cross_diffusion_workdir/archive/experiment1/results')
 old=json.loads((historical/'eval/pretrain_finetune_100pct_seed11/summary.json').read_text());old_train=json.loads((historical/'train/pretrain_finetune_100pct_seed11/metrics.json').read_text())
 comparison={'reference':str(historical),'old_test':old['episode_macro'],'g1_minus_original_test':{k:groups['g1']['test'][k]-old['episode_macro'][k] for k in ('sr','cr','spl')},'original_best_epoch':old_train['best_navigation']['epoch'],'original_updates':old_train['target_updates'],'original_validation':{k:old_train['best_navigation']['macro_'+k] for k in ('sr','cr','spl')}}
 output={'status':'COMPLETE','RUN_ID':root.name,'groups':groups,'g4_on_g1_test_subset':shared_test,'experiment1_comparison':comparison,'limitations':['one training seed; inference seeds are not independent training repetitions','new complete-trajectory splits differ from original 16/3/5','original map-index scale differs from native occupancy scale; scores are not an isolated platform ablation','original SPL-first/loss scheduler/epoch budget differs from current SR-first/SR scheduler/update budget','observed camera reference, assumed circular footprints and simplified heading controller','D024 route failures stay in denominator']}
 pd.DataFrame(table).to_csv(root/'main_table.csv',index=False);publish(output,root/'summary.json',True)
 lines=['# Four-group results','', '| Group | Updates | Epochs | Best update | Best epoch | Test SR | Test CR | Test SPL | ADE | FDE |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for group,g in groups.items():lines.append('| '+ ' | '.join(str(v) for v in [group,g['updates'],g['completed_epochs'],g['best_update'],g['best_epoch'],g['test']['sr'],g['test']['cr'],g['test']['spl'],g['offline']['ade'],g['offline']['fde']])+' |')
 lines+=['','## Group 1 versus original experiment 1','',json.dumps(comparison,ensure_ascii=False,indent=2),'','## Group 4 on group 1 held-out ANYmal subset','',json.dumps(shared_test,indent=2),'','## Limits','']+['- '+v for v in output['limitations']]
 publish_text('\n'.join(lines)+'\n',root/'report.md');publish({'status':'COMPLETE','RUN_ID':root.name,'pending':[],'next_command':None,'report':str(root/'report.md')},root/'status.json',True)

if __name__=='__main__':
 if '--profile' in sys.argv and sys.argv[sys.argv.index('--profile')+1]=='four_groups':four_group_summary()
 elif '--profile' in sys.argv:proxy_summary()
 else:transfer_summary()
