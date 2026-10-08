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

if __name__=='__main__':
 if '--profile' in sys.argv:proxy_summary()
 else:transfer_summary()
