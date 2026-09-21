from pathlib import Path
import json,pandas as pd
base=Path('/tj-share/cross_diffusion_workdir/research_score_v1_2/05_transfer');train=base/'formal_seed11_v3';ev=base/'formal_seed11_v3_eval_nonoverlap8m';rows=[]
for d in sorted(ev.glob('*pct_seed11')):
 s=json.loads((d/'summary.json').read_text());m=json.loads((train/d.name/'metrics.json').read_text());e=s['episode_macro']
 rows.append({'method':m['method'],'budget_pct':m['budget'],'train_windows':m['target_samples'],'segments':s['episodes'],'SR':s['sr'],'CR':s['cr'],'SPL':s['spl'],'episode_macro_SR':e['sr'],'episode_macro_CR':e['cr'],'episode_macro_SPL':e['spl'],'route_failure_rate':s['route_failure_rate'],'goal_progress':s['goal_progress'],'best_val_loss':m['best_validation_loss'],'best_target_update':m['best_target_update']})
df=pd.DataFrame(rows).sort_values(['method','budget_pct']);df.to_csv(ev/'formal_main_table.csv',index=False)
cols=list(df.columns);table=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
for _,r in df.iterrows():table.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in r)+' |')
(ev/'formal_main_table.md').write_text('\n'.join(['# v1.2.4 non-overlapping 8 m closed-loop results','',*table,'']))
print(ev/'formal_main_table.md')
