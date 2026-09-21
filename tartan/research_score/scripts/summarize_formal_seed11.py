from pathlib import Path
import json
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path('/tj-share/cross_diffusion_workdir/research_score_v1_2/05_transfer')
train=base/'formal_seed11';ev=base/'formal_seed11_eval';rows=[];training=[]
for d in sorted(train.glob('*pct_seed11')):
 m=json.loads((d/'metrics.json').read_text());s=json.loads((ev/d.name/'summary.json').read_text())
 rows.append({'method':m['method'],'budget_pct':m['budget'],'seed':m['seed'],'episodes':s['episodes'],'SR':s['sr'],'CR':s['cr'],'SPL':s['spl'],'moving_N':s['moving']['episodes'],'moving_SR':s['moving']['sr'],'short_N':s['short']['episodes'],'short_SR':s['short']['sr'],'stuck_rate':s['stuck_rate'],'route_failure_rate':s['route_failure_rate'],'goal_progress':s['goal_progress'],'mean_inference_s':s['mean_inference_s'],'best_val_loss':m['best_validation_loss'],'best_target_update':m['best_target_update']})
 training.append({k:m.get(k,0) for k in ('method','budget','seed','target_samples','validation_samples','source_cache_samples','source_manifest_pool_samples','target_updates','source_updates','best_target_update','best_validation_loss','seconds')})
df=pd.DataFrame(rows).sort_values(['method','budget_pct']);df.to_csv(ev/'formal_main_table.csv',index=False);pd.DataFrame(training).sort_values(['method','budget']).to_csv(ev/'training_summary.csv',index=False)
cols=list(df.columns);table=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
for _,r in df.iterrows():table.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in r)+' |')
md=['# 实验一 seed 11 正式结果','', '> 本表来自有效早停训练与统一 goal-conditioned 闭环。当前只有 1 个 seed、5 条独立 ANYmal 测试 episode，属于正式流程结果，但统计说服力仍需其余 seeds 扩充。','',*table,'','核心指标优先级：SR → CR → SPL。不得仅按 validation loss 排名。','']
(ev/'formal_main_table.md').write_text('\n'.join(md))
fig,ax=plt.subplots(figsize=(8,5))
for method,g in df.groupby('method'):g=g.sort_values('budget_pct');ax.plot(g.budget_pct,g.SR,marker='o',label=method)
ax.set_xscale('log');ax.set_xticks([1,10,100],['1%','10%','100%']);ax.set_ylim(-.03,1.03);ax.set_xlabel('ANYmal target-data budget');ax.set_ylabel('Success Rate');ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ev/'target_budget_curve.png',dpi=160);plt.close(fig)
print(ev/'formal_main_table.md')
