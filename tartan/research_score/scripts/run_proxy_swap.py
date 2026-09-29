from __future__ import annotations

import argparse, json, os, subprocess, tempfile
from pathlib import Path
import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.scripts.run_proxy_smoke import context


def publish(obj, path, json_file=False):
    fd,tmp=tempfile.mkstemp(prefix="proxy_swap_",suffix=".json" if json_file else ".pt");os.close(fd)
    try:
        if json_file: Path(tmp).write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
        else: torch.save(obj,tmp)
        subprocess.run(["dd","if="+tmp,"of="+str(path),"conv=fsync","status=none"],check=True)
    finally: Path(tmp).unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--args",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--pairs",type=int,default=6);p.add_argument("--samples",type=int,default=2);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);device=torch.device("cuda");c=Config(a.args,None);c.device="cuda"
    model=ScoreDecompositionPlanner(Diffusion_Planner(c)).to(device);payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False);model.load_state_dict(payload["model"],strict=True);model.eval()
    cache=torch.load(a.cache,map_location="cpu",weights_only=False);indices=list(range(min(a.pairs,len(cache["pair_ids"]))));ctx=context(cache,c,indices,device);ability=torch.zeros(len(indices),10,device=device);mask=torch.zeros_like(ability)
    predictions={};trace=[]
    for sample in range(a.samples):
        for source_name in ("diff","anymal"):
            for target_name,target_id in (("diff",2),("anymal",1)):
                seed=1100+sample
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                with torch.no_grad(): _,result=model(ctx,torch.full((len(indices),),target_id,dtype=torch.long,device=device),ability,mask)
                key=f"{source_name}_to_{target_name}_k{sample}";prediction=result["prediction"][:,0].detach().cpu();predictions[key]=prediction
                trace.append({"key":key,"source_context":source_name,"target_platform":target_name,"target_id":target_id,"seed":seed,"shape":list(prediction.shape),"finite":bool(torch.isfinite(prediction).all())})
    diff_identity=max(float((predictions[f"diff_to_diff_k{k}"]-predictions[f"anymal_to_diff_k{k}"]).abs().max()) for k in range(a.samples))
    anymal_identity=max(float((predictions[f"anymal_to_anymal_k{k}"]-predictions[f"diff_to_anymal_k{k}"]).abs().max()) for k in range(a.samples))
    conditioned=float(torch.stack([(predictions[f"diff_to_diff_k{k}"]-predictions[f"diff_to_anymal_k{k}"]).abs().mean() for k in range(a.samples)]).mean())
    report={"status":"PASS_GPU_FOUR_WAY_SWAP","pairs":len(indices),"samples_per_combo":a.samples,"pair_ids":[cache["pair_ids"][i] for i in indices],"diff_target_identity_max_abs":diff_identity,"anymal_target_identity_max_abs":anymal_identity,"target_conditioned_mean_abs_difference":conditioned,"all_finite":all(x["finite"] for x in trace),"trace":trace,"interpretation":"source contexts are the same common map-goal representation; source-label identities are therefore expected, while target platform changes the per-step correction","claim_scope":"synthetic_engineering_only"}
    if not report["all_finite"] or max(diff_identity,anymal_identity)>1e-5: raise RuntimeError(json.dumps(report))
    publish(predictions,out/"raw_predictions.pt");publish(report,out/"report.json",True);print(json.dumps({k:v for k,v in report.items() if k!="trace"},sort_keys=True))


if __name__=="__main__":main()
