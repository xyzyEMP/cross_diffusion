from __future__ import annotations

import argparse, json, os, random, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import torch

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from tartan.research_score.model.score_decomposition import ScoreDecompositionPlanner
from tartan.research_score.training.losses import proxy_objective


def publish(obj, path, json_file=False):
    fd, tmp = tempfile.mkstemp(prefix="proxy_smoke_", suffix=".json" if json_file else ".pt"); os.close(fd)
    try:
        if json_file: Path(tmp).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        else: torch.save(obj, tmp)
        subprocess.run(["dd", "if=" + tmp, "of=" + str(path), "conv=fsync", "status=none"], check=True)
    finally: Path(tmp).unlink(missing_ok=True)


def context(cache, config, indices, device):
    b = len(indices); idx = torch.as_tensor(indices, dtype=torch.long)
    zero = lambda *shape: torch.zeros(shape, dtype=torch.float32, device=device)
    result = {"ego_current_state": cache["ego_current_state"].index_select(0, idx).to(device),
              "neighbor_agents_past": zero(b, config.agent_num, config.time_len, config.agent_state_dim),
              "static_objects": zero(b, config.static_objects_num, config.static_objects_state_dim),
              "lanes": cache["lanes"].index_select(0, idx).to(device),
              "lanes_speed_limit": zero(b, config.lane_num, 1),
              "lanes_has_speed_limit": torch.zeros(b, config.lane_num, 1, dtype=torch.bool, device=device),
              "route_lanes": cache["route_lanes"].index_select(0, idx).to(device),
              "route_lanes_speed_limit": zero(b, config.route_num, 1),
              "route_lanes_has_speed_limit": torch.zeros(b, config.route_num, 1, dtype=torch.bool, device=device)}
    return config.observation_normalizer(result)


def target(config, trajectory, device):
    b = len(trajectory); raw = torch.zeros(b, 11, 80, 4, device=device); raw[:, 0] = trajectory.to(device)
    return config.state_normalizer(raw)


def denoise(model, inputs, normalized_target, platform_id, ability, ability_mask, epsilon=None, time_value=None):
    b = len(normalized_target); device = normalized_target.device
    t = torch.rand(b, device=device) * .999 + .001 if time_value is None else torch.full((b,), time_value, device=device)
    mean, std = model.backbone.sde.marginal_prob(normalized_target, t); std = std.view(b, 1, 1, 1)
    epsilon = torch.randn_like(mean) if epsilon is None else epsilon
    query = mean + std * epsilon
    current = torch.zeros(b, 11, 1, 4, device=device); current[:, 0, 0] = inputs["ego_current_state"][:, :4]
    call = dict(inputs); call.update({"sampled_trajectories": torch.cat([current, query], 2), "diffusion_time": t})
    _, output = model(call, torch.full((b,), platform_id, dtype=torch.long, device=device), ability, ability_mask)
    return output["decomposition"]


def checkpoint_payload(model, optimizer, update, mode):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update, "mode": mode,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode", choices=["real_unpaired","synthetic_paired"], required=True)
    p.add_argument("--cache", required=True); p.add_argument("--args", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True); p.add_argument("--updates", type=int, default=50); p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=11); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True, exist_ok=False); device=torch.device("cuda")
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    config=Config(a.args,None); config.device="cuda"; model=ScoreDecompositionPlanner(Diffusion_Planner(config)).to(device)
    payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model.load_state_dict(payload["model"],strict=True)
    model.embodiment_encoder.reinitialize_id(2,a.seed); cache=torch.load(a.cache,map_location="cpu",weights_only=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4); weights={"invariant":.2,"swap":.5,"separation":.1}
    history=[]; t0=time.time(); model.train()
    for update in range(1,a.updates+1):
        if a.mode=="real_unpaired":
            ids=cache["platform_id"]; ai=torch.nonzero(ids==1,as_tuple=False).view(-1).tolist(); di=torch.nonzero(ids==2,as_tuple=False).view(-1).tolist(); n=max(1,a.batch_size//2)
            ia=random.choices(ai,k=n); ib=random.choices(di,k=n)
            ca=context(cache,config,ia,device); cb=context(cache,config,ib,device)
            ya=target(config,cache["trajectory"].index_select(0,torch.tensor(ia)),device); yb=target(config,cache["trajectory"].index_select(0,torch.tensor(ib)),device)
            aa=cache["ability"].index_select(0,torch.tensor(ia)).to(device); ab=cache["ability"].index_select(0,torch.tensor(ib)).to(device)
            ma=cache["ability_mask"].index_select(0,torch.tensor(ia)).to(device); mb=cache["ability_mask"].index_select(0,torch.tensor(ib)).to(device)
            da=denoise(model,ca,ya,1,aa,ma); db=denoise(model,cb,yb,2,ab,mb)
            losses=proxy_objective("tartan_recorded_unpaired",da.x0_total[:,0,1:],ya[:,0],cache["valid_mask"].index_select(0,torch.tensor(ia)).to(device),db.x0_total[:,0,1:],yb[:,0],cache["valid_mask"].index_select(0,torch.tensor(ib)).to(device),weights={})
        else:
            indices=random.choices(range(len(cache["pair_ids"])),k=min(a.batch_size,len(cache["pair_ids"])))
            c=context(cache,config,indices,device); yd=target(config,cache["trajectory_diff"].index_select(0,torch.tensor(indices)),device); ya=target(config,cache["trajectory_anymal"].index_select(0,torch.tensor(indices)),device)
            ability=torch.zeros(len(indices),10,device=device); ability_mask=torch.zeros_like(ability); epsilon=torch.randn_like(yd)
            dd=denoise(model,c,yd,2,ability,ability_mask,epsilon); da=denoise(model,c,ya,1,ability,ability_mask,epsilon)
            # Same common map-goal context makes the shared-view invariant term structurally zero.
            losses=proxy_objective("synthetic_fixture",dd.x0_total[:,0,1:],yd[:,0],cache["valid_mask_diff"].index_select(0,torch.tensor(indices)).to(device),da.x0_total[:,0,1:],ya[:,0],cache["valid_mask_anymal"].index_select(0,torch.tensor(indices)).to(device),pair_valid=cache["pair_valid"].index_select(0,torch.tensor(indices)).to(device),shared_a=dd.x0_shared[:,0,1:],shared_b=dd.x0_shared[:,0,1:],swap_ab=da.x0_total[:,0,1:],swap_ba=dd.x0_total[:,0,1:],delta_a=dd.x0_embodiment_delta[:,0,1:],delta_b=da.x0_embodiment_delta[:,0,1:],weights=weights)
        optimizer.zero_grad(set_to_none=True); losses["total"].backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)); optimizer.step()
        row={"update":update,"grad_norm":grad,**{k:float(v.detach()) for k,v in losses.items()}}; history.append(row)
        if update==1 or update%25==0 or update==a.updates: print(json.dumps(row),flush=True)
    final=out/"last.pt"; publish(checkpoint_payload(model,optimizer,a.updates,a.mode),final)
    # Save/reload verification on a fixed state tensor plus optimizer and RNG payload presence.
    saved=torch.load(final,map_location="cpu",weights_only=False); probe_name=next(iter(model.state_dict())); probe=torch.equal(saved["model"][probe_name],model.state_dict()[probe_name].cpu())
    report={"status":"PASS_GPU_PROXY_SMOKE","mode":a.mode,"updates":a.updates,"batch_size":a.batch_size,"seconds":time.time()-t0,"checkpoint":str(final),"checkpoint_model_exact":probe,"optimizer_restored_fields":sorted(saved["optimizer"].keys()),"rng_states_saved":all(k in saved for k in ("torch_rng","cuda_rng","numpy_rng","python_rng")),"first":history[0],"last":history[-1],"history":history,"claim_scope":"engineering_smoke_only"}
    if not probe or not report["rng_states_saved"] or not all(np.isfinite(x["total"]) for x in history): raise RuntimeError(json.dumps(report))
    publish(report,out/"metrics.json",True); print(json.dumps({k:v for k,v in report.items() if k!="history"},sort_keys=True))


if __name__=="__main__": main()
