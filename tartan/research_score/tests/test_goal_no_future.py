import copy,numpy as np
import tartan.research_score.evaluation.goal_benchmark as gb

def test_future_mutation_invariant_and_goal_sensitive(monkeypatch):
 def fake(rec):
  g=np.asarray(rec["fixed_goal"]["xy_local"],np.float32)
  route={"route_candidates_xy":np.tile(np.linspace([0,0],g,80),(6,1,1)),"route_candidate_mask":np.ones(6,bool)}
  return route,None,g
 monkeypatch.setattr(gb,"build_from_record",fake)
 r={"fixed_goal":{"xy_local":[8.,1.]},"trajectory":{"raw_future_xy_yaw":np.zeros((80,3)).tolist()}}
 assert gb.no_future_check(r)["passed"]
 a=gb.route_digest(fake(r)[0]);z=copy.deepcopy(r);z["fixed_goal"]["xy_local"]=[2.,7.]
 assert a!=gb.route_digest(fake(z)[0])
