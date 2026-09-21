import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
REV="score-decomp-transfer-v1.2.3";Z=1.6448536269514722
def js(p):return json.loads(p.read_text())
def rows(p):
 with p.open() as h:return [json.loads(x) for x in h if x.strip()]
def sha(p):
 d=hashlib.sha256()
 with p.open("rb") as h:
  for b in iter(lambda:h.read(8*1024*1024),b""):d.update(b)
 return d.hexdigest()
def wl(k,n):
 if not n:return 0.
 q=k/n;d=1+Z*Z/n;return (q+Z*Z/(2*n)-Z*math.sqrt(q*(1-q)/n+Z*Z/(4*n*n)))/d
def main():
 p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--output",required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);checks=[]
 def ck(name,ok,detail):checks.append({"name":name,"passed":bool(ok),"detail":detail})
 rm=js(run/"run_manifest.json");sel=js(run/"selected_length.json");freeze=js(run/"source_audit_freeze.json");profile=js(run/"transfer_primary/profile_manifest.json");proxy=js(run/"proxy_pair_auxiliary/profile_manifest.json");strict=js(run/"strict_pair/profile_manifest.json");pluto=js(run/"pluto_cache_disposition.json");reject=js(run/"rejected_samples.json")
 ids=(run/"source_length_audit_ids.txt").read_text().splitlines();calc=hashlib.sha256(("\n".join(ids)+"\n").encode()).hexdigest();ck("revision",all(x.get("protocol_revision")==REV for x in (rm,sel,freeze,profile,proxy,strict)),REV);ck("source_index_population",rm["native_index_rows"]==profile["source_index_rows"]==freeze["population_size"]==1_000_000,rm["native_index_rows"]);ck("audit_frozen_50k",len(ids)==len(set(ids))==freeze["audit_size"]==profile["source_length_audit_rows"]==50_000,{"rows":len(ids),"unique":len(set(ids))});ck("audit_hash",calc==freeze["sample_ids_sha256"]==profile["source_length_audit_sha256"]==rm["length_audit_sha256"],calc);ck("frozen_before_outcomes",freeze["frozen_before_outcomes"] and freeze["algorithm"]=="normalized-path-sha256-smallest-v1",freeze["algorithm"])
 audit=rows(run/"transfer_primary/source_length_audit.jsonl");ck("audit_accounting",len(audit)+len(reject["source_native_npz"])==50_000,{"valid":len(audit),"rejected":len(reject["source_native_npz"])});ck("native_80x3",all(r["trajectory"]["shape"]==[80,3] and abs(r["physical_horizon_s"]-8)<1e-9 for r in audit),len(audit));ck("pluto_excluded",pluto["status"]=="INVALID_SOURCE_DIAGNOSTIC" and not pluto["included_in_formal_pipeline"],pluto)
 eligible=[]
 for key,v in sel["coverage"].items():
  s=v["source_car"];t=v["target_anymal"];calcwl=wl(s["covered"],s["denominator"]);ck(f"wilson_{key}",abs(calcwl-s["wilson_lower_95_one_sided"])<1e-12,{"stored":s["wilson_lower_95_one_sided"],"calc":calcwl});
  if calcwl>=sel["min_train_coverage"] and t["exact_coverage"]>=sel["min_train_coverage"]:eligible.append(float(key))
 ck("largest_eligible",bool(eligible) and sel["selected_length_m"]==max(eligible),{"eligible":eligible,"selected":sel["selected_length_m"]});ck("proxy_excluded",sel["proxy_excluded"] and not proxy["included_in_formal_training"] and not proxy["included_in_length_selection"] and not proxy["included_in_main_table"],proxy);ck("strict_blocked",strict["status"]=="BLOCKED_WAITING_DATA",strict["status"])
 splitrows={s:rows(run/f"transfer_primary/target_{s}_windows.jsonl") for s in ("train","val","test")};eps={s:{r["episode_id"] for r in z} for s,z in splitrows.items()};ck("episode_disjoint",not(eps["train"]&eps["val"] or eps["train"]&eps["test"] or eps["val"]&eps["test"]),{s:len(x) for s,x in eps.items()});alltarget=sum(splitrows.values(),[]);ck("target_schema",all(r["route_set"]["future_gt_dependency"] is False and "frozen" in r["fixed_goal"]["rule"] and r["branch"] in {"moving_planning","stop_or_short"} for r in alltarget),len(alltarget))
 import numpy as np
 bad_repr=[]
 for r in alltarget:
  x=np.asarray(r["trajectory"].get("fixed_arc_length_80",[]),dtype=float);m=np.asarray(r["trajectory"].get("valid_mask",[]))
  if x.shape!=(80,4) or m.shape!=(80,) or not np.isfinite(x).all() or (x.shape==(80,4) and not np.allclose(x[:,2]**2+x[:,3]**2,1.0,atol=2e-4)):bad_repr.append(r["sample_id"])
 ck("target_fixed_arc_xy_cos_sin",not bad_repr,{"bad_count":len(bad_repr),"examples":bad_repr[:5]})
 pub=js(run/"publication_manifest.json");bad_pub=[]
 for rel,want in pub["expected_sha256"].items():
  p=run/rel
  if not p.exists() or sha(p)!=want:bad_pub.append(rel)
 ck("publication_hashes",not bad_pub,{"checked":len(pub["expected_sha256"]),"bad":bad_pub[:5]})
 summary=js(run/"split_summary.json");bad=[]
 ck("stratified_episode_split",summary.get("algorithm")=="branch-stratified-complete-episode-sha256-v1" and all(any(r["branch"]=="moving_planning" for r in splitrows[s]) for s in ("train","val","test")),summary.get("algorithm"))
 for seed,b in summary["budgets"].items():
  if not set(b["1"])<=set(b["10"])<=set(b["100"]):bad.append(seed)
 ck("nested_budgets",not bad,bad);fail=[x for x in checks if not x["passed"]];result={"status":"PASS" if not fail else "FAIL","checks_passed":len(checks)-len(fail),"checks_failed":len(fail),"source_audit":dict(Counter(r["branch"] for r in audit)),"target":dict(Counter(r["branch"] for r in alltarget)),"split_rows":{s:len(v) for s,v in splitrows.items()},"checks":checks,"failures":fail};(out/"validation.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps({k:result[k] for k in ("status","checks_passed","checks_failed","source_audit","target","split_rows")},indent=2));raise SystemExit(bool(fail))
if __name__=="__main__":main()
