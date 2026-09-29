from __future__ import annotations

import argparse, json, os, subprocess, tempfile
from pathlib import Path
import torch

from tartan.research_score.data.budget_sampler import nested_window_budgets


def publish_text(path, text):
    fd,tmp=tempfile.mkstemp(prefix="d029_ext_",suffix=".txt");
    try:
        with os.fdopen(fd,"w") as f: f.write(text);f.flush();os.fsync(f.fileno())
        subprocess.run(["dd","if="+tmp,"of="+str(path),"conv=fsync","status=none"],check=True)
    finally: Path(tmp).unlink(missing_ok=True)


def publish_torch(path, value):
    fd,tmp=tempfile.mkstemp(prefix="d029_ext_",suffix=".pt");os.close(fd)
    try:
        torch.save(value,tmp);subprocess.run(["dd","if="+tmp,"of="+str(path),"conv=fsync","status=none"],check=True)
    finally: Path(tmp).unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--cache",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=11);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()]
    fractions=(("1",.01),("10",.10),("20",.20),("50",.50),("100",1.0));result=nested_window_budgets(rows,a.seed,fractions)
    membership=[]
    for row in rows:
        tags=result["memberships"][str(row["sample_id"])];row["budget_membership"]={str(a.seed):tags};membership.append({str(a.seed):tags})
    cache=torch.load(a.cache,map_location="cpu",weights_only=False)
    if list(cache["sample_ids"]) != [str(r["sample_id"]) for r in rows]: raise RuntimeError("manifest/cache sample order mismatch")
    cache["budget_membership"]=membership
    publish_text(out/"target_train_windows_d029_extended.jsonl","".join(json.dumps(r,sort_keys=True,separators=(",",":"))+"\n" for r in rows))
    publish_torch(out/"target_train_features_d029_extended.pt",cache)
    summary={k:v for k,v in result.items() if k not in ("budgets","memberships")};summary.update({"status":"COMPLETE","fractions":[x[0] for x in fractions],"source_manifest":a.manifest,"source_cache":a.cache})
    publish_text(out/"budget_summary.json",json.dumps(summary,indent=2,sort_keys=True)+"\n");print(json.dumps(summary,sort_keys=True))


if __name__=="__main__":main()
