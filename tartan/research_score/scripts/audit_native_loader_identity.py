import argparse, hashlib, io, json
from pathlib import Path
import numpy as np
from mmengine import fileio

def original_loader(path):
    return np.load(io.BytesIO(fileio.get(str(path))), allow_pickle=False)

def main():
    p=argparse.ArgumentParser();p.add_argument("--index",required=True);p.add_argument("--root",required=True);p.add_argument("--audit-ids",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    names=json.loads(Path(a.index).read_text());root=Path(a.root)
    by_id={"nuplan-native:"+hashlib.sha256((root/n).as_posix().encode()).hexdigest():root/n for n in names}
    ids=Path(a.audit_ids).read_text().splitlines();chosen=[ids[i] for i in np.linspace(0,len(ids)-1,100,dtype=int)]
    mismatches=[];checked=0
    for sid in chosen:
        path=by_id[sid]
        with original_loader(path) as old, np.load(path,allow_pickle=False) as new:
            if set(old.files)!=set(new.files):mismatches.append({"sample_id":sid,"reason":"keys"});continue
            bad=[]
            for k in old.files:
                x,y=old[k],new[k]
                if x.shape!=y.shape or x.dtype!=y.dtype or not np.array_equal(x,y):bad.append(k)
            if bad:mismatches.append({"sample_id":sid,"reason":"values","keys":bad})
            checked+=1
    result={"status":"PASS" if not mismatches else "FAIL","selection":"100_evenly_spaced_over_frozen_50k","checked":checked,"mismatches":mismatches,"claim":"original_mmengine_fileio_npz_loader_equals_native_manifest_adapter_read"}
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2));raise SystemExit(bool(mismatches))
if __name__=="__main__":main()
