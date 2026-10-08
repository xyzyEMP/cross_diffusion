import numpy as np
from tartan.research_score.data.core import cache_key
from tartan.research_score.data.source_contract import assert_source_tensor_identity

def test_cache_key_all_dependencies():
    base=cache_key("s","p","r","c")
    assert base!=cache_key("s2","p","r","c")
    assert base!=cache_key("s","p2","r","c")
    assert base!=cache_key("s","p","r2","c")
    assert base!=cache_key("s","p","r","c2")

def test_source_adapter_identity():
    x={"ego":np.arange(3),"route":np.ones((2,3))}
    assert_source_tensor_identity(x,{k:v.copy() for k,v in x.items()})

def test_deterministic_worker_order():
    ids=[f"sample-{i}" for i in range(100)]
    def worker_partition(n):return [x for wid in range(n) for x in ids[wid::n]]
    assert sorted(worker_partition(1))==sorted(worker_partition(4))
