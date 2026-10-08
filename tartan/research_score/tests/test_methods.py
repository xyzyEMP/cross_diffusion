import pytest
from tartan.research_score.training.methods import method_spec
from tartan.research_score.evaluation.fairness_audit import audit
def test_profiles_and_fairness():
 assert method_spec("pretrain_finetune","transfer_primary")["init"]=="nuplan"
 with pytest.raises(KeyError):method_spec("unknown_method","transfer_primary")
 with pytest.raises(ValueError):method_spec("pretrain_finetune","proposed_b")
 a={"method":"a","target_manifest":"m","budget":1,"repeat_id":11,"future_points":80,"length_m":8,"controller":"c","eval_manifest":"e"};b={**a,"method":"b"};assert audit([a,b])["passed"]
