from tartan.research_score.data.core import assert_no_split_leak,group_split,nested_budgets
def test_group_split_and_nested():
    s=group_split([f"p{i}" for i in range(24)]);assert set(s.values())=={"train","val","test"}
    b=nested_budgets([x for x,v in s.items() if v=="train"],11)
    assert set(b["1"])<=set(b["10"])<=set(b["100"])
def test_no_leak():assert_no_split_leak([{"map_id":"a","trajectory_id":"t","split":"train"}])
