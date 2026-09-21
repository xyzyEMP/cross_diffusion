import pytest
from tartan.research_score.data.trajectory_spec import select_length
def test_largest_candidate():
    x=select_length({"a":[21]*9+[7],"b":[16]*10},threshold=.9);assert x["selected_length_m"]==15
def test_no_candidate_blocks():
    with pytest.raises(RuntimeError,match="BLOCKED"):select_length({"a":[2,3],"b":[4,5]})
