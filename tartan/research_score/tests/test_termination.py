from tartan.research_score.evaluation.termination import TerminationConfig,termination_reason

def test_priority_and_stuck():
    c=TerminationConfig(goal_radius_m=.5,no_progress_window=3,no_progress_epsilon_m=.1,max_steps=8)
    assert termination_reason([.1],1,c,collision=True)=="collision"
    assert termination_reason([1,1,1],3,c)=="stuck"
    assert termination_reason([.4],1,c)=="success"
    assert termination_reason([3],8,c)=="timeout"
    assert termination_reason([3],1,c,route_failure=True)=="route_failure"
