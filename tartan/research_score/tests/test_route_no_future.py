import inspect,numpy as np
from tartan.research_score.data.route_builder import RouteSetBuilder,RouteSpec

def test_api_has_no_future_gt():
    assert "future" not in inspect.signature(RouteSetBuilder.build).parameters
    assert "future_gt" not in RouteSetBuilder.dependency_fields

def test_future_mutation_negative_control():
    b=RouteSetBuilder(RouteSpec()); kwargs=dict(current_xy=[0,0],goal_xy=[10,2],map_polylines=[np.array([[0,0],[5,1],[10,2]])])
    future_a=np.zeros((80,3)); a=b.build(**kwargs); future_a[:]=999
    c=b.build(**kwargs)
    assert np.array_equal(a["route_candidates_xy"],c["route_candidates_xy"])
    assert np.array_equal(a["route_candidate_mask"],c["route_candidate_mask"])
