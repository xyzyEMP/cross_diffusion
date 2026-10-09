import inspect
import numpy as np
from datasets.route_builder import OccupancyRouteSetBuilder,RouteSetBuilder,RouteSpec
def test_route_shape_mask_and_dedup():
    b=RouteSetBuilder(RouteSpec(max_candidates=6)); line=np.array([[0,0],[5,2],[10,0]],float)
    x=b.build([0,0],[10,0],[line,line.copy()])
    assert x["route_candidates_xy"].shape==(6,80,2)
    assert 1<=x["route_candidate_mask"].sum()<=2
    assert x["route_source"]=="map_goal"

def test_occupancy_builder_avoids_static_and_no_future_dependency():
    # observed flat cells cover the grid; one static cell lies on direct path
    flat=np.array([[i,j,0,1] for i in range(30) for j in range(30)],dtype=int)
    sparse=np.vstack([flat,np.array([[17,15,0,3]])])
    b=OccupancyRouteSetBuilder(RouteSpec(max_candidates=3,points_per_candidate=20),resolution_m=1)
    x=b.build(sparse,[5,0])
    assert x["route_candidate_mask"].any() and x["map_source"]=="current_coarse_occupancy"
    assert "future_gt" not in b.dependency_fields


def test_api_has_no_future_gt():
    assert "future" not in inspect.signature(RouteSetBuilder.build).parameters
    assert "future_gt" not in RouteSetBuilder.dependency_fields


def test_future_mutation_negative_control():
    b=RouteSetBuilder(RouteSpec()); kwargs=dict(current_xy=[0,0],goal_xy=[10,2],map_polylines=[np.array([[0,0],[5,1],[10,2]])])
    future_a=np.zeros((80,3)); a=b.build(**kwargs); future_a[:]=999
    c=b.build(**kwargs)
    assert np.array_equal(a["route_candidates_xy"],c["route_candidates_xy"])
    assert np.array_equal(a["route_candidate_mask"],c["route_candidate_mask"])
