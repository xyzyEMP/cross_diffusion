import numpy as np
from tartan.research_score.data.route_builder import OccupancyRouteSetBuilder,RouteSetBuilder,RouteSpec
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
