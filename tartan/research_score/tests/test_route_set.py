import numpy as np
from tartan.research_score.data.route_builder import OccupancyRouteSetBuilder,RouteSpec
def test_occupancy_builder_avoids_static_and_no_future_dependency():
    # observed flat cells cover the grid; one static cell lies on direct path
    flat=np.array([[i,j,0,1] for i in range(30) for j in range(30)],dtype=int)
    sparse=np.vstack([flat,np.array([[17,15,0,3]])])
    b=OccupancyRouteSetBuilder(RouteSpec(max_candidates=3,points_per_candidate=20),resolution_m=1)
    x=b.build(sparse,[5,0])
    assert x["route_candidate_mask"].any() and x["map_source"]=="current_coarse_occupancy"
    assert "future_gt" not in b.dependency_fields
