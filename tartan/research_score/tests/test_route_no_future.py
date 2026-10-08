import inspect
from tartan.research_score.data.route_builder import OccupancyRouteSetBuilder

def test_route_api_accepts_only_observed_map_and_fixed_goal():
    assert "future_gt" not in inspect.signature(OccupancyRouteSetBuilder.build).parameters
    assert "future_gt" not in OccupancyRouteSetBuilder.dependency_fields
