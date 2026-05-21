from ftis.routes.route_models import RouteRequest
from ftis.routes.simulator import route_simulator


def test_route_simulation_without_live_weather() -> None:
    analysis = route_simulator.analyze_route_sync(
        RouteRequest("LAX", "JFK", cruising_altitude_m=11000, aircraft_speed_kt=455, waypoint_count=12),
        use_live_weather=False,
    )

    assert analysis.route_distance_nm > 0
    assert len(analysis.waypoints) == 12
    assert 0 <= analysis.cumulative_fti <= 100


def test_route_exports() -> None:
    analysis = route_simulator.analyze_route_sync(
        RouteRequest("DEN", "ORD", waypoint_count=8),
        use_live_weather=False,
    )

    geojson = route_simulator.to_geojson(analysis)
    csv_text = route_simulator.to_csv(analysis)

    assert geojson["type"] == "FeatureCollection"
    assert "waypoint_id" in csv_text
