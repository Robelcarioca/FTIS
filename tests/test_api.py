from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_system_status() -> None:
    response = client.get("/system/status")
    assert response.status_code == 200
    assert "weather" in response.json()


def test_route_analyze_endpoint_without_live_weather() -> None:
    response = client.post(
        "/route/analyze",
        json={
            "departure_airport": "LAX",
            "destination_airport": "JFK",
            "cruising_altitude_m": 11000,
            "aircraft_speed_kt": 455,
            "waypoint_count": 8,
            "use_live_weather": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis"]["route_distance_nm"] > 0
    assert payload["geojson"]["type"] == "FeatureCollection"
