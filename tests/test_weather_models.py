from ftis.weather.weather_models import CloudLayer, WeatherCondition, WeatherQuery


def test_weather_condition_serialization_and_model_payload() -> None:
    condition = WeatherCondition(
        latitude=39.0,
        longitude=-104.0,
        provider="unit",
        altitude_m=10000,
        wind_speed_kmh=42,
        pressure_hpa=1002,
        temperature_c=-40,
        cloud_layers=[CloudLayer("BKN", base_m=1200)],
    )

    payload = condition.to_model_payload()

    assert payload["windspeed"] == 42
    assert payload["pressure"] == 1002
    assert condition.as_dict()["cloud_layers"][0]["cover"] == "BKN"


def test_weather_query_cache_key_is_stable() -> None:
    first = WeatherQuery(39.001, -104.002, altitude_m=10120, station_id="kden")
    second = WeatherQuery(39.004, -104.004, altitude_m=10110, station_id="KDEN")

    assert first.cache_key() == second.cache_key()
