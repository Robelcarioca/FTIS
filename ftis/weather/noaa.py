"""NOAA/National Weather Service provider adapter."""

from __future__ import annotations

from typing import Any

from ftis.weather.normalization import as_float, estimate_turbulence_indicator
from ftis.weather.provider_base import AsyncWeatherProvider, WeatherProviderError
from ftis.weather.weather_models import CloudLayer, WeatherCondition, WeatherQuery


class NOAAProvider(AsyncWeatherProvider):
    """Fetch US grid forecast fields from api.weather.gov."""

    name = "noaa"
    priority = 20
    base_url = "https://api.weather.gov"

    def _first_grid_value(self, grid: dict[str, Any], key: str) -> float | None:
        values = (((grid.get("properties") or {}).get(key) or {}).get("values")) or []
        if not values:
            return None
        return as_float(values[0].get("value"))

    async def fetch(self, query: WeatherQuery) -> WeatherCondition:
        point_url = f"{self.base_url}/points/{query.latitude:.4f},{query.longitude:.4f}"
        point_payload = await self._get_json(point_url)
        properties = point_payload.get("properties") or {}
        grid_url = properties.get("forecastGridData")
        hourly_url = properties.get("forecastHourly")
        if not grid_url:
            raise WeatherProviderError("NOAA point response did not include grid data")

        grid_payload = await self._get_json(grid_url)
        hourly_payload: dict[str, Any] = {}
        if hourly_url:
            try:
                hourly_payload = await self._get_json(hourly_url)
            except WeatherProviderError:
                hourly_payload = {}

        sky_cover = self._first_grid_value(grid_payload, "skyCover")
        cloud_layers = []
        if sky_cover is not None and sky_cover >= 15:
            cloud_layers.append(CloudLayer(cover=f"SKY:{sky_cover:.0f}%", base_m=0.0))

        first_period = ((hourly_payload.get("properties") or {}).get("periods") or [{}])[0]
        wind_speed = self._first_grid_value(grid_payload, "windSpeed")
        if wind_speed is None:
            wind_speed = as_float(str(first_period.get("windSpeed", "")).split(" ")[0])

        condition = WeatherCondition(
            latitude=float(query.latitude),
            longitude=float(query.longitude),
            altitude_m=query.altitude_m,
            station_id=query.station_id,
            provider=self.name,
            observed_at=first_period.get("startTime"),
            wind_speed_kmh=wind_speed,
            wind_direction_deg=self._first_grid_value(grid_payload, "windDirection"),
            pressure_hpa=self._first_grid_value(grid_payload, "pressure"),
            humidity_percent=self._first_grid_value(grid_payload, "relativeHumidity"),
            temperature_c=self._first_grid_value(grid_payload, "temperature"),
            jet_stream_indicator=None,
            cloud_layers=cloud_layers,
            visibility_m=self._first_grid_value(grid_payload, "visibility"),
            source_priority=self.priority,
            raw={
                "provider": self.name,
                "grid_id": properties.get("gridId"),
                "grid_x": properties.get("gridX"),
                "grid_y": properties.get("gridY"),
            },
        )
        condition.turbulence_indicator = estimate_turbulence_indicator(condition)
        return condition
