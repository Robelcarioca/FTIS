"""Open-Meteo provider adapter."""

from __future__ import annotations

from datetime import timezone

from ftis.weather.normalization import as_float, estimate_turbulence_indicator, open_meteo_cloud_layers
from ftis.weather.provider_base import AsyncWeatherProvider, WeatherProviderError
from ftis.weather.weather_models import WeatherCondition, WeatherQuery


class OpenMeteoProvider(AsyncWeatherProvider):
    """Fetch global model weather from Open-Meteo."""

    name = "open_meteo"
    priority = 10
    base_url = "https://api.open-meteo.com/v1/forecast"

    async def fetch(self, query: WeatherQuery) -> WeatherCondition:
        current_fields = [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "surface_pressure",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "visibility",
        ]
        hourly_fields = [
            "wind_speed_250hPa",
            "wind_speed_300hPa",
            "wind_direction_250hPa",
        ]
        params = {
            "latitude": query.latitude,
            "longitude": query.longitude,
            "current": ",".join(current_fields),
            "hourly": ",".join(hourly_fields),
            "forecast_days": 1,
            "timezone": "UTC",
            "wind_speed_unit": "kmh",
        }
        payload = await self._get_json(self.base_url, params=params)
        current = payload.get("current") or {}
        if not current:
            raise WeatherProviderError("Open-Meteo response did not include current weather")

        hourly = payload.get("hourly") or {}
        jet_candidates = []
        for key in ("wind_speed_250hPa", "wind_speed_300hPa"):
            values = hourly.get(key) or []
            if values:
                candidate = as_float(values[0])
                if candidate is not None:
                    jet_candidates.append(candidate)

        condition = WeatherCondition(
            latitude=float(query.latitude),
            longitude=float(query.longitude),
            altitude_m=query.altitude_m,
            station_id=query.station_id,
            provider=self.name,
            observed_at=current.get("time"),
            wind_speed_kmh=as_float(current.get("wind_speed_10m")),
            wind_direction_deg=as_float(current.get("wind_direction_10m")),
            pressure_hpa=as_float(current.get("pressure_msl"))
            or as_float(current.get("surface_pressure")),
            humidity_percent=as_float(current.get("relative_humidity_2m")),
            temperature_c=as_float(current.get("temperature_2m")),
            jet_stream_indicator=max(jet_candidates) if jet_candidates else None,
            cloud_layers=open_meteo_cloud_layers(current),
            visibility_m=as_float(current.get("visibility")),
            source_priority=self.priority,
            raw={"provider": self.name, "elevation": payload.get("elevation")},
        )
        condition.turbulence_indicator = estimate_turbulence_indicator(condition)
        return condition
