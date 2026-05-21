"""AviationWeather.gov provider adapter for METAR and PIREP signals."""

from __future__ import annotations

import asyncio

from ftis.weather.normalization import (
    as_float,
    cloud_layers_from_awc,
    estimate_turbulence_indicator,
    knots_to_kmh,
    normalize_pressure,
    parse_visibility,
    turbulence_from_pireps,
)
from ftis.weather.provider_base import AsyncWeatherProvider, WeatherProviderError
from ftis.weather.weather_models import WeatherCondition, WeatherQuery


class AviationWeatherProvider(AsyncWeatherProvider):
    """Fetch station-centric METAR and PIREP data from AviationWeather.gov."""

    name = "aviation_weather"
    priority = 30
    base_url = "https://aviationweather.gov/api/data"

    async def fetch(self, query: WeatherQuery) -> WeatherCondition:
        if not query.station_id:
            raise WeatherProviderError("AviationWeather.gov requires a station_id")

        station = query.station_id.upper()
        metar_params = {"ids": station, "format": "json"}
        pirep_params = {"ids": station, "format": "json", "hours": 3}

        metar_task = self._get_json(f"{self.base_url}/metar", params=metar_params)
        pirep_task = self._get_json(f"{self.base_url}/pirep", params=pirep_params)
        metar_payload, pirep_payload = await asyncio.gather(
            metar_task,
            pirep_task,
            return_exceptions=True,
        )

        if isinstance(metar_payload, Exception) or not metar_payload:
            raise WeatherProviderError("AviationWeather.gov returned no METAR data")
        if not isinstance(metar_payload, list):
            metar_payload = [metar_payload]

        metar = metar_payload[0]
        pireps = [] if isinstance(pirep_payload, Exception) else pirep_payload
        if not isinstance(pireps, list):
            pireps = [pireps]

        wind_speed = (
            as_float(metar.get("windSpeed"))
            or as_float(metar.get("wspd"))
            or as_float(metar.get("wind_speed_kt"))
        )
        condition = WeatherCondition(
            latitude=float(as_float(metar.get("lat"), query.latitude) or query.latitude),
            longitude=float(as_float(metar.get("lon"), query.longitude) or query.longitude),
            altitude_m=query.altitude_m,
            station_id=station,
            provider=self.name,
            observed_at=metar.get("obsTime") or metar.get("reportTime"),
            wind_speed_kmh=knots_to_kmh(wind_speed),
            wind_direction_deg=as_float(metar.get("windDirection"))
            or as_float(metar.get("wdir")),
            pressure_hpa=normalize_pressure(
                metar.get("altim")
                or metar.get("altimeter")
                or metar.get("pressure")
            ),
            humidity_percent=as_float(metar.get("humidity")),
            temperature_c=as_float(metar.get("temp"))
            or as_float(metar.get("temperature")),
            jet_stream_indicator=None,
            cloud_layers=cloud_layers_from_awc(metar),
            visibility_m=parse_visibility(metar.get("visib") or metar.get("visibility")),
            source_priority=self.priority,
            raw={"provider": self.name, "flight_category": metar.get("fltCat")},
        )
        pirep_turbulence = turbulence_from_pireps(pireps)
        condition.turbulence_indicator = (
            pirep_turbulence
            if pirep_turbulence is not None
            else estimate_turbulence_indicator(condition)
        )
        return condition
