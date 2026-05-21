"""Realistic route interpolation, weather sampling, and turbulence analysis."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from ftis.config import RECOMMENDATIONS
from ftis.features import engineer_turbulence_features
from ftis.inference import PredictionService
from ftis.routes.airports import resolve_airport
from ftis.routes.exports import route_to_csv, route_to_geojson, route_to_map_overlay
from ftis.routes.route_models import RouteAnalysis, RouteRequest, RouteWaypoint
from ftis.weather.weather_models import WeatherCondition, WeatherQuery
from ftis.weather.weather_service import WeatherService, weather_service


logger = logging.getLogger(__name__)


RISK_RANK = {"Low": 1, "Moderate": 2, "High": 3}


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in nautical miles."""

    radius_nm = 3440.065
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return float(2 * radius_nm * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def _slerp_points(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    count: int,
) -> list[tuple[float, float]]:
    """Interpolate great-circle waypoints."""

    lat1, lon1, lat2, lon2 = map(math.radians, [start_lat, start_lon, end_lat, end_lon])
    start = np.array([math.cos(lat1) * math.cos(lon1), math.cos(lat1) * math.sin(lon1), math.sin(lat1)])
    end = np.array([math.cos(lat2) * math.cos(lon2), math.cos(lat2) * math.sin(lon2), math.sin(lat2)])
    omega = math.acos(float(np.clip(np.dot(start, end), -1.0, 1.0)))
    if abs(omega) < 1e-9:
        return [(start_lat, start_lon)] * count

    points: list[tuple[float, float]] = []
    for fraction in np.linspace(0.0, 1.0, count):
        vector = (
            math.sin((1 - fraction) * omega) / math.sin(omega) * start
            + math.sin(fraction * omega) / math.sin(omega) * end
        )
        lat = math.degrees(math.atan2(vector[2], math.sqrt(vector[0] ** 2 + vector[1] ** 2)))
        lon = math.degrees(math.atan2(vector[1], vector[0]))
        points.append((float(lat), float(lon)))
    return points


def _altitude_at_progress(progress: float, cruise_altitude_m: float) -> float:
    """Generate a conservative climb-cruise-descent altitude profile."""

    start_altitude = 450.0
    climb_fraction = 0.22
    descent_fraction = 0.22
    if progress < climb_fraction:
        x = progress / climb_fraction
        return start_altitude + (cruise_altitude_m - start_altitude) * math.sin(x * math.pi / 2)
    if progress > 1 - descent_fraction:
        x = (1 - progress) / descent_fraction
        return start_altitude + (cruise_altitude_m - start_altitude) * math.sin(x * math.pi / 2)
    return cruise_altitude_m


def _synthetic_weather(lat: float, lon: float, altitude_m: float, progress: float) -> WeatherCondition:
    """Deterministic fallback used when live providers are unavailable."""

    jet_bonus = max(0.0, math.sin(progress * math.pi)) * 28.0 if altitude_m >= 8000 else 0.0
    mountain_wave = 18.0 * math.exp(-((lon + 106.0) ** 2) / 28.0) if 34 <= lat <= 46 else 0.0
    wind = 22.0 + jet_bonus + mountain_wave
    pressure = 1013.25 - altitude_m / 100.0 + 5.0 * math.sin(progress * math.pi * 2)
    temperature = 16.0 - altitude_m * 0.0065
    condition = WeatherCondition(
        latitude=lat,
        longitude=lon,
        altitude_m=altitude_m,
        provider="synthetic_route_fallback",
        wind_speed_kmh=wind,
        wind_direction_deg=(240.0 + progress * 35.0) % 360,
        pressure_hpa=pressure,
        humidity_percent=45.0 + 20.0 * math.sin(progress * math.pi),
        temperature_c=temperature,
        visibility_m=14_000.0,
        jet_stream_indicator=wind + 40.0 if altitude_m >= 8000 else wind,
    )
    condition.turbulence_indicator = min(100.0, wind * 0.9 + mountain_wave)
    return condition


class RouteSimulator:
    """Build route trajectories, sample weather, and score turbulence risk."""

    def __init__(
        self,
        prediction_service: PredictionService | None = None,
        live_weather: WeatherService | None = None,
    ) -> None:
        self.prediction_service = prediction_service or PredictionService()
        self.weather_service = live_weather or weather_service

    async def analyze_route(
        self,
        request: RouteRequest,
        *,
        use_live_weather: bool = True,
    ) -> RouteAnalysis:
        """Analyze an airport-to-airport route."""

        waypoint_count = int(max(6, min(160, request.waypoint_count)))
        departure = resolve_airport(request.departure_airport)
        destination = resolve_airport(request.destination_airport)
        points = _slerp_points(
            departure.latitude,
            departure.longitude,
            destination.latitude,
            destination.longitude,
            waypoint_count,
        )

        cumulative_distance = [0.0]
        for index in range(1, len(points)):
            cumulative_distance.append(
                cumulative_distance[-1]
                + haversine_nm(*points[index - 1], *points[index])
            )
        total_distance = cumulative_distance[-1]
        estimated_time = total_distance / max(request.aircraft_speed_kt, 1.0) * 60.0

        queries = []
        for index, (lat, lon) in enumerate(points):
            progress = index / max(waypoint_count - 1, 1)
            altitude = _altitude_at_progress(progress, request.cruising_altitude_m)
            station_id = departure.code if index < waypoint_count / 2 else destination.code
            queries.append(WeatherQuery(lat, lon, altitude_m=altitude, station_id=station_id))

        sampled: list[WeatherCondition | None] = []
        if use_live_weather:
            sampled = await self.weather_service.get_route_weather(queries)
        else:
            sampled = [None] * len(queries)

        waypoints: list[RouteWaypoint] = []
        for index, query in enumerate(queries):
            progress = index / max(waypoint_count - 1, 1)
            condition = sampled[index] or _synthetic_weather(
                query.latitude,
                query.longitude,
                query.altitude_m or request.cruising_altitude_m,
                progress,
            )
            payload = condition.to_model_payload(query.altitude_m)
            try:
                prediction = self.prediction_service.predict(payload).as_dict()
            except Exception:
                features = engineer_turbulence_features(pd.DataFrame([payload]))
                risk = str(features["turbulence_label"].iloc[0])
                fti = float(features["FTI"].iloc[0])
                prediction = {
                    "risk": risk,
                    "confidence": round(0.55 + abs(fti - 50) / 100, 4),
                    "FTI": round(fti, 2),
                    "recommendation": RECOMMENDATIONS.get(risk, "Review route."),
                }

            segment_distance = 0.0 if index == 0 else cumulative_distance[index] - cumulative_distance[index - 1]
            waypoints.append(
                RouteWaypoint(
                    waypoint_id=f"WP-{index + 1:03d}",
                    latitude=query.latitude,
                    longitude=query.longitude,
                    altitude_m=float(query.altitude_m or request.cruising_altitude_m),
                    elapsed_minutes=round(estimated_time * progress, 2),
                    distance_from_origin_nm=round(cumulative_distance[index], 2),
                    segment_distance_nm=round(segment_distance, 2),
                    wind_speed_kmh=float(condition.wind_speed_kmh or 0.0),
                    wind_direction_deg=condition.wind_direction_deg,
                    pressure_hpa=float(condition.pressure_hpa or 1013.25),
                    humidity_percent=condition.humidity_percent,
                    temperature_c=float(condition.temperature_c or 0.0),
                    visibility_m=condition.visibility_m,
                    turbulence_indicator=condition.turbulence_indicator,
                    jet_stream_indicator=condition.jet_stream_indicator,
                    risk=str(prediction["risk"]),
                    confidence=float(prediction.get("confidence", 0.0)),
                    FTI=float(prediction["FTI"]),
                    provider=condition.provider,
                    recommendation=str(prediction.get("recommendation", "")),
                )
            )

        max_fti = max(waypoint.FTI for waypoint in waypoints)
        cumulative_fti = sum(
            waypoint.FTI * max(waypoint.segment_distance_nm, 1.0)
            for waypoint in waypoints
        ) / sum(max(waypoint.segment_distance_nm, 1.0) for waypoint in waypoints)
        route_risk = max(waypoints, key=lambda waypoint: (RISK_RANK.get(waypoint.risk, 0), waypoint.FTI)).risk

        analysis = RouteAnalysis(
            request=RouteRequest(
                request.departure_airport.upper(),
                request.destination_airport.upper(),
                request.cruising_altitude_m,
                request.aircraft_speed_kt,
                waypoint_count,
            ),
            departure_name=departure.name,
            destination_name=destination.name,
            waypoints=waypoints,
            route_distance_nm=round(total_distance, 2),
            estimated_time_minutes=round(estimated_time, 2),
            route_risk=route_risk,
            cumulative_fti=round(cumulative_fti, 2),
            max_fti=round(max_fti, 2),
            turbulence_corridors=self.detect_corridors(waypoints),
        )
        logger.info(
            "Route analysis complete dep=%s dst=%s risk=%s cumulative_fti=%.2f",
            request.departure_airport,
            request.destination_airport,
            analysis.route_risk,
            analysis.cumulative_fti,
        )
        return analysis

    def analyze_route_sync(self, request: RouteRequest, *, use_live_weather: bool = True) -> RouteAnalysis:
        """Synchronous adapter for scripts and Streamlit."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("Use await analyze_route inside an active event loop")
        return asyncio.run(self.analyze_route(request, use_live_weather=use_live_weather))

    def detect_corridors(self, waypoints: list[RouteWaypoint]) -> list[dict[str, Any]]:
        """Return contiguous route sections with elevated turbulence risk."""

        corridors: list[dict[str, Any]] = []
        active: list[RouteWaypoint] = []
        for waypoint in waypoints:
            elevated = waypoint.risk == "High" or waypoint.FTI >= 66.0
            if elevated:
                active.append(waypoint)
                continue
            if active:
                corridors.append(self._corridor_from_waypoints(active))
                active = []
        if active:
            corridors.append(self._corridor_from_waypoints(active))
        return corridors

    def _corridor_from_waypoints(self, waypoints: list[RouteWaypoint]) -> dict[str, Any]:
        return {
            "start_waypoint": waypoints[0].waypoint_id,
            "end_waypoint": waypoints[-1].waypoint_id,
            "distance_start_nm": waypoints[0].distance_from_origin_nm,
            "distance_end_nm": waypoints[-1].distance_from_origin_nm,
            "max_fti": round(max(waypoint.FTI for waypoint in waypoints), 2),
            "mean_wind_kmh": round(
                sum(waypoint.wind_speed_kmh for waypoint in waypoints) / len(waypoints),
                2,
            ),
        }

    def to_geojson(self, analysis: RouteAnalysis) -> dict[str, Any]:
        return route_to_geojson(analysis)

    def to_csv(self, analysis: RouteAnalysis) -> str:
        return route_to_csv(analysis)

    def to_map_overlay(self, analysis: RouteAnalysis) -> dict[str, Any]:
        return route_to_map_overlay(analysis)


route_simulator = RouteSimulator()
