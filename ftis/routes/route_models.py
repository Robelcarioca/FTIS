"""Schemas for FTIS route simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouteRequest:
    """Flight-route analysis inputs."""

    departure_airport: str
    destination_airport: str
    cruising_altitude_m: float = 10_700.0
    aircraft_speed_kt: float = 450.0
    waypoint_count: int = 36


@dataclass
class RouteWaypoint:
    """An interpolated route waypoint with atmospheric and risk fields."""

    waypoint_id: str
    latitude: float
    longitude: float
    altitude_m: float
    elapsed_minutes: float
    distance_from_origin_nm: float
    segment_distance_nm: float
    wind_speed_kmh: float
    wind_direction_deg: float | None
    pressure_hpa: float
    humidity_percent: float | None
    temperature_c: float
    visibility_m: float | None
    turbulence_indicator: float | None
    jet_stream_indicator: float | None
    risk: str
    confidence: float
    FTI: float
    provider: str
    recommendation: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RouteAnalysis:
    """Route-level FTIS analysis output."""

    request: RouteRequest
    departure_name: str
    destination_name: str
    waypoints: list[RouteWaypoint]
    route_distance_nm: float
    estimated_time_minutes: float
    route_risk: str
    cumulative_fti: float
    max_fti: float
    turbulence_corridors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.__dict__,
            "departure_name": self.departure_name,
            "destination_name": self.destination_name,
            "route_distance_nm": self.route_distance_nm,
            "estimated_time_minutes": self.estimated_time_minutes,
            "route_risk": self.route_risk,
            "cumulative_fti": self.cumulative_fti,
            "max_fti": self.max_fti,
            "turbulence_corridors": self.turbulence_corridors,
            "waypoints": [waypoint.as_dict() for waypoint in self.waypoints],
        }
