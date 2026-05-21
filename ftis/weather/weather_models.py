"""Unified weather schemas used by live providers and route analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _iso_timestamp(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class CloudLayer:
    """A normalized aviation cloud layer."""

    cover: str
    base_m: float | None = None
    top_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cover": self.cover,
            "base_m": self.base_m,
            "top_m": self.top_m,
        }


@dataclass(frozen=True)
class WeatherQuery:
    """Location query for provider adapters."""

    latitude: float
    longitude: float
    altitude_m: float | None = None
    station_id: str | None = None
    timestamp: datetime | None = None

    def cache_key(self) -> str:
        """Return a stable, privacy-safe cache key for repeated route samples."""

        station = (self.station_id or "route").upper()
        altitude = "na" if self.altitude_m is None else f"{round(self.altitude_m, -2):.0f}"
        timestamp = "latest"
        if self.timestamp:
            timestamp = self.timestamp.astimezone(timezone.utc).strftime("%Y%m%d%H")
        return (
            f"{station}:"
            f"{round(self.latitude, 2):.2f}:"
            f"{round(self.longitude, 2):.2f}:"
            f"{altitude}:"
            f"{timestamp}"
        )


@dataclass
class WeatherCondition:
    """Provider-neutral atmospheric state aligned to FTIS model features."""

    latitude: float
    longitude: float
    provider: str
    observed_at: datetime | str | None = None
    altitude_m: float | None = None
    station_id: str | None = None
    wind_speed_kmh: float | None = None
    wind_direction_deg: float | None = None
    pressure_hpa: float | None = None
    humidity_percent: float | None = None
    temperature_c: float | None = None
    turbulence_indicator: float | None = None
    jet_stream_indicator: float | None = None
    cloud_layers: list[CloudLayer] = field(default_factory=list)
    visibility_m: float | None = None
    source_priority: int = 100
    raw: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialize the weather state for API, dashboard, and cache consumers."""

        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "station_id": self.station_id,
            "provider": self.provider,
            "observed_at": _iso_timestamp(self.observed_at),
            "wind_speed_kmh": self.wind_speed_kmh,
            "wind_direction_deg": self.wind_direction_deg,
            "pressure_hpa": self.pressure_hpa,
            "humidity_percent": self.humidity_percent,
            "temperature_c": self.temperature_c,
            "turbulence_indicator": self.turbulence_indicator,
            "jet_stream_indicator": self.jet_stream_indicator,
            "cloud_layers": [layer.as_dict() for layer in self.cloud_layers],
            "visibility_m": self.visibility_m,
            "source_priority": self.source_priority,
            "warnings": self.warnings,
        }

    def to_model_payload(self, altitude_m: float | None = None) -> dict[str, float]:
        """Convert live weather into the current FTIS inference payload."""

        return {
            "latitude": float(self.latitude),
            "longitude": float(self.longitude),
            "altitude": float(altitude_m or self.altitude_m or 10_000.0),
            "windspeed": float(self.wind_speed_kmh or 0.0),
            "pressure": float(self.pressure_hpa or 1013.25),
            "temperature": float(self.temperature_c or 0.0),
        }

    @classmethod
    def from_cache(cls, payload: dict[str, Any]) -> "WeatherCondition":
        """Rehydrate a cached weather condition."""

        layers = [
            CloudLayer(
                cover=str(layer.get("cover", "UNK")),
                base_m=layer.get("base_m"),
                top_m=layer.get("top_m"),
            )
            for layer in payload.get("cloud_layers", [])
            if isinstance(layer, dict)
        ]
        return cls(
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            altitude_m=payload.get("altitude_m"),
            station_id=payload.get("station_id"),
            provider=str(payload.get("provider", "cache")),
            observed_at=payload.get("observed_at"),
            wind_speed_kmh=payload.get("wind_speed_kmh"),
            wind_direction_deg=payload.get("wind_direction_deg"),
            pressure_hpa=payload.get("pressure_hpa"),
            humidity_percent=payload.get("humidity_percent"),
            temperature_c=payload.get("temperature_c"),
            turbulence_indicator=payload.get("turbulence_indicator"),
            jet_stream_indicator=payload.get("jet_stream_indicator"),
            cloud_layers=layers,
            visibility_m=payload.get("visibility_m"),
            source_priority=int(payload.get("source_priority", 100)),
            warnings=list(payload.get("warnings", [])),
        )
