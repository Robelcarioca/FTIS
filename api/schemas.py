"""Pydantic schemas for the FTIS API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.version import VERSION as PYDANTIC_VERSION


PYDANTIC_V2 = int(PYDANTIC_VERSION.split(".", maxsplit=1)[0]) >= 2

if PYDANTIC_V2:
    from pydantic import ConfigDict


EXAMPLE_PAYLOAD = {
    "altitude": 11000,
    "velocity": 240,
    "heading": 180,
    "temperature": 12,
    "windspeed": 35,
    "winddirection": 220,
    "pressure": 1008,
    "humidity": 40,
    "vertical_speed": 2,
    "turn_rate": 0.3,
    "speed_variation": 5,
    "altitude_variation": 20,
    "pressure_variation": 3,
    "wind_shear_proxy": 0.002,
}

PREDICTION_EXAMPLE = {
    "latitude": 39.8729,
    "longitude": -104.6737,
    "altitude": 10800,
    "windspeed": 42,
    "pressure": 1002,
    "temperature": -42,
}

PREDICTION_RESPONSE_EXAMPLE = {
    "risk": "High",
    "confidence": 0.91,
    "FTI": 82,
    "recommendation": "Avoid route section or request tactical reroute.",
    "probabilities": {
        "Low": 0.03,
        "Moderate": 0.06,
        "High": 0.91,
    },
}


class FlightFeatures(BaseModel):
    """Feature payload for real-time turbulence inference."""

    altitude: float = Field(..., description="Aircraft altitude")
    velocity: float = Field(..., description="Aircraft velocity")
    heading: float = Field(..., ge=0, le=360, description="Aircraft heading")
    temperature: float = Field(..., description="Ambient temperature")
    windspeed: float = Field(..., ge=0, description="Wind speed")
    winddirection: float = Field(..., ge=0, le=360, description="Wind direction")
    pressure: float = Field(..., gt=0, description="Atmospheric pressure")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity")
    vertical_speed: float = Field(..., description="Aircraft vertical speed")
    turn_rate: float = Field(..., description="Aircraft turn rate")
    speed_variation: float = Field(..., ge=0, description="Recent speed variation")
    altitude_variation: float = Field(..., ge=0, description="Recent altitude variation")
    pressure_variation: float = Field(..., ge=0, description="Recent pressure variation")
    wind_shear_proxy: float = Field(..., ge=0, description="Derived wind shear proxy")

    if PYDANTIC_V2:
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"example": EXAMPLE_PAYLOAD},
        )
    else:
        class Config:
            extra = "forbid"
            schema_extra = {"example": EXAMPLE_PAYLOAD}


class TurbulencePredictionRequest(BaseModel):
    """Minimal flight and atmosphere payload for FTIS production inference."""

    latitude: float = Field(..., ge=-90, le=90, description="Aircraft latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Aircraft longitude")
    altitude: float = Field(..., ge=0, le=20000, description="Aircraft altitude in meters")
    windspeed: float = Field(..., ge=0, le=150, description="Wind speed")
    pressure: float = Field(..., gt=800, lt=1100, description="Mean sea-level pressure")
    temperature: float = Field(..., ge=-90, le=60, description="Ambient temperature")

    if PYDANTIC_V2:
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"example": PREDICTION_EXAMPLE},
        )
    else:
        class Config:
            extra = "forbid"
            schema_extra = {"example": PREDICTION_EXAMPLE}


class StationPredictionRequest(BaseModel):
    """Station-oriented dashboard payload for backend-dependent prediction."""

    altitude: float = Field(..., ge=0, le=20000, description="Cruising altitude in meters")
    speed: float = Field(..., ge=0, le=1200, description="Aircraft speed in knots")
    temperature: float = Field(..., ge=-90, le=60, description="Outside air temperature")
    windspeed: float = Field(..., ge=0, le=150, description="Wind speed")
    departure_station: str = Field(..., min_length=3, max_length=4)
    destination_station: str = Field(..., min_length=3, max_length=4)
    pressure: float = Field(1013.25, gt=800, lt=1100, description="Pressure fallback")

    if PYDANTIC_V2:
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={
                "example": {
                    "altitude": 10600,
                    "speed": 455,
                    "temperature": -38,
                    "windspeed": 42,
                    "departure_station": "ADD",
                    "destination_station": "DXB",
                }
            },
        )
    else:
        class Config:
            extra = "forbid"
            schema_extra = {
                "example": {
                    "altitude": 10600,
                    "speed": 455,
                    "temperature": -38,
                    "windspeed": 42,
                    "departure_station": "ADD",
                    "destination_station": "DXB",
                }
            }


class TurbulencePredictionResponse(BaseModel):
    """FTIS prediction response returned by the production endpoint."""

    risk: Literal["Low", "Moderate", "High"]
    confidence: float = Field(..., ge=0, le=1)
    FTI: float = Field(..., ge=0, le=100)
    recommendation: str
    probabilities: dict[str, float]

    if PYDANTIC_V2:
        model_config = ConfigDict(
            json_schema_extra={"example": PREDICTION_RESPONSE_EXAMPLE},
        )
    else:
        class Config:
            schema_extra = {"example": PREDICTION_RESPONSE_EXAMPLE}


class BatchPredictionRequest(BaseModel):
    """Batch prediction payload for operational route or replay scoring."""

    records: list[TurbulencePredictionRequest | StationPredictionRequest] = Field(
        ...,
        min_items=1,
        max_items=500,
        description="Prediction payloads to score in one request",
    )

    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    count: int
    predictions: list[TurbulencePredictionResponse]


class CloudLayerResponse(BaseModel):
    """Normalized cloud layer returned by live weather endpoints."""

    cover: str
    base_m: float | None = None
    top_m: float | None = None


class WeatherLiveRequest(BaseModel):
    """Live aviation weather query."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude_m: float | None = Field(None, ge=0, le=20000)
    station_id: str | None = Field(
        None,
        min_length=3,
        max_length=4,
        description="Optional ICAO station for AviationWeather.gov METAR/PIREP enrichment",
    )

    if PYDANTIC_V2:
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={
                "example": {
                    "latitude": 39.8561,
                    "longitude": -104.6737,
                    "altitude_m": 10600,
                    "station_id": "KDEN",
                }
            },
        )
    else:
        class Config:
            extra = "forbid"
            schema_extra = {
                "example": {
                    "latitude": 39.8561,
                    "longitude": -104.6737,
                    "altitude_m": 10600,
                    "station_id": "KDEN",
                }
            }


class WeatherLiveResponse(BaseModel):
    """Unified live weather response."""

    latitude: float
    longitude: float
    altitude_m: float | None = None
    station_id: str | None = None
    provider: str
    observed_at: str | None = None
    wind_speed_kmh: float | None = None
    wind_direction_deg: float | None = None
    pressure_hpa: float | None = None
    humidity_percent: float | None = None
    temperature_c: float | None = None
    turbulence_indicator: float | None = None
    jet_stream_indicator: float | None = None
    cloud_layers: list[CloudLayerResponse] = Field(default_factory=list)
    visibility_m: float | None = None
    source_priority: int
    warnings: list[str] = Field(default_factory=list)


class RouteAnalyzeRequest(BaseModel):
    """Airport-to-airport turbulence route analysis request."""

    departure_airport: str = Field(..., min_length=3, max_length=4)
    destination_airport: str = Field(..., min_length=3, max_length=4)
    cruising_altitude_m: float = Field(10700, ge=1500, le=16000)
    aircraft_speed_kt: float = Field(450, ge=120, le=650)
    waypoint_count: int = Field(36, ge=6, le=160)
    use_live_weather: bool = Field(True, description="Use provider weather when available")

    if PYDANTIC_V2:
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={
                "example": {
                    "departure_airport": "LAX",
                    "destination_airport": "JFK",
                    "cruising_altitude_m": 11000,
                    "aircraft_speed_kt": 455,
                    "waypoint_count": 40,
                    "use_live_weather": False,
                }
            },
        )
    else:
        class Config:
            extra = "forbid"
            schema_extra = {
                "example": {
                    "departure_airport": "LAX",
                    "destination_airport": "JFK",
                    "cruising_altitude_m": 11000,
                    "aircraft_speed_kt": 455,
                    "waypoint_count": 40,
                    "use_live_weather": False,
                }
            }


class RouteAnalyzeResponse(BaseModel):
    """Route analysis response with exports for map overlays."""

    analysis: dict[str, Any]
    geojson: dict[str, Any]
    csv_preview: str
    overlay: dict[str, Any]


class SystemStatusResponse(BaseModel):
    """Operational health diagnostics."""

    status: Literal["healthy", "degraded"]
    model_available: bool
    weather: dict[str, Any]
    cache: dict[str, Any]
    version: str


class ModelMetricsResponse(BaseModel):
    """Model metrics and monitoring summary."""

    model: dict[str, Any]
    explainability: dict[str, Any] | None = None
