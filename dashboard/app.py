"""FTIS Streamlit mission-control dashboard.

Frontend-only architecture:
- No local model loading.
- All turbulence predictions are requested from the configured FastAPI backend.
- Ethiopian Airlines station context is maintained in this UI layer.
"""

from __future__ import annotations

import math
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


DEVELOPMENT_BACKEND_URL = "http://localhost:8000"
BACKEND_URL_ENV = "BACKEND_URL"
LEGACY_BACKEND_URL_ENV = "FTIS_BACKEND_URL"

logging.basicConfig(
    level=os.getenv("FTIS_DASHBOARD_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ftis.dashboard")


AIRCRAFT_ICON = """
<svg class="aircraft-icon" viewBox="0 0 96 96" aria-label="Aircraft icon" role="img">
  <path d="M84.4 45.2 55.1 35.1 43.3 8.8c-.7-1.7-2.4-2.8-4.2-2.8h-5.6l5.7 30.4-18.8 7.1-8.1-7.6H3.6l9.7 12.1-9.7 12.1h14.7l8.1-7.6 18.8 7.1-5.7 30.4h5.6c1.8 0 3.5-1.1 4.2-2.8l11.8-26.3 29.3-10.1c2-.7 3.4-2.6 3.4-4.8s-1.4-4.1-3.4-4.8Z"/>
</svg>
"""


@dataclass(frozen=True)
class Station:
    name: str
    code: str
    latitude: float
    longitude: float
    station_type: str

    @property
    def label(self) -> str:
        tag = "MAIN HUB" if self.station_type == "hub" else self.station_type.upper()
        return f"{self.code} - {self.name} [{tag}]"


@dataclass
class BackendStatus:
    status: str = "OFFLINE"
    version: str = "N/A"
    model_version: str = "N/A"
    latency_ms: float | None = None
    backend_url: str = DEVELOPMENT_BACKEND_URL
    health: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "version": self.version,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "backend_url": self.backend_url,
            "health": self.health,
            "system": self.system,
            "model": self.model,
            "errors": self.errors,
        }


STATIONS: dict[str, Station] = {
    "ADD": Station("Addis Ababa Bole International Airport", "ADD", 8.9779, 38.7993, "hub"),
    "DIR": Station("Dire Dawa Aba Tenna Dejazmach Yilma Airport", "DIR", 9.6247, 41.8542, "domestic"),
    "BJR": Station("Bahir Dar Airport", "BJR", 11.6081, 37.3216, "domestic"),
    "MQX": Station("Mekelle Alula Aba Nega Airport", "MQX", 13.4674, 39.5335, "domestic"),
    "JIM": Station("Jimma Aba Jifar Airport", "JIM", 7.6661, 36.8166, "domestic"),
    "GDQ": Station("Gondar Atse Tewodros Airport", "GDQ", 12.5199, 37.4339, "domestic"),
    "AWA": Station("Hawassa Airport", "AWA", 7.0670, 38.5000, "domestic"),
    "DXB": Station("Dubai International Airport", "DXB", 25.2532, 55.3657, "international"),
    "NBO": Station("Jomo Kenyatta International Airport", "NBO", -1.3192, 36.9278, "international"),
    "JED": Station("King Abdulaziz International Airport", "JED", 21.6796, 39.1565, "international"),
    "FRA": Station("Frankfurt Airport", "FRA", 50.0379, 8.5622, "international"),
}


RISK_COLORS = {
    "LOW": "#22c55e",
    "MODERATE": "#facc15",
    "HIGH": "#ef4444",
    "UNKNOWN": "#94a3b8",
}


class PredictionClientError(RuntimeError):
    """Raised when the backend prediction service cannot return a valid result."""


def normalize_backend_url(raw_url: str | None) -> str:
    """Return a backend base URL, accepting either base URLs or /predict URLs."""

    candidate = (raw_url or "").strip() or DEVELOPMENT_BACKEND_URL
    if not candidate.startswith(("http://", "https://")):
        candidate = f"http://{candidate}"
    candidate = candidate.rstrip("/")
    if candidate.endswith("/predict"):
        candidate = candidate[: -len("/predict")]
    return candidate.rstrip("/")


def configured_backend_url() -> str:
    """Read BACKEND_URL from the environment with a local development fallback."""

    return normalize_backend_url(
        os.getenv(BACKEND_URL_ENV)
        or os.getenv(LEGACY_BACKEND_URL_ENV)
        or DEVELOPMENT_BACKEND_URL
    )


DEFAULT_BACKEND_URL = configured_backend_url()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def haversine_nm(origin: Station, destination: Station) -> float:
    radius_nm = 3440.065
    lat1, lon1 = math.radians(origin.latitude), math.radians(origin.longitude)
    lat2, lon2 = math.radians(destination.latitude), math.radians(destination.longitude)
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_nm * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def prediction_endpoint(backend_url: str) -> str:
    return f"{normalize_backend_url(backend_url)}/predict"


def endpoint_url(backend_url: str, path: str) -> str:
    return f"{normalize_backend_url(backend_url)}/{path.lstrip('/')}"


def backend_base_url(backend_url: str) -> str:
    parsed = urlparse(normalize_backend_url(backend_url))
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def backend_get_json(backend_url: str, path: str, *, timeout: float = 4.0) -> tuple[dict[str, Any], float]:
    url = endpoint_url(backend_url, path)
    started = time.perf_counter()
    response = requests.get(url, timeout=timeout)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise PredictionClientError(f"Backend returned non-JSON response from {path}") from exc
    return payload, latency_ms


def backend_post_json(
    backend_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float = 8.0,
) -> tuple[dict[str, Any], float, int]:
    url = endpoint_url(backend_url, path)
    started = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    status_code = response.status_code
    response.raise_for_status()
    try:
        data = response.json()
    except ValueError as exc:
        raise PredictionClientError(f"Backend returned non-JSON response from {path}") from exc
    return data, latency_ms, status_code


def system_status(backend_url: str) -> dict[str, Any]:
    base_url = backend_base_url(backend_url)
    if not base_url:
        return BackendStatus(
            status="OFFLINE",
            version="N/A",
            model_version="N/A",
            latency_ms=None,
            backend_url=base_url,
            errors=["Invalid BACKEND_URL"],
        ).as_dict()

    status = BackendStatus(backend_url=base_url)
    try:
        health, latency_ms = backend_get_json(base_url, "/health", timeout=3.0)
        status.health = health
        status.latency_ms = latency_ms
        status.status = "ONLINE"
        status.version = str(health.get("version", "2.0"))
        if not bool(health.get("model_available", True)):
            status.status = "DEGRADED"
            status.errors.append("Health endpoint reports model unavailable")
    except requests.RequestException as exc:
        logger.exception("Backend /health check failed backend_url=%s", base_url)
        status.errors.append(f"/health failed: {exc}")
        return status.as_dict()
    except PredictionClientError as exc:
        logger.exception("Backend /health response parse failed backend_url=%s", base_url)
        status.status = "DEGRADED"
        status.errors.append(str(exc))

    try:
        system_payload, system_latency = backend_get_json(base_url, "/system/status", timeout=4.0)
        status.system = system_payload
        status.latency_ms = min(
            value for value in [status.latency_ms, system_latency] if value is not None
        )
        raw_status = str(system_payload.get("status", "healthy")).upper()
        if raw_status not in {"HEALTHY", "ONLINE", "OK", "RUNNING"}:
            status.status = "DEGRADED"
    except Exception as exc:
        logger.exception("Backend /system/status check failed backend_url=%s", base_url)
        status.status = "DEGRADED"
        status.errors.append(f"/system/status failed: {exc}")

    try:
        model_payload, _ = backend_get_json(base_url, "/model/metrics", timeout=5.0)
        status.model = model_payload
        model = model_payload.get("model", {})
        status.model_version = str(
            model.get("artifact_version")
            or model.get("model_name")
            or status.health.get("model_version")
            or "active"
        )
    except Exception as exc:
        logger.exception("Backend /model/metrics check failed backend_url=%s", base_url)
        status.status = "DEGRADED"
        status.errors.append(f"/model/metrics failed: {exc}")
        status.model_version = str(status.health.get("model_version") or "active")

    return status.as_dict()


def station_by_label(label: str) -> Station:
    code = label.split(" - ", maxsplit=1)[0]
    return STATIONS[code]


def build_backend_payload(
    departure: Station,
    destination: Station,
    altitude: float,
    speed: float,
    temperature: float,
    windspeed: float,
) -> dict[str, Any]:
    return {
        "altitude": altitude,
        "speed": speed,
        "temperature": temperature,
        "windspeed": windspeed,
        "departure_station": departure.code,
        "destination_station": destination.code,
    }


def build_compatibility_payload(
    departure: Station,
    destination: Station,
    altitude: float,
    temperature: float,
    windspeed: float,
) -> dict[str, Any]:
    mid_lat = (departure.latitude + destination.latitude) / 2
    mid_lon = (departure.longitude + destination.longitude) / 2
    return {
        "latitude": mid_lat,
        "longitude": mid_lon,
        "altitude": altitude,
        "windspeed": windspeed,
        "pressure": 1013.25,
        "temperature": temperature,
    }


def normalize_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    risk_level = (
        payload.get("turbulence_level")
        or payload.get("risk_level")
        or payload.get("risk")
        or payload.get("prediction")
        or "UNKNOWN"
    )
    risk_level = str(risk_level).upper()
    if risk_level in {"LOW", "MODERATE", "HIGH"}:
        normalized_risk = risk_level
    elif risk_level.title() in {"Low", "Moderate", "High"}:
        normalized_risk = risk_level.upper()
    else:
        normalized_risk = "UNKNOWN"

    probability = payload.get("probability", payload.get("confidence", 0.0))
    confidence = payload.get("confidence", probability)
    risk_score = payload.get("risk_score", payload.get("FTI", payload.get("fti", 0.0)))

    try:
        probability = float(probability)
    except (TypeError, ValueError):
        probability = 0.0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = probability

    try:
        risk_score = float(risk_score)
    except (TypeError, ValueError):
        risk_score = 0.0

    return {
        "risk_level": normalized_risk,
        "probability": clamp(probability, 0.0, 1.0),
        "confidence": clamp(confidence, 0.0, 1.0),
        "risk_score": clamp(risk_score, 0.0, 100.0),
        "recommendation": payload.get("recommendation"),
        "probabilities": payload.get("probabilities", {}),
        "raw": payload,
    }


def call_prediction_backend(
    backend_url: str,
    departure: Station,
    destination: Station,
    altitude: float,
    speed: float,
    temperature: float,
    windspeed: float,
) -> tuple[dict[str, Any], float]:
    mission_payload = build_backend_payload(
        departure,
        destination,
        altitude,
        speed,
        temperature,
        windspeed,
    )
    started = time.perf_counter()
    predict_url = prediction_endpoint(backend_url)
    logger.info("Sending prediction request backend_url=%s endpoint=%s", normalize_backend_url(backend_url), predict_url)

    try:
        response = requests.post(predict_url, json=mission_payload, timeout=8)
        if response.status_code == 422:
            compatibility_payload = build_compatibility_payload(
                departure,
                destination,
                altitude,
                temperature,
                windspeed,
            )
            response = requests.post(predict_url, json=compatibility_payload, timeout=8)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("Backend prediction request failed endpoint=%s", predict_url)
        raise PredictionClientError(
            "Prediction backend is unavailable. Start the FastAPI backend or update BACKEND_URL."
        ) from exc

    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    try:
        payload = response.json()
    except ValueError as exc:
        logger.exception("Backend returned a non-JSON prediction response endpoint=%s", predict_url)
        raise PredictionClientError(
            "Prediction backend returned an invalid response. Check backend logs."
        ) from exc

    return normalize_prediction(payload), latency_ms


def call_route_analysis_backend(
    backend_url: str,
    departure: Station,
    destination: Station,
    altitude: float,
    speed: float,
) -> dict[str, Any] | None:
    """Fetch route intelligence from the backend without blocking prediction success."""

    payload = {
        "departure_airport": departure.code,
        "destination_airport": destination.code,
        "cruising_altitude_m": altitude,
        "aircraft_speed_kt": speed,
        "waypoint_count": 36,
        "use_live_weather": False,
    }
    try:
        data, _, _ = backend_post_json(backend_url, "/route/analyze", payload, timeout=12.0)
        return data
    except Exception as exc:
        logger.exception("Route analysis backend request failed")
        st.session_state["route_error"] = f"Route intelligence unavailable: {exc}"
        return None


def call_weather_backend(
    backend_url: str,
    departure: Station,
    destination: Station,
    altitude: float,
) -> dict[str, Any] | None:
    """Fetch midpoint live-weather intelligence from the backend."""

    payload = {
        "latitude": (departure.latitude + destination.latitude) / 2,
        "longitude": (departure.longitude + destination.longitude) / 2,
        "altitude_m": altitude,
    }
    try:
        data, _, _ = backend_post_json(backend_url, "/weather/live", payload, timeout=4.0)
        return data
    except Exception as exc:
        logger.exception("Weather intelligence backend request failed")
        st.session_state["weather_error"] = f"Live weather unavailable: {exc}"
        return None


def weather_from_route_analysis(route_analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    frame = route_analysis_frame(route_analysis)
    if frame.empty:
        return None
    midpoint = frame.iloc[len(frame) // 2]
    return {
        "provider": midpoint.get("provider", "route_weather"),
        "wind_speed_kmh": midpoint.get("wind_speed_kmh"),
        "temperature_c": midpoint.get("temperature_c"),
        "pressure_hpa": midpoint.get("pressure_hpa"),
        "turbulence_indicator": midpoint.get("turbulence_indicator") or midpoint.get("FTI"),
        "source": "route_analysis",
    }


def safe_route_recommendation(
    departure: Station,
    destination: Station,
    risk_level: str,
    confidence: float,
) -> dict[str, Any]:
    distance_nm = haversine_nm(departure, destination)
    route_direct = [departure.code, destination.code]

    if risk_level == "HIGH":
        intermediate = STATIONS["NBO"] if departure.code == "ADD" or destination.code == "ADD" else STATIONS["ADD"]
        recommended_route = [departure.code, intermediate.code, destination.code]
        explanation = (
            "High turbulence risk detected. Recommend routing through a safer intermediate "
            f"hub and requesting dispatch review before release. Estimated direct distance is {distance_nm:.0f} NM."
        )
        recommendation_confidence = clamp(confidence - 0.04, 0.68, 0.96)
    elif risk_level == "MODERATE":
        recommended_route = route_direct
        explanation = (
            "Moderate turbulence risk detected. Direct routing is acceptable with active "
            "weather monitoring, altitude flexibility, and crew advisory briefing."
        )
        recommendation_confidence = clamp(confidence, 0.62, 0.9)
    elif risk_level == "LOW":
        recommended_route = route_direct
        explanation = (
            "Low turbulence risk detected. Direct route is recommended with routine "
            "operational monitoring."
        )
        recommendation_confidence = clamp(confidence + 0.05, 0.72, 0.98)
    else:
        recommended_route = route_direct
        explanation = (
            "Backend risk level is unavailable. Keep route in monitored status until the "
            "prediction service returns an operational risk classification."
        )
        recommendation_confidence = 0.0

    return {
        "recommended_route": " -> ".join(recommended_route),
        "risk_level": risk_level,
        "explanation": explanation,
        "confidence_score": round(recommendation_confidence, 3),
        "distance_nm": round(distance_nm, 1),
    }


def route_coordinates(route_codes: list[str]) -> pd.DataFrame:
    rows = []
    for index, code in enumerate(route_codes):
        station = STATIONS[code]
        rows.append(
            {
                "sequence": index + 1,
                "code": station.code,
                "name": station.name,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "type": station.station_type,
            }
        )
    return pd.DataFrame(rows)


def route_figure(route_codes: list[str], risk_level: str) -> go.Figure:
    frame = route_coordinates(route_codes)
    color = RISK_COLORS.get(risk_level, RISK_COLORS["UNKNOWN"])
    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=frame["longitude"],
            lat=frame["latitude"],
            mode="lines+markers+text",
            text=frame["code"],
            textposition="top center",
            line=dict(width=3, color=color),
            marker=dict(size=12, color=color, line=dict(width=1, color="#e5edf7")),
            hovertext=frame["name"],
            hoverinfo="text",
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#111827",
        showocean=True,
        oceancolor="#050914",
        showcountries=True,
        countrycolor="#334155",
        lataxis_range=[-10, 55],
        lonaxis_range=[20, 65],
    )
    fig.update_layout(
        height=405,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5edf7"),
    )
    return fig


def route_analysis_frame(route_analysis: dict[str, Any] | None) -> pd.DataFrame:
    if not route_analysis:
        return pd.DataFrame()
    waypoints = (route_analysis.get("analysis") or {}).get("waypoints", [])
    if not waypoints:
        return pd.DataFrame()
    return pd.DataFrame(waypoints)


def operational_route_figure(
    route_codes: list[str],
    risk_level: str,
    route_analysis: dict[str, Any] | None,
) -> go.Figure:
    frame = route_analysis_frame(route_analysis)
    if frame.empty or not {"latitude", "longitude"}.issubset(frame.columns):
        return route_figure(route_codes, risk_level)

    risks = frame.get("risk", pd.Series(["UNKNOWN"] * len(frame))).astype(str).str.upper()
    colors = [RISK_COLORS.get(risk, RISK_COLORS["UNKNOWN"]) for risk in risks]
    fig = go.Figure()
    fig.add_trace(
        go.Scattergeo(
            lon=frame["longitude"],
            lat=frame["latitude"],
            mode="lines+markers",
            line=dict(width=3, color=RISK_COLORS.get(risk_level, "#38bdf8")),
            marker=dict(
                size=(frame.get("FTI", pd.Series([30] * len(frame))).astype(float) / 12 + 5),
                color=colors,
                line=dict(width=1, color="#e5edf7"),
            ),
            hovertext=[
                f"{row.get('waypoint_id', 'WP')} | {row.get('risk', 'UNKNOWN')} | FTI {float(row.get('FTI', 0)):.1f}"
                for _, row in frame.iterrows()
            ],
            hoverinfo="text",
        )
    )
    for code in route_codes:
        station = STATIONS.get(code)
        if station:
            fig.add_trace(
                go.Scattergeo(
                    lon=[station.longitude],
                    lat=[station.latitude],
                    mode="markers+text",
                    text=[station.code],
                    textposition="top center",
                    marker=dict(size=14, color="#38bdf8", symbol="star"),
                    hovertext=[station.name],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#111827",
        showocean=True,
        oceancolor="#050914",
        showcountries=True,
        countrycolor="#334155",
        lataxis_range=[-10, 55],
        lonaxis_range=[20, 65],
    )
    fig.update_layout(
        height=405,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5edf7"),
        showlegend=False,
    )
    return fig


def route_profile_figure(route_analysis: dict[str, Any] | None) -> go.Figure:
    frame = route_analysis_frame(route_analysis)
    fig = go.Figure()
    if frame.empty:
        fig.add_annotation(
            text="Route intelligence awaiting mission scan",
            showarrow=False,
            font=dict(color="#94a3b8", size=14),
        )
    else:
        x_values = frame.get("distance_from_origin_nm", pd.Series(range(len(frame))))
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame.get("FTI", pd.Series([0] * len(frame))),
                mode="lines+markers",
                name="FTI",
                line=dict(color="#f97316", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=frame.get("wind_speed_kmh", pd.Series([0] * len(frame))),
                mode="lines",
                name="Wind km/h",
                line=dict(color="#38bdf8", width=2),
                yaxis="y2",
            )
        )
    fig.add_hrect(y0=66, y1=100, fillcolor="rgba(239,68,68,0.10)", line_width=0)
    fig.add_hrect(y0=33, y1=66, fillcolor="rgba(250,204,21,0.08)", line_width=0)
    fig.update_layout(
        height=260,
        margin=dict(l=12, r=12, t=20, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5edf7"),
        yaxis=dict(title="FTI", range=[0, 100], gridcolor="rgba(148,163,184,0.16)"),
        yaxis2=dict(title="Wind", overlaying="y", side="right", showgrid=False),
        xaxis=dict(title="Distance (NM)", gridcolor="rgba(148,163,184,0.08)"),
        legend=dict(orientation="h", y=1.16),
    )
    return fig


def risk_trend_figure(history: list[dict[str, Any]]) -> go.Figure:
    if not history:
        history = [{"time": datetime.now().strftime("%H:%M:%S"), "risk_score": 0, "risk_level": "UNKNOWN"}]
    frame = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["time"],
            y=frame["risk_score"],
            mode="lines+markers",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=8, color=[RISK_COLORS.get(r, "#94a3b8") for r in frame["risk_level"]]),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
        )
    )
    fig.add_hrect(y0=66, y1=100, fillcolor="rgba(239,68,68,0.12)", line_width=0)
    fig.add_hrect(y0=33, y1=66, fillcolor="rgba(250,204,21,0.10)", line_width=0)
    fig.update_layout(
        height=260,
        margin=dict(l=12, r=12, t=20, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5edf7"),
        yaxis=dict(range=[0, 100], title="Risk score", gridcolor="rgba(148,163,184,0.16)"),
        xaxis=dict(title="Mission time", gridcolor="rgba(148,163,184,0.08)"),
    )
    return fig


def metric_card(label: str, value: str, color: str = "#e5edf7") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color};">{value}</div>
    </div>
    """


def apply_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0B0F19;
            --panel: rgba(15, 23, 42, 0.68);
            --stroke: rgba(148, 163, 184, 0.22);
            --cyan: #38bdf8;
            --green: #22c55e;
            --red: #ef4444;
            --text: #e5edf7;
            --muted: #94a3b8;
        }
        body, .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(56,189,248,0.10), transparent 28%),
                radial-gradient(circle at 80% 10%, rgba(34,197,94,0.08), transparent 26%),
                #0B0F19;
            color: var(--text);
        }
        .block-container {
            padding-top: 1.1rem;
            max-width: 1480px;
        }
        [data-testid="stSidebar"] {
            background: #070b13;
            border-right: 1px solid var(--stroke);
        }
        .top-bar, .glass-panel, .metric-card {
            background: var(--panel);
            border: 1px solid var(--stroke);
            box-shadow: 0 18px 48px rgba(0, 0, 0, 0.30);
            backdrop-filter: blur(14px);
            border-radius: 12px;
        }
        .top-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.95rem 1rem;
            margin-bottom: 0.9rem;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .aircraft-icon {
            width: 42px;
            height: 42px;
            fill: #38bdf8;
            filter: drop-shadow(0 0 14px rgba(56,189,248,0.45));
        }
        .title {
            color: #f8fafc;
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }
        .subtitle {
            color: var(--muted);
            font-size: 0.78rem;
            letter-spacing: 0.04em;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(105px, 1fr));
            gap: 0.55rem;
        }
        .status-item {
            border-left: 2px solid var(--cyan);
            padding-left: 0.65rem;
        }
        .status-label, .metric-label {
            color: var(--muted);
            font-size: 0.70rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .status-value {
            color: #f8fafc;
            font-weight: 800;
            font-size: 0.9rem;
            margin-top: 0.12rem;
        }
        .glass-panel {
            padding: 1rem;
            min-height: 100%;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin-bottom: 0.9rem;
        }
        .panel-title {
            color: #f8fafc;
            font-weight: 800;
            letter-spacing: 0.06em;
            margin-bottom: 0.7rem;
            text-transform: uppercase;
        }
        .metric-card {
            padding: 0.85rem;
            margin-bottom: 0.65rem;
        }
        .metric-value {
            font-size: 1.25rem;
            font-weight: 800;
            margin-top: 0.15rem;
        }
        .route-strip {
            color: #e5edf7;
            border: 1px solid rgba(56,189,248,0.28);
            border-radius: 10px;
            padding: 0.85rem;
            background: rgba(8,13,24,0.82);
            font-weight: 800;
            letter-spacing: 0.08em;
            text-align: center;
        }
        .explanation {
            color: #dbeafe;
            line-height: 1.55;
            border-left: 3px solid #38bdf8;
            padding-left: 0.85rem;
            margin-top: 0.65rem;
        }
        div[data-testid="stButton"] > button {
            background: linear-gradient(90deg, #0891b2, #22c55e);
            color: #04111f;
            border: 0;
            font-weight: 900;
            letter-spacing: 0.04em;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("last_recommendation", None)
    st.session_state.setdefault("route_analysis", None)
    st.session_state.setdefault("weather_snapshot", None)
    st.session_state.setdefault("api_latency_ms", None)
    st.session_state.setdefault("risk_history", [])
    st.session_state.setdefault("backend_error", None)
    st.session_state.setdefault("route_error", None)
    st.session_state.setdefault("weather_error", None)


def render_top_bar(status: dict[str, Any]) -> None:
    api_latency = st.session_state.get("api_latency_ms") or status.get("latency_ms")
    latency_label = f"{api_latency:.1f} ms" if api_latency is not None else "STANDBY"
    status_text = status.get("status", "DEGRADED")
    status_color = (
        "#22c55e"
        if status_text == "ONLINE"
        else "#facc15"
        if status_text == "DEGRADED"
        else "#ef4444"
    )
    st.markdown(
        f"""
        <div class="top-bar">
          <div class="brand">
            {AIRCRAFT_ICON}
            <div>
              <div class="title">FTIS CONTROL CENTER</div>
              <div class="subtitle">ETHIOPIAN AIRLINES TURBULENCE INTELLIGENCE NETWORK</div>
            </div>
          </div>
          <div class="status-grid">
            <div class="status-item">
              <div class="status-label">System</div>
              <div class="status-value" style="color:{status_color};">{status_text}</div>
            </div>
            <div class="status-item">
              <div class="status-label">API Latency</div>
              <div class="status-value">{latency_label}</div>
            </div>
            <div class="status-item">
              <div class="status-label">Model Version</div>
              <div class="status-value">{status.get("model_version", "active")}</div>
            </div>
            <div class="status-item">
              <div class="status-label">Console</div>
              <div class="status-value">MISSION READY</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="FTIS Control Center",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_css()
    initialize_state()

    station_labels = [station.label for station in STATIONS.values()]
    default_departure = STATIONS["ADD"].label
    default_destination = STATIONS["DXB"].label

    with st.sidebar:
        st.header("Backend")
        backend_input = st.text_input("BACKEND_URL", DEFAULT_BACKEND_URL)
        backend_url = normalize_backend_url(backend_input)
        logger.info("Active FTIS backend URL: %s", backend_url)
        st.caption("Set BACKEND_URL in the environment for deployment.")
        st.code(backend_url, language="text")

    status = system_status(backend_url)
    render_top_bar(status)
    health_color = (
        RISK_COLORS["LOW"]
        if status["status"] == "ONLINE"
        else RISK_COLORS["MODERATE"]
        if status["status"] == "DEGRADED"
        else RISK_COLORS["HIGH"]
    )
    st.markdown(
        '<div class="metric-grid">'
        + metric_card("Backend", status["status"], health_color)
        + metric_card("Active URL", status["backend_url"], "#38bdf8")
        + metric_card(
            "Health Latency",
            f"{status['latency_ms']:.1f} ms" if status.get("latency_ms") is not None else "No signal",
            "#e5edf7",
        )
        + metric_card("Model Version", status.get("model_version", "active"), "#e5edf7")
        + "</div>",
        unsafe_allow_html=True,
    )
    if status.get("errors"):
        st.warning("Backend partial service warning: " + " | ".join(status["errors"][:2]))

    col_input, col_prediction, col_recommendation = st.columns([1.05, 1.05, 1.2])

    with col_input:
        st.markdown('<div class="glass-panel"><div class="panel-title">Flight Input Panel</div>', unsafe_allow_html=True)
        departure_label = st.selectbox(
            "Departure Station",
            station_labels,
            index=station_labels.index(default_departure),
        )
        destination_label = st.selectbox(
            "Destination Station",
            station_labels,
            index=station_labels.index(default_destination),
        )
        departure = station_by_label(departure_label)
        destination = station_by_label(destination_label)
        altitude = st.slider("Cruising Altitude", 1500, 13000, 10600, 100)
        speed = st.slider("Aircraft Speed", 180, 560, 455, 5)
        temperature = st.slider("Outside Air Temperature", -70, 45, -38, 1)
        windspeed = st.slider("Wind Speed", 0, 160, 42, 1)
        st.markdown(
            f'<div class="route-strip">{departure.code} -> {destination.code}</div>',
            unsafe_allow_html=True,
        )
        run_prediction = st.button("EXECUTE TURBULENCE SCAN", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if run_prediction:
        if departure.code == destination.code:
            st.session_state["backend_error"] = "Departure and destination stations must be different."
        else:
            try:
                st.session_state["route_error"] = None
                st.session_state["weather_error"] = None
                prediction, latency_ms = call_prediction_backend(
                    backend_url,
                    departure,
                    destination,
                    altitude,
                    speed,
                    temperature,
                    windspeed,
                )
                recommendation = safe_route_recommendation(
                    departure,
                    destination,
                    prediction["risk_level"],
                    prediction["confidence"],
                )
                route_analysis = call_route_analysis_backend(
                    backend_url,
                    departure,
                    destination,
                    altitude,
                    speed,
                )
                weather_snapshot = call_weather_backend(
                    backend_url,
                    departure,
                    destination,
                    altitude,
                ) or weather_from_route_analysis(route_analysis)
                st.session_state["last_prediction"] = prediction
                st.session_state["last_recommendation"] = recommendation
                st.session_state["route_analysis"] = route_analysis
                st.session_state["weather_snapshot"] = weather_snapshot
                st.session_state["api_latency_ms"] = latency_ms
                st.session_state["backend_error"] = None
                st.session_state["risk_history"].append(
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "risk_score": prediction["risk_score"],
                        "risk_level": prediction["risk_level"],
                    }
                )
                st.session_state["risk_history"] = st.session_state["risk_history"][-20:]
                st.rerun()
            except PredictionClientError as exc:
                st.session_state["backend_error"] = str(exc)
                st.session_state["last_prediction"] = None
                st.session_state["last_recommendation"] = None
                st.session_state["route_analysis"] = None
                st.session_state["weather_snapshot"] = None

    prediction = st.session_state.get("last_prediction")
    recommendation = st.session_state.get("last_recommendation")
    route_analysis = st.session_state.get("route_analysis")
    weather_snapshot = st.session_state.get("weather_snapshot")
    backend_error = st.session_state.get("backend_error")

    with col_prediction:
        st.markdown('<div class="glass-panel"><div class="panel-title">Turbulence Prediction Output</div>', unsafe_allow_html=True)
        if backend_error:
            st.error(backend_error)
            st.markdown(metric_card("Risk Level", "UNAVAILABLE", RISK_COLORS["UNKNOWN"]), unsafe_allow_html=True)
            st.markdown(metric_card("Probability", "0.0%", RISK_COLORS["UNKNOWN"]), unsafe_allow_html=True)
            st.markdown(metric_card("Risk Score", "0.0 / 100", RISK_COLORS["UNKNOWN"]), unsafe_allow_html=True)
        elif prediction:
            risk = prediction["risk_level"]
            color = RISK_COLORS.get(risk, RISK_COLORS["UNKNOWN"])
            st.markdown(metric_card("Risk Level", risk, color), unsafe_allow_html=True)
            st.markdown(
                metric_card("Probability", f"{prediction['probability'] * 100:.1f}%", "#38bdf8"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("Confidence", f"{prediction['confidence'] * 100:.1f}%", "#22c55e"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("Risk Score", f"{prediction['risk_score']:.1f} / 100", color),
                unsafe_allow_html=True,
            )
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction["risk_score"],
                    title={"text": "Mission Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 33], "color": "rgba(34,197,94,0.20)"},
                            {"range": [33, 66], "color": "rgba(250,204,21,0.20)"},
                            {"range": [66, 100], "color": "rgba(239,68,68,0.20)"},
                        ],
                    },
                )
            )
            gauge.update_layout(
                height=245,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5edf7"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.markdown(metric_card("Risk Level", "STANDBY", RISK_COLORS["UNKNOWN"]), unsafe_allow_html=True)
            st.markdown(metric_card("Probability", "Awaiting scan", "#38bdf8"), unsafe_allow_html=True)
            st.markdown(metric_card("Risk Score", "Awaiting scan", "#38bdf8"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_recommendation:
        st.markdown('<div class="glass-panel"><div class="panel-title">Route Recommendation + Telemetry</div>', unsafe_allow_html=True)
        if recommendation:
            risk = recommendation["risk_level"]
            color = RISK_COLORS.get(risk, RISK_COLORS["UNKNOWN"])
            st.markdown(
                metric_card("Recommended Route", recommendation["recommended_route"], color),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("Confidence Score", f"{recommendation['confidence_score'] * 100:.1f}%", "#38bdf8"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("Station Distance", f"{recommendation['distance_nm']:.1f} NM", "#e5edf7"),
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="explanation">{recommendation["explanation"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            direct_distance = haversine_nm(departure, destination)
            st.markdown(metric_card("Recommended Route", f"{departure.code} -> {destination.code}", "#38bdf8"), unsafe_allow_html=True)
            st.markdown(metric_card("Confidence Score", "Awaiting backend", "#94a3b8"), unsafe_allow_html=True)
            st.markdown(metric_card("Station Distance", f"{direct_distance:.1f} NM", "#e5edf7"), unsafe_allow_html=True)
            st.markdown(
                '<div class="explanation">Execute a turbulence scan to receive route safety guidance from the backend model.</div>',
                unsafe_allow_html=True,
            )
        telemetry = pd.DataFrame(
            [
                {"Channel": "Backend", "Value": status["status"]},
                {"Channel": "Health Endpoint", "Value": "/health OK" if status.get("health") else "No signal"},
                {"Channel": "Prediction Endpoint", "Value": "/predict"},
                {"Channel": "Route Endpoint", "Value": "/route/analyze" if route_analysis else st.session_state.get("route_error", "Standby")},
                {"Channel": "Weather Feed", "Value": weather_snapshot.get("provider", "Standby") if weather_snapshot else st.session_state.get("weather_error", "Standby")},
                {"Channel": "Station Pair", "Value": f"{departure.code}-{destination.code}"},
                {"Channel": "Aircraft Profile", "Value": f"{speed} kt / {altitude} m"},
                {"Channel": "Last Update", "Value": datetime.now().strftime("%H:%M:%S")},
            ]
        )
        st.dataframe(telemetry, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if recommendation:
        route_codes = recommendation["recommended_route"].split(" -> ")
        route_risk = recommendation["risk_level"]
    else:
        route_codes = [departure.code, destination.code]
        route_risk = "UNKNOWN"

    bottom_left, bottom_right = st.columns([1.45, 1])
    with bottom_left:
        st.markdown('<div class="glass-panel"><div class="panel-title">Flight Route Visualization</div>', unsafe_allow_html=True)
        st.plotly_chart(
            operational_route_figure(route_codes, route_risk, route_analysis),
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with bottom_right:
        st.markdown('<div class="glass-panel"><div class="panel-title">Risk Trend Chart</div>', unsafe_allow_html=True)
        st.plotly_chart(risk_trend_figure(st.session_state["risk_history"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    intel_left, intel_right = st.columns([1.25, 1])
    with intel_left:
        st.markdown('<div class="glass-panel"><div class="panel-title">Turbulence Corridor Intelligence</div>', unsafe_allow_html=True)
        st.plotly_chart(route_profile_figure(route_analysis), use_container_width=True)
        if route_analysis:
            analysis = route_analysis.get("analysis", {})
            corridor_count = len(analysis.get("turbulence_corridors", []))
            st.markdown(
                '<div class="metric-grid">'
                + metric_card("Route Risk", str(analysis.get("route_risk", "UNKNOWN")).upper(), RISK_COLORS.get(str(analysis.get("route_risk", "UNKNOWN")).upper(), "#94a3b8"))
                + metric_card("Cumulative FTI", f"{float(analysis.get('cumulative_fti', 0)):.1f}", "#f97316")
                + metric_card("Distance", f"{float(analysis.get('route_distance_nm', 0)):.0f} NM", "#e5edf7")
                + metric_card("Corridors", str(corridor_count), "#38bdf8")
                + "</div>",
                unsafe_allow_html=True,
            )
        elif st.session_state.get("route_error"):
            st.warning(st.session_state["route_error"])
        st.markdown("</div>", unsafe_allow_html=True)

    with intel_right:
        st.markdown('<div class="glass-panel"><div class="panel-title">Weather Integration Display</div>', unsafe_allow_html=True)
        if weather_snapshot:
            st.markdown(
                metric_card("Provider", str(weather_snapshot.get("provider", "UNKNOWN")), "#38bdf8"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("Wind", f"{float(weather_snapshot.get('wind_speed_kmh') or 0):.1f} km/h", "#e5edf7"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("Temperature", f"{float(weather_snapshot.get('temperature_c') or 0):.1f} C", "#e5edf7"),
                unsafe_allow_html=True,
            )
            st.markdown(
                metric_card("Turbulence Indicator", f"{float(weather_snapshot.get('turbulence_indicator') or 0):.1f}", "#f97316"),
                unsafe_allow_html=True,
            )
        elif st.session_state.get("weather_error"):
            st.warning(st.session_state["weather_error"])
        else:
            st.markdown(metric_card("Provider", "Standby", "#94a3b8"), unsafe_allow_html=True)
            st.markdown(metric_card("Wind", "Awaiting scan", "#94a3b8"), unsafe_allow_html=True)
            st.markdown(metric_card("Turbulence Indicator", "Awaiting scan", "#94a3b8"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
