"""Normalization helpers for aviation weather providers."""

from __future__ import annotations

import math
import re
from typing import Any

from ftis.weather.weather_models import CloudLayer, WeatherCondition


def as_float(value: Any, default: float | None = None) -> float | None:
    """Convert provider values to finite floats."""

    if value in (None, "", "M", "NA", "N/A"):
        return default
    try:
        parsed = float(str(value).replace("+", "").strip())
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def clamp(value: float | None, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    return max(lower, min(upper, value))


def knots_to_kmh(value: float | None) -> float | None:
    return None if value is None else value * 1.852


def miles_to_meters(value: float | None) -> float | None:
    return None if value is None else value * 1609.344


def feet_to_meters(value: float | None) -> float | None:
    return None if value is None else value * 0.3048


def inches_hg_to_hpa(value: float | None) -> float | None:
    return None if value is None else value * 33.8638866667


def normalize_pressure(value: Any) -> float | None:
    """Return pressure in hPa from common METAR/API formats."""

    pressure = as_float(value)
    if pressure is None:
        return None
    if 20.0 <= pressure <= 35.0:
        return inches_hg_to_hpa(pressure)
    return pressure


def parse_visibility(value: Any) -> float | None:
    """Return visibility in meters from numeric, statute-mile, or plus formats."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = as_float(value)
        if numeric is None:
            return None
        return miles_to_meters(numeric) if numeric <= 100 else numeric

    text = str(value).strip().upper().replace("SM", "")
    if text.endswith("+"):
        text = text[:-1]
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        denom = as_float(denominator)
        if denom:
            return miles_to_meters((as_float(numerator) or 0.0) / denom)
    return miles_to_meters(as_float(text)) if as_float(text) is not None else None


def cloud_layers_from_awc(payload: dict[str, Any]) -> list[CloudLayer]:
    """Normalize AviationWeather cloud layer payloads."""

    layers: list[CloudLayer] = []
    raw_layers = payload.get("clouds") or payload.get("cloudLayers") or []
    if isinstance(raw_layers, dict):
        raw_layers = [raw_layers]

    for layer in raw_layers:
        if not isinstance(layer, dict):
            continue
        cover = str(
            layer.get("cover")
            or layer.get("coverCode")
            or layer.get("type")
            or "UNK"
        )
        base = (
            as_float(layer.get("base"))
            or as_float(layer.get("base_ft_agl"))
            or as_float(layer.get("baseFeet"))
        )
        layers.append(CloudLayer(cover=cover, base_m=feet_to_meters(base)))
    return layers


def open_meteo_cloud_layers(current: dict[str, Any]) -> list[CloudLayer]:
    """Create approximate low/mid/high cloud layers from Open-Meteo cover fields."""

    bands = [
        ("LOW", "cloud_cover_low", 0.0, 3000.0),
        ("MID", "cloud_cover_mid", 3000.0, 8000.0),
        ("HIGH", "cloud_cover_high", 8000.0, 13000.0),
    ]
    layers: list[CloudLayer] = []
    for label, key, base, top in bands:
        cover = as_float(current.get(key))
        if cover is not None and cover >= 15.0:
            layers.append(CloudLayer(cover=f"{label}:{cover:.0f}%", base_m=base, top_m=top))
    return layers


def turbulence_from_pireps(payload: list[dict[str, Any]]) -> float | None:
    """Estimate a turbulence severity score from recent PIREP/AIREP text."""

    if not payload:
        return None

    severity = 0.0
    for report in payload:
        text = " ".join(str(value) for value in report.values()).upper()
        if re.search(r"\b(SEV|SEVERE|EXTREME)\b", text):
            severity = max(severity, 95.0)
        elif re.search(r"\b(MOD|MODERATE)\b", text):
            severity = max(severity, 70.0)
        elif re.search(r"\b(LGT|LIGHT)\b", text):
            severity = max(severity, 42.0)
        elif "TB" in text or "TURB" in text:
            severity = max(severity, 55.0)
    return severity or None


def estimate_turbulence_indicator(condition: WeatherCondition) -> float:
    """Create a transparent 0-100 turbulence proxy from normalized weather."""

    wind = clamp((condition.wind_speed_kmh or 0.0) / 95.0, 0.0, 1.0) or 0.0
    jet = clamp((condition.jet_stream_indicator or 0.0) / 150.0, 0.0, 1.0) or 0.0
    cloud = clamp(len(condition.cloud_layers) / 4.0, 0.0, 1.0) or 0.0
    humidity = clamp(abs((condition.humidity_percent or 50.0) - 55.0) / 55.0, 0.0, 1.0)
    pressure = clamp(abs((condition.pressure_hpa or 1013.25) - 1013.25) / 35.0, 0.0, 1.0)
    visibility_penalty = 0.0
    if condition.visibility_m is not None:
        visibility_penalty = 1.0 - clamp(condition.visibility_m / 12_000.0, 0.0, 1.0)
    score = (
        100.0
        * (
            0.34 * wind
            + 0.26 * jet
            + 0.16 * cloud
            + 0.12 * (humidity or 0.0)
            + 0.08 * (pressure or 0.0)
            + 0.04 * visibility_penalty
        )
    )
    return round(float(clamp(score, 0.0, 100.0) or 0.0), 2)
