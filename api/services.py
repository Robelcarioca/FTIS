"""Service adapters for the FTIS API layer."""

from __future__ import annotations

from typing import Any

from ftis.inference import PredictionService
from ftis.routes.simulator import route_simulator
from ftis.weather.weather_service import weather_service


service = PredictionService()


def predict_turbulence(payload: dict[str, float]) -> dict[str, Any]:
    """Run FTIS model inference and return an API-safe dictionary."""

    return service.predict(payload).as_dict()


def predict_batch(payloads: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Run shared inference over a bounded batch of payloads."""

    return [predict_turbulence(payload) for payload in payloads]
