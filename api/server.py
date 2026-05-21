"""FastAPI service for FTIS real-time turbulence inference."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException

from api.schemas import FlightFeatures
from models.predict import PredictionError, predict_turbulence


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FTIS API",
    description="Flight Turbulence Intelligence System inference API",
    version="1.0.0",
)


def _payload_to_dict(payload: FlightFeatures) -> dict[str, Any]:
    """Support both Pydantic v1 and v2 serialization APIs."""

    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


@app.get("/")
def root() -> dict[str, str]:
    """API service health response."""

    return {
        "service": "FTIS API",
        "status": "running",
    }


@app.post("/predict")
def predict(payload: FlightFeatures) -> dict[str, Any]:
    """Predict turbulence severity from a validated flight feature payload."""

    try:
        result = predict_turbulence(_payload_to_dict(payload))
    except FileNotFoundError as exc:
        logger.warning("Prediction requested before model artifacts were available: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("Invalid prediction payload: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PredictionError as exc:
        logger.exception("FTIS prediction engine failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected FTIS API prediction failure")
        raise HTTPException(status_code=500, detail="Unexpected prediction failure") from exc

    logger.info(
        "Prediction completed turbulence=%s confidence=%.4f",
        result["prediction"],
        result["confidence"],
    )
    return result
