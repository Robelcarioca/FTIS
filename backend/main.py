"""Lightweight FTIS backend used for local classroom demos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException


logger = logging.getLogger(__name__)
app = FastAPI(title="FTIS Classroom Backend")

MODEL_PATH = Path("models") / "ftis_model.pkl"
model: Any | None = None
model_error: str | None = None

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded FTIS classroom model path=%s", MODEL_PATH)
except Exception as exc:
    model_error = str(exc)
    logger.exception("Unable to load FTIS classroom model path=%s", MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return backend health for Streamlit status checks."""

    return {
        "status": "healthy" if model is not None else "degraded",
        "model_available": model is not None,
        "model_path": str(MODEL_PATH),
        "model_error": model_error,
        "version": "classroom-1.0",
    }


@app.post("/predict")
def predict(data: dict[str, Any]) -> dict[str, Any]:
    """Predict turbulence from the classroom payload."""

    if model is None:
        raise HTTPException(status_code=503, detail="Model is unavailable")

    try:
        input_data = [[data["altitude"], data["speed"], data["temperature"]]]
        result = model.predict(input_data)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"Missing field: {exc.args[0]}") from exc
    except Exception as exc:
        logger.exception("Classroom backend prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    prediction = result.tolist()
    risk = prediction[0] if prediction else "UNKNOWN"
    return {
        "prediction": risk,
        "turbulence_level": str(risk).upper(),
        "probability": 1.0,
        "risk_score": 50.0,
    }
