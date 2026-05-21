"""Prediction engine for FTIS turbulence inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "turbulence_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "saved" / "label_encoder.pkl"

FEATURE_COLUMNS = [
    "altitude",
    "velocity",
    "heading",
    "temperature",
    "windspeed",
    "winddirection",
    "pressure",
    "humidity",
    "vertical_speed",
    "turn_rate",
    "speed_variation",
    "altitude_variation",
    "pressure_variation",
    "wind_shear_proxy",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_ARTIFACT_CACHE: dict[str, Any] = {}


class PredictionError(RuntimeError):
    """Raised when FTIS inference cannot complete."""


def load_artifacts(
    model_path: Path = MODEL_PATH,
    encoder_path: Path = ENCODER_PATH,
) -> tuple[Any, Any]:
    """Load persisted model and encoder artifacts."""

    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            "FTIS model artifacts are missing. Run "
            "`python models/train_model.py` from the FTIS project root first."
        )

    model_mtime = model_path.stat().st_mtime
    encoder_mtime = encoder_path.stat().st_mtime
    cache_key = f"{model_path}:{encoder_path}"
    cached = _ARTIFACT_CACHE.get(cache_key)

    if cached and cached["mtimes"] == (model_mtime, encoder_mtime):
        return cached["model_artifact"], cached["label_encoder"]

    try:
        model_artifact = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
    except Exception as exc:
        raise PredictionError("Unable to load FTIS model artifacts") from exc

    _ARTIFACT_CACHE[cache_key] = {
        "mtimes": (model_mtime, encoder_mtime),
        "model_artifact": model_artifact,
        "label_encoder": label_encoder,
    }
    logger.info("Loaded FTIS model artifacts from %s and %s", model_path, encoder_path)
    return model_artifact, label_encoder


def _feature_frame(features: dict[str, Any]) -> pd.DataFrame:
    """Validate and order a single inference payload."""

    missing_features = [column for column in FEATURE_COLUMNS if column not in features]
    if missing_features:
        raise ValueError("Missing required features: " + ", ".join(missing_features))

    ordered_features: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        value = features[column]
        if value is None:
            raise ValueError(f"Feature {column} cannot be null")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {column} must be numeric") from exc

        if not np.isfinite(numeric_value):
            raise ValueError(f"Feature {column} must be finite")

        ordered_features[column] = numeric_value

    return pd.DataFrame([ordered_features], columns=FEATURE_COLUMNS)


def _predict_probabilities(
    model_artifact: Any,
    features: pd.DataFrame,
    class_count: int,
) -> np.ndarray:
    """Return probabilities aligned to LOW/MODERATE/HIGH label indices."""

    if isinstance(model_artifact, dict):
        model_type = model_artifact.get("model_type")

        if model_type == "constant":
            probabilities = np.zeros((len(features), class_count), dtype=float)
            probabilities[:, int(model_artifact["class_index"])] = 1.0
            return probabilities

        if model_type == "xgboost":
            estimator = model_artifact["estimator"]
            raw_probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
            probabilities = np.zeros((len(features), class_count), dtype=float)
            training_label_indices = model_artifact.get(
                "training_label_indices",
                list(range(class_count)),
            )

            if raw_probabilities.ndim == 1:
                raw_probabilities = raw_probabilities.reshape(-1, 1)

            for compact_index, label_index in enumerate(training_label_indices):
                if compact_index < raw_probabilities.shape[1]:
                    probabilities[:, int(label_index)] = raw_probabilities[:, compact_index]

            row_sums = probabilities.sum(axis=1, keepdims=True)
            return np.divide(
                probabilities,
                row_sums,
                out=np.zeros_like(probabilities),
                where=row_sums != 0,
            )

        raise PredictionError(f"Unsupported FTIS model artifact type: {model_type}")

    raw_probabilities = np.asarray(model_artifact.predict_proba(features), dtype=float)

    if raw_probabilities.shape[1] == class_count:
        return raw_probabilities

    probabilities = np.zeros((len(features), class_count), dtype=float)
    probabilities[:, : raw_probabilities.shape[1]] = raw_probabilities
    return probabilities


def predict_turbulence(features: dict[str, Any]) -> dict[str, Any]:
    """Predict turbulence severity for a single flight/weather feature payload."""

    model_artifact, label_encoder = load_artifacts()
    feature_frame = _feature_frame(features)
    probabilities = _predict_probabilities(
        model_artifact,
        feature_frame,
        class_count=len(label_encoder.classes_),
    )[0]

    prediction_index = int(np.argmax(probabilities))
    prediction_label = str(label_encoder.inverse_transform([prediction_index])[0])
    confidence = float(probabilities[prediction_index])

    probability_map = {
        str(label): round(float(probabilities[index]), 6)
        for index, label in enumerate(label_encoder.classes_)
    }

    result = {
        "prediction": prediction_label,
        "confidence": round(confidence, 6),
        "probabilities": probability_map,
    }
    logger.info(
        "Predicted turbulence=%s confidence=%.4f",
        result["prediction"],
        result["confidence"],
    )
    return result


if __name__ == "__main__":
    sample_features = {
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
    print(json.dumps(predict_turbulence(sample_features), indent=2))
