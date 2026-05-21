"""Prediction service for FTIS API and dashboard consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ftis.config import MODEL_PATH, RECOMMENDATIONS
from ftis.features import engineer_turbulence_features, features_from_prediction_payload
from ftis.modeling import load_artifact


@dataclass
class PredictionResult:
    """Dashboard and API-ready turbulence prediction result."""

    risk: str
    confidence: float
    FTI: float
    recommendation: str
    probabilities: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "confidence": self.confidence,
            "FTI": self.FTI,
            "recommendation": self.recommendation,
            "probabilities": self.probabilities,
        }


class PredictionService:
    """Lazy-loading FTIS model service."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self._artifact: dict[str, Any] | None = None
        self._mtime: float | None = None

    def load(self) -> dict[str, Any]:
        mtime = self.model_path.stat().st_mtime if self.model_path.exists() else None
        if self._artifact is None or self._mtime != mtime:
            self._artifact = load_artifact(self.model_path)
            self._mtime = mtime
        return self._artifact

    def predict(self, payload: dict[str, float]) -> PredictionResult:
        artifact = self.load()
        feature_frame = features_from_prediction_payload(payload)
        pipeline = artifact["pipeline"]
        label_encoder = artifact["label_encoder"]

        if hasattr(pipeline, "predict_proba"):
            probabilities_raw = pipeline.predict_proba(feature_frame)[0]
        else:
            prediction = int(pipeline.predict(feature_frame)[0])
            probabilities_raw = np.zeros(len(label_encoder.classes_))
            probabilities_raw[prediction] = 1.0

        prediction_index = int(np.argmax(probabilities_raw))
        risk = str(label_encoder.inverse_transform([prediction_index])[0])
        confidence = round(float(probabilities_raw[prediction_index]), 4)
        probabilities = {
            str(label): round(float(probabilities_raw[index]), 4)
            for index, label in enumerate(label_encoder.classes_)
        }

        engineered = engineer_turbulence_features(feature_frame)
        fti = round(float(engineered["FTI"].iloc[0]), 2)

        return PredictionResult(
            risk=risk,
            confidence=confidence,
            FTI=fti,
            recommendation=RECOMMENDATIONS.get(
                risk,
                "Review conditions with dispatch and flight crew.",
            ),
            probabilities=probabilities,
        )


prediction_service = PredictionService()
