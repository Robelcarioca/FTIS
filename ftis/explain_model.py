"""Model explainability helpers for FTIS predictions and reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ftis.config import FEATURES_DATA_PATH, MODEL_FEATURES, MODEL_PATH, MODEL_RESULTS_DIR
from ftis.features import features_from_prediction_payload
from ftis.inference import PredictionService
from ftis.modeling import load_artifact


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfidenceSummary:
    """Human-readable probability and uncertainty summary."""

    predicted_label: str
    confidence: float
    entropy: float
    margin: float
    interval_low: float
    interval_high: float

    def as_dict(self) -> dict[str, float | str]:
        return self.__dict__.copy()


def load_feature_sample(
    path: Path = FEATURES_DATA_PATH,
    *,
    limit: int = 500,
) -> pd.DataFrame:
    """Load a bounded feature sample for explainability and SHAP reports."""

    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {path}")
    frame = pd.read_csv(path)
    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError("Feature dataset missing columns: " + ", ".join(missing))
    return frame[MODEL_FEATURES].head(limit).copy()


def transformed_feature_names(pipeline: Any) -> list[str]:
    """Return post-preprocessor feature names when available."""

    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocess")
    if preprocessor is None:
        return list(MODEL_FEATURES)
    try:
        names = preprocessor.get_feature_names_out()
        return [str(name).replace("numeric__", "").replace("categorical__", "") for name in names]
    except Exception:
        return list(MODEL_FEATURES)


def feature_importance_ranking(
    artifact: dict[str, Any] | None = None,
    *,
    top_n: int = 12,
) -> list[dict[str, float | str]]:
    """Return model feature importance ordered from strongest to weakest."""

    artifact = artifact or load_artifact(MODEL_PATH)
    pipeline = artifact["pipeline"]
    model = getattr(pipeline, "named_steps", {}).get("model", pipeline)
    names = transformed_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    else:
        values = np.ones(len(names), dtype=float)

    names = names[: len(values)]
    total = float(values.sum()) or 1.0
    ranking = sorted(
        (
            {"feature": name, "importance": round(float(value / total), 6)}
            for name, value in zip(names, values)
        ),
        key=lambda item: float(item["importance"]),
        reverse=True,
    )
    return ranking[:top_n]


def probability_confidence_summary(
    probabilities: dict[str, float],
    *,
    predicted_label: str,
) -> ConfidenceSummary:
    """Estimate prediction confidence, entropy, and a simple probability interval."""

    ordered = np.asarray(list(probabilities.values()), dtype=float)
    ordered = np.clip(ordered, 1e-9, 1.0)
    ordered = ordered / ordered.sum()
    confidence = float(probabilities[predicted_label])
    sorted_probs = np.sort(ordered)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else confidence
    entropy = float(-np.sum(ordered * np.log2(ordered)) / np.log2(len(ordered)))
    effective_n = 120.0
    half_width = 1.96 * np.sqrt(confidence * (1.0 - confidence) / effective_n)
    return ConfidenceSummary(
        predicted_label=predicted_label,
        confidence=round(confidence, 6),
        entropy=round(entropy, 6),
        margin=round(margin, 6),
        interval_low=round(max(0.0, confidence - half_width), 6),
        interval_high=round(min(1.0, confidence + half_width), 6),
    )


def explain_prediction(payload: dict[str, float]) -> dict[str, Any]:
    """Return prediction, confidence, and top model drivers for one payload."""

    service = PredictionService()
    result = service.predict(payload).as_dict()
    confidence = probability_confidence_summary(
        result["probabilities"],
        predicted_label=result["risk"],
    )

    try:
        importances = feature_importance_ranking(top_n=8)
    except Exception as exc:
        logger.warning("Feature importance unavailable: %s", exc)
        importances = []

    feature_frame = features_from_prediction_payload(payload)
    feature_values = feature_frame.iloc[0].to_dict()
    drivers = []
    for item in importances:
        feature = str(item["feature"])
        base_name = feature.split("_", maxsplit=1)[-1]
        value = feature_values.get(feature, feature_values.get(base_name))
        drivers.append({**item, "value": value})

    return {
        "prediction": result,
        "confidence": confidence.as_dict(),
        "top_drivers": drivers,
    }


def ensemble_vote(
    pipelines: list[Any],
    feature_frame: pd.DataFrame,
    labels: list[str],
) -> dict[str, Any]:
    """Average probabilities across compatible estimators for ensemble voting."""

    if not pipelines:
        raise ValueError("At least one pipeline is required for ensemble voting")

    votes: list[np.ndarray] = []
    for pipeline in pipelines:
        if hasattr(pipeline, "predict_proba"):
            proba = np.asarray(pipeline.predict_proba(feature_frame), dtype=float)[0]
        else:
            prediction = int(pipeline.predict(feature_frame)[0])
            proba = np.zeros(len(labels), dtype=float)
            proba[prediction] = 1.0
        votes.append(proba)

    mean_probabilities = np.mean(np.vstack(votes), axis=0)
    index = int(np.argmax(mean_probabilities))
    return {
        "risk": labels[index],
        "confidence": round(float(mean_probabilities[index]), 6),
        "probabilities": {
            label: round(float(mean_probabilities[position]), 6)
            for position, label in enumerate(labels)
        },
        "voters": len(pipelines),
    }


def generate_shap_summary(
    output_path: Path = MODEL_RESULTS_DIR / "shap_summary.png",
    *,
    sample_path: Path = FEATURES_DATA_PATH,
    limit: int = 300,
) -> dict[str, Any]:
    """Generate a SHAP summary plot, with a feature-importance fallback."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = load_artifact(MODEL_PATH)
    sample = load_feature_sample(sample_path, limit=limit)
    pipeline = artifact["pipeline"]
    feature_names = transformed_feature_names(pipeline)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            import shap

            preprocessor = pipeline.named_steps.get("preprocess")
            model = pipeline.named_steps.get("model")
            transformed = preprocessor.transform(sample) if preprocessor else sample
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(transformed)
            values = shap_values[0] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(
                values,
                transformed,
                feature_names=feature_names[: transformed.shape[1]],
                show=False,
                plot_type="bar",
            )
            method = "shap"
        except Exception as exc:
            logger.warning("SHAP unavailable, writing feature-importance fallback: %s", exc)
            ranking = feature_importance_ranking(artifact, top_n=12)
            features = [str(item["feature"]) for item in ranking][::-1]
            values = [float(item["importance"]) for item in ranking][::-1]
            plt.figure(figsize=(8, 5))
            plt.barh(features, values, color="#38bdf8")
            plt.xlabel("Relative importance")
            plt.title("FTIS Feature Importance")
            method = "feature_importance_fallback"

        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close()
    except Exception as exc:
        raise RuntimeError("Unable to generate explainability plot") from exc

    return {"path": str(output_path), "method": method}
