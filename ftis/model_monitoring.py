"""Model metrics, calibration, and drift monitoring for FTIS."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from ftis.config import FEATURES_DATA_PATH, MODEL_FEATURES, MODEL_PATH, MODEL_RESULTS_DIR
from ftis.modeling import load_artifact


logger = logging.getLogger(__name__)


def model_metrics_summary() -> dict[str, Any]:
    """Return production model metadata and comparison metrics."""

    try:
        artifact = load_artifact(MODEL_PATH)
    except Exception as exc:
        return {
            "model_available": False,
            "model_path": str(MODEL_PATH),
            "error": str(exc),
        }

    comparison = artifact.get("comparison", {})
    best_metrics = comparison.get(artifact.get("model_name"), {})
    return {
        "model_available": True,
        "model_path": str(MODEL_PATH),
        "model_name": artifact.get("model_name"),
        "artifact_version": artifact.get("artifact_version"),
        "trained_at_utc": artifact.get("trained_at_utc"),
        "training_rows": artifact.get("training_rows"),
        "test_rows": artifact.get("test_rows"),
        "labels": artifact.get("labels"),
        "best_metrics": best_metrics,
        "comparison": comparison,
    }


def expected_calibration_error(
    y_true_binary: np.ndarray,
    y_probability: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Compute expected calibration error for one-vs-rest probabilities."""

    y_true_binary = np.asarray(y_true_binary, dtype=float)
    y_probability = np.asarray(y_probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (y_probability >= lower) & (y_probability < upper)
        if not np.any(mask):
            continue
        accuracy = float(y_true_binary[mask].mean())
        confidence = float(y_probability[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - confidence)
    return round(ece, 6)


def calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Calculate calibration metrics for multiclass model probabilities."""

    metrics: dict[str, Any] = {}
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true)
    for index, label in enumerate(labels):
        one_vs_rest = (y_true == index).astype(int)
        class_probability = probabilities[:, index]
        metrics[label] = {
            "brier_score": round(float(brier_score_loss(one_vs_rest, class_probability)), 6),
            "expected_calibration_error": expected_calibration_error(
                one_vs_rest,
                class_probability,
            ),
        }
    try:
        metrics["multiclass_log_loss"] = round(float(log_loss(y_true, probabilities)), 6)
    except ValueError:
        metrics["multiclass_log_loss"] = None
    return metrics


def generate_calibration_curve(
    y_true_binary: np.ndarray,
    y_probability: np.ndarray,
    output_path: Path = MODEL_RESULTS_DIR / "calibration_curve.png",
) -> dict[str, str]:
    """Write a calibration curve image for reports."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prob_true, prob_pred = calibration_curve(
        y_true_binary,
        y_probability,
        n_bins=10,
        strategy="uniform",
    )
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", color="#38bdf8", label="FTIS")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("FTIS Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return {"path": str(output_path)}


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float:
    """Return PSI drift score for one numeric feature."""

    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    reference_pct = np.clip(reference_counts / max(reference_counts.sum(), 1), 1e-6, None)
    current_pct = np.clip(current_counts / max(current_counts.sum(), 1), 1e-6, None)
    return round(float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))), 6)


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Compare reference and current feature distributions."""

    features = features or [column for column in MODEL_FEATURES if column in reference and column in current]
    feature_reports = {}
    for feature in features:
        if not pd.api.types.is_numeric_dtype(reference[feature]):
            continue
        psi = population_stability_index(reference[feature], current[feature])
        feature_reports[feature] = {
            "psi": psi,
            "status": "drift" if psi >= 0.25 else "watch" if psi >= 0.1 else "stable",
            "reference_mean": round(float(pd.to_numeric(reference[feature], errors="coerce").mean()), 6),
            "current_mean": round(float(pd.to_numeric(current[feature], errors="coerce").mean()), 6),
        }
    status = "drift" if any(item["status"] == "drift" for item in feature_reports.values()) else "stable"
    return {"status": status, "features": feature_reports}


def write_drift_monitoring_report(
    current: pd.DataFrame,
    *,
    reference_path: Path = FEATURES_DATA_PATH,
    output_path: Path = MODEL_RESULTS_DIR / "drift_monitoring_report.json",
) -> dict[str, Any]:
    """Write a drift monitoring report against the saved feature dataset."""

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference features not found at {reference_path}")
    reference = pd.read_csv(reference_path)
    report = drift_report(reference, current)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
