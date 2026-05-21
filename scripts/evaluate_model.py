"""Evaluate the persisted FTIS model and write model result artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ftis.config import FEATURES_DATA_PATH, MODEL_RESULTS_DIR, MODEL_PATH, TARGET_COLUMN
from ftis.features import engineer_turbulence_features
from ftis.logging_utils import configure_logging
from ftis.modeling import load_artifact, split_dataset, write_json


logger = configure_logging(__name__, "evaluate_model.log")


def load_evaluation_data() -> pd.DataFrame:
    """Load the feature dataset used for evaluation."""

    if FEATURES_DATA_PATH.exists():
        return pd.read_csv(FEATURES_DATA_PATH)

    from ftis.config import TRAINING_DATA_PATH

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"No evaluation data found at {FEATURES_DATA_PATH}")
    return engineer_turbulence_features(pd.read_csv(TRAINING_DATA_PATH))


def save_comparison_csv(comparison: dict[str, Any], path: Path) -> None:
    """Write model comparison metrics as a compact CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "f1_macro",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model", *metric_names])
        for model_name, metrics in comparison.items():
            writer.writerow([model_name, *[metrics.get(metric) for metric in metric_names]])


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    """Render and save the FTIS confusion matrix."""

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("FTIS Turbulence Classifier Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> bool:
    """Save a one-vs-rest ROC curve when class probabilities are available."""

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return False

    y_binary = label_binarize(y_true, classes=list(range(len(labels))))
    if y_binary.shape[1] != probabilities.shape[1]:
        return False

    fig, ax = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(labels):
        RocCurveDisplay.from_predictions(
            y_binary[:, index],
            probabilities[:, index],
            name=label,
            ax=ax,
        )
    ax.set_title("FTIS One-vs-Rest ROC Curves")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    """Evaluate the persisted best model and write report files."""

    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_artifact(MODEL_PATH)
    df = load_evaluation_data()
    _, X_test, _, y_test, label_encoder = split_dataset(df)
    pipeline = artifact["pipeline"]

    y_pred = pipeline.predict(X_test)
    labels = list(label_encoder.classes_)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
    )

    write_json(report_dict, MODEL_RESULTS_DIR / "classification_report.json")
    (MODEL_RESULTS_DIR / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    write_json(artifact.get("comparison", {}), MODEL_RESULTS_DIR / "model_comparison.json")
    save_comparison_csv(
        artifact.get("comparison", {}),
        MODEL_RESULTS_DIR / "model_comparison_report.csv",
    )
    save_confusion_matrix(
        y_test,
        y_pred,
        labels,
        MODEL_RESULTS_DIR / "confusion_matrix.png",
    )

    roc_saved = False
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        roc_saved = save_roc_curve(
            y_test,
            probabilities,
            labels,
            MODEL_RESULTS_DIR / "roc_curve.png",
        )

    summary = {
        "model_path": str(MODEL_PATH),
        "best_model": artifact.get("model_name"),
        "test_rows": int(len(X_test)),
        "target_column": TARGET_COLUMN,
        "roc_curve_saved": roc_saved,
        "outputs": [
            "classification_report.json",
            "classification_report.txt",
            "model_comparison.json",
            "model_comparison_report.csv",
            "confusion_matrix.png",
            *(['roc_curve.png'] if roc_saved else []),
        ],
    }
    write_json(summary, MODEL_RESULTS_DIR / "evaluation_summary.json")
    print(json.dumps(summary, indent=2))
    logger.info("Evaluation artifacts written to %s", MODEL_RESULTS_DIR)


if __name__ == "__main__":
    main()
