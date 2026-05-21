"""Training and evaluation helpers for FTIS models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ftis.config import (
    CATEGORICAL_MODEL_FEATURES,
    LABEL_ORDER,
    MODEL_FEATURES,
    MODEL_PATH,
    NUMERIC_MODEL_FEATURES,
    SETTINGS,
    TARGET_COLUMN,
)


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing graph used by all candidate models."""

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_MODEL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_MODEL_FEATURES),
        ],
        remainder="drop",
    )


def encode_target(labels: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Normalize and encode target labels in the fixed FTIS risk order."""

    normalized = labels.astype(str).str.strip().str.title()
    invalid = sorted(set(normalized) - set(LABEL_ORDER))
    if invalid:
        raise ValueError("Unsupported turbulence labels: " + ", ".join(invalid))

    encoder = LabelEncoder()
    encoder.classes_ = np.array(LABEL_ORDER, dtype=object)
    return encoder.transform(normalized), encoder


def split_dataset(
    df: pd.DataFrame,
    test_size: float = SETTINGS.test_size,
    random_state: int = SETTINGS.random_state,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """Create a stratified train/test split for FTIS risk classification."""

    missing = [column for column in [*MODEL_FEATURES, TARGET_COLUMN] if column not in df]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))

    X = df[MODEL_FEATURES].copy()
    y, label_encoder = encode_target(df[TARGET_COLUMN])

    class_counts = pd.Series(y).value_counts()
    stratify = y if len(class_counts) > 1 and int(class_counts.min()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test, label_encoder


def build_candidate_models(random_state: int = SETTINGS.random_state) -> dict[str, Pipeline]:
    """Return the baseline, Random Forest, and XGBoost candidate pipelines."""

    candidates: dict[str, Pipeline] = {
        "baseline_dummy": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=14,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return candidates

    candidates["xgboost"] = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                XGBClassifier(
                    n_estimators=180,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.86,
                    colsample_bytree=0.86,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    return candidates


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict[str, Any]:
    """Calculate standard classification metrics for a model."""

    labels = list(range(len(label_encoder.classes_)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(label_encoder.classes_),
            output_dict=True,
            zero_division=0,
        ),
    }


def train_candidates(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> tuple[str, Pipeline, dict[str, Any]]:
    """Train candidate models and return the best one by weighted F1 score."""

    comparison: dict[str, Any] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, pipeline in build_candidate_models().items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions, label_encoder)
        comparison[name] = metrics
        fitted_models[name] = pipeline

    best_name = max(comparison, key=lambda item: comparison[item]["f1_weighted"])
    return best_name, fitted_models[best_name], comparison


def build_model_artifact(
    model_name: str,
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    comparison: dict[str, Any],
    training_rows: int,
    test_rows: int,
) -> dict[str, Any]:
    """Package a trained FTIS model with metadata required for inference."""

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "labels": list(label_encoder.classes_),
        "feature_columns": MODEL_FEATURES,
        "numeric_features": NUMERIC_MODEL_FEATURES,
        "categorical_features": CATEGORICAL_MODEL_FEATURES,
        "comparison": comparison,
        "training_rows": int(training_rows),
        "test_rows": int(test_rows),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_version": "ftis-mvp-1",
    }


def save_artifact(artifact: dict[str, Any], path: Path = MODEL_PATH) -> None:
    """Persist the selected FTIS model artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the production FTIS model artifact."""

    if not path.exists():
        raise FileNotFoundError(
            f"FTIS model artifact not found at {path}. Run scripts/train_model.py."
        )

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"Unsupported FTIS model artifact format at {path}")
    return artifact


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON with directories created automatically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
