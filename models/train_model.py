"""Train and persist the FTIS turbulence prediction model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "training_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "saved" / "turbulence_model.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "saved" / "label_encoder.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

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

TARGET_COLUMN = "turbulence_label"
LABEL_ORDER = ["LOW", "MODERATE", "HIGH"]
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the ML-ready FTIS dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    dataset = pd.read_csv(path)
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataset.columns]

    if missing_columns:
        raise ValueError(
            "Training dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    logger.info("Loaded training dataset from %s with %s rows", path, len(dataset))
    return dataset


def _build_label_encoder() -> LabelEncoder:
    """Build a LabelEncoder with the aviation risk mapping required by FTIS."""

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(LABEL_ORDER, dtype=object)
    return label_encoder


def preprocess_features(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Prepare model features and encode the turbulence target."""

    features = dataset[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)

    column_medians = features.median(numeric_only=True).fillna(0.0)
    features = features.fillna(column_medians).astype(float)

    labels = dataset[TARGET_COLUMN].astype(str).str.strip().str.upper()
    invalid_labels = sorted(set(labels) - set(LABEL_ORDER))

    if invalid_labels:
        raise ValueError(
            "Unsupported turbulence labels found: " + ", ".join(invalid_labels)
        )

    label_encoder = _build_label_encoder()
    encoded_labels = label_encoder.transform(labels.to_numpy())
    validate_class_diversity(encoded_labels)

    logger.info(
        "Prepared %s features and encoded target distribution=%s",
        len(FEATURE_COLUMNS),
        dict(pd.Series(labels).value_counts()),
    )
    return features, encoded_labels, label_encoder


def validate_class_diversity(labels: np.ndarray) -> None:
    """Reject collapsed or unstable class distributions before training."""

    unique_classes = np.unique(labels)

    if len(unique_classes) < 2:
        raise ValueError("Insufficient turbulence class diversity")

    label_counts = pd.Series(labels).value_counts().reindex(
        list(range(len(LABEL_ORDER))),
        fill_value=0,
    )
    missing_labels = [
        LABEL_ORDER[index]
        for index, count in label_counts.items()
        if int(count) == 0
    ]

    if missing_labels:
        raise ValueError(
            "Training dataset must contain all turbulence classes. Missing: "
            + ", ".join(missing_labels)
        )

    if int(label_counts.min()) < 2:
        raise ValueError(
            "Each turbulence class needs at least two rows for stable stratified "
            f"training. Distribution: {label_counts.to_dict()}"
        )


def split_data(
    features: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split the dataset into training and test partitions."""

    if len(features) < 2:
        raise ValueError("At least two rows are required to create a train/test split")

    label_counts = pd.Series(labels).value_counts()
    desired_test_rows = max(len(label_counts), int(np.ceil(len(features) * test_size)))
    desired_train_rows = len(features) - desired_test_rows

    if desired_train_rows < len(label_counts):
        raise ValueError(
            "Not enough rows to create a stratified train/test split across "
            f"{len(label_counts)} classes"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=desired_test_rows,
        random_state=random_state,
        stratify=labels,
    )

    logger.info(
        "Split data into train=%s test=%s stratified=%s",
        len(X_train),
        len(X_test),
        True,
    )
    return X_train, X_test, y_train, y_test


def balance_training_data(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[str, int]]]:
    """Randomly oversample minority classes in the training partition."""

    training = X_train.copy()
    training["_label"] = y_train

    before_counts = training["_label"].value_counts().sort_index()
    target_count = int(before_counts.max())
    balanced_parts: list[pd.DataFrame] = []

    for label_index, count in before_counts.items():
        class_rows = training[training["_label"] == label_index]

        if int(count) < target_count:
            class_rows = class_rows.sample(
                n=target_count,
                replace=True,
                random_state=random_state + int(label_index),
            )

        balanced_parts.append(class_rows)

    balanced = pd.concat(balanced_parts, ignore_index=True).sample(
        frac=1.0,
        random_state=random_state,
    )
    after_counts = balanced["_label"].value_counts().sort_index()

    logger.info(
        "Balanced training distribution before=%s after=%s",
        {
            LABEL_ORDER[int(label)]: int(count)
            for label, count in before_counts.items()
        },
        {
            LABEL_ORDER[int(label)]: int(count)
            for label, count in after_counts.items()
        },
    )

    metadata = {
        "before": {
            LABEL_ORDER[int(label)]: int(count)
            for label, count in before_counts.items()
        },
        "after": {
            LABEL_ORDER[int(label)]: int(count)
            for label, count in after_counts.items()
        },
    }

    return (
        balanced[FEATURE_COLUMNS].reset_index(drop=True),
        balanced["_label"].to_numpy(),
        metadata,
    )


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """Train an XGBoost multi-class turbulence classifier."""

    observed_labels = np.sort(np.unique(y_train))

    if len(observed_labels) < 2:
        raise ValueError("Insufficient turbulence class diversity")

    if set(observed_labels.tolist()) != set(range(len(LABEL_ORDER))):
        raise ValueError(
            "XGBoost training requires LOW, MODERATE, and HIGH classes in "
            f"the training split. Observed={observed_labels.tolist()}"
        )

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for multi-class training. Install it with "
            "`pip install xgboost` or `pip install -r requirements.txt`."
        ) from exc

    compact_lookup = {int(label): index for index, label in enumerate(observed_labels)}
    compact_y_train = np.array([compact_lookup[int(label)] for label in y_train])

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        num_class=len(observed_labels),
        n_jobs=-1,
    )
    model.fit(X_train, compact_y_train)

    logger.info(
        "Trained XGBoost classifier on labels=%s",
        [LABEL_ORDER[int(label)] for label in observed_labels],
    )
    return {
        "model_type": "xgboost",
        "estimator": model,
        "feature_columns": FEATURE_COLUMNS,
        "classes": LABEL_ORDER,
        "training_label_indices": [int(label) for label in observed_labels],
        "balancing_strategy": "random_oversampling",
    }


def predict_probabilities(
    model_artifact: dict[str, Any],
    features: pd.DataFrame,
) -> np.ndarray:
    """Return class-aligned probabilities for an FTIS model artifact."""

    class_count = len(LABEL_ORDER)

    if model_artifact["model_type"] == "constant":
        probabilities = np.zeros((len(features), class_count), dtype=float)
        probabilities[:, int(model_artifact["class_index"])] = 1.0
        return probabilities

    if model_artifact["model_type"] != "xgboost":
        raise ValueError(f"Unsupported model artifact type: {model_artifact['model_type']}")

    estimator = model_artifact["estimator"]
    raw_probabilities = np.asarray(estimator.predict_proba(features), dtype=float)

    if raw_probabilities.ndim == 1:
        raw_probabilities = raw_probabilities.reshape(-1, 1)

    probabilities = np.zeros((len(features), class_count), dtype=float)
    training_label_indices = model_artifact.get("training_label_indices", list(range(class_count)))

    for compact_index, label_index in enumerate(training_label_indices):
        if compact_index < raw_probabilities.shape[1]:
            probabilities[:, int(label_index)] = raw_probabilities[:, compact_index]

    row_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        probabilities,
        row_sums,
        out=np.zeros_like(probabilities),
        where=row_sums != 0,
    )
    return probabilities


def evaluate_model(
    model_artifact: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    train_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate a trained FTIS model artifact."""

    probabilities = predict_probabilities(model_artifact, X_test)
    predictions = probabilities.argmax(axis=1)

    label_indices = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_test,
        predictions,
        labels=label_indices,
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        predictions,
        labels=label_indices,
        target_names=list(label_encoder.classes_),
        zero_division=0,
    )

    metrics = {
        "model_type": model_artifact["model_type"],
        "accuracy": float(accuracy_score(y_test, predictions)),
        "labels": list(label_encoder.classes_),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=label_indices,
        ).tolist(),
        "classification_report": report,
        "train_size": train_size,
        "test_size": int(len(X_test)),
        "feature_columns": FEATURE_COLUMNS,
    }

    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Confusion matrix:\n%s", np.array(metrics["confusion_matrix"]))
    logger.info("Classification report:\n%s", report_text)
    return metrics


def save_model(
    model_artifact: dict[str, Any],
    label_encoder: LabelEncoder,
    model_path: Path = MODEL_PATH,
    encoder_path: Path = ENCODER_PATH,
) -> None:
    """Persist the trained model artifact and label encoder."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_artifact, model_path)
    joblib.dump(label_encoder, encoder_path)

    logger.info("Saved model artifact to %s", model_path)
    logger.info("Saved label encoder to %s", encoder_path)


def save_metrics(metrics: dict[str, Any], path: Path = METRICS_PATH) -> None:
    """Persist evaluation metrics as dashboard/API-ready JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", path)


def main() -> None:
    """Train, evaluate, and persist the FTIS turbulence model."""

    dataset = load_dataset()
    features, labels, label_encoder = preprocess_features(dataset)
    X_train, X_test, y_train, y_test = split_data(features, labels)
    X_train_balanced, y_train_balanced, balancing_metadata = balance_training_data(
        X_train,
        y_train,
    )
    model_artifact = train_xgboost(X_train_balanced, y_train_balanced)
    metrics = evaluate_model(
        model_artifact,
        X_test,
        y_test,
        label_encoder,
        train_size=len(X_train_balanced),
    )
    metrics["balancing"] = balancing_metadata

    save_model(model_artifact, label_encoder)
    save_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("FTIS model training failed")
        raise
