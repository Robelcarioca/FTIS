"""Train and persist the production FTIS turbulence classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ftis.config import (
    FEATURES_DATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    TRAINING_DATA_PATH,
)
from ftis.features import engineer_turbulence_features
from ftis.logging_utils import configure_logging
from ftis.modeling import (
    build_model_artifact,
    save_artifact,
    split_dataset,
    train_candidates,
    write_json,
)


logger = configure_logging(__name__, "train_model.log")


def load_feature_dataset() -> pd.DataFrame:
    """Load features.csv, generating it from training_data.csv when needed."""

    if FEATURES_DATA_PATH.exists():
        df = pd.read_csv(FEATURES_DATA_PATH)
        logger.info("Loaded feature dataset from %s rows=%s", FEATURES_DATA_PATH, len(df))
        return df

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Neither {FEATURES_DATA_PATH} nor {TRAINING_DATA_PATH} exists"
        )

    logger.info("features.csv not found; engineering features from training_data.csv")
    source = pd.read_csv(TRAINING_DATA_PATH)
    features = engineer_turbulence_features(source)
    FEATURES_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_DATA_PATH, index=False)
    return features


def validate_dataset(df: pd.DataFrame) -> None:
    """Fail fast when the dataset cannot support supervised training."""

    if df.empty:
        raise ValueError("Feature dataset is empty")

    distribution = df[TARGET_COLUMN].astype(str).str.title().value_counts()
    if len(distribution) < 2:
        raise ValueError("At least two turbulence classes are required for training")

    logger.info("Training label distribution=%s", distribution.to_dict())


def main() -> None:
    """Train baseline, Random Forest, and XGBoost models, then persist the best."""

    df = load_feature_dataset()
    validate_dataset(df)
    X_train, X_test, y_train, y_test, label_encoder = split_dataset(df)
    best_name, best_pipeline, comparison = train_candidates(
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
    )

    artifact = build_model_artifact(
        model_name=best_name,
        pipeline=best_pipeline,
        label_encoder=label_encoder,
        comparison=comparison,
        training_rows=len(X_train),
        test_rows=len(X_test),
    )
    save_artifact(artifact, MODEL_PATH)

    comparison_path = REPORTS_DIR / "model_comparison.json"
    write_json(
        {
            "best_model": best_name,
            "model_path": str(MODEL_PATH),
            "comparison": comparison,
        },
        comparison_path,
    )

    print(json.dumps({"best_model": best_name, "model_path": str(MODEL_PATH)}, indent=2))
    print(f"Model comparison saved to {comparison_path}")
    logger.info("Saved best FTIS model=%s path=%s", best_name, MODEL_PATH)


if __name__ == "__main__":
    main()
