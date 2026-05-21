"""Generate FTIS turbulence features from the merged training dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ftis.config import FEATURES_DATA_PATH, TARGET_COLUMN, TRAINING_DATA_PATH
from ftis.features import OUTPUT_COLUMNS, engineer_turbulence_features
from ftis.logging_utils import configure_logging


logger = configure_logging(__name__, "feature_engineering.log")


def load_training_data(path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load the merged FTIS training dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Expected training dataset at {path}")
    df = pd.read_csv(path)
    logger.info("Loaded training data rows=%s columns=%s", len(df), len(df.columns))
    return df


def save_features(df: pd.DataFrame, path: Path = FEATURES_DATA_PATH) -> None:
    """Persist the feature-engineered FTIS dataset."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = [column for column in OUTPUT_COLUMNS if column in df.columns]
    remaining_columns = [column for column in df.columns if column not in ordered_columns]
    df[[*ordered_columns, *remaining_columns]].to_csv(path, index=False)
    logger.info("Saved feature dataset rows=%s path=%s", len(df), path)


def summarize_features(df: pd.DataFrame) -> None:
    """Print a concise feature engineering summary for CLI users."""

    distribution = df[TARGET_COLUMN].value_counts().reindex(
        ["Low", "Moderate", "High"],
        fill_value=0,
    )
    print("FTIS feature engineering complete")
    print(f"Rows: {len(df)}")
    print(f"Output: {FEATURES_DATA_PATH}")
    print("Turbulence labels:")
    print(distribution.to_string())
    print(
        "FTI range: "
        f"{df['FTI'].min():.2f} - {df['FTI'].max():.2f} "
        f"(mean {df['FTI'].mean():.2f})"
    )


def main() -> None:
    """Run the FTIS feature engineering layer."""

    source = load_training_data()
    features = engineer_turbulence_features(source)
    save_features(features)
    summarize_features(features)


if __name__ == "__main__":
    main()
