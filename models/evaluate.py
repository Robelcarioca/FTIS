"""Standalone evaluation entry point for persisted FTIS models."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.predict import load_artifacts
from models.train_model import (
    DATASET_PATH,
    evaluate_model,
    load_dataset,
    preprocess_features,
    save_metrics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Evaluate the saved FTIS model against the current processed dataset."""

    dataset = load_dataset(DATASET_PATH)
    features, labels, _ = preprocess_features(dataset)
    model_artifact, label_encoder = load_artifacts()
    metrics = evaluate_model(
        model_artifact,
        features,
        labels,
        label_encoder,
        train_size=None,
    )
    metrics["evaluation_scope"] = "full_processed_dataset"

    save_metrics(metrics)
    print(json.dumps(metrics, indent=2))
    logger.info("Standalone FTIS evaluation completed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("FTIS model evaluation failed")
        raise
