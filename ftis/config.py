"""Central configuration for the Flight Turbulence Intelligence System."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINING_DATA_PATH = PROCESSED_DATA_DIR / "training_data.csv"
FEATURES_DATA_PATH = PROCESSED_DATA_DIR / "features.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "ftis_model.pkl"
LEGACY_MODEL_PATH = MODEL_DIR / "saved" / "turbulence_model.pkl"
LEGACY_ENCODER_PATH = MODEL_DIR / "saved" / "label_encoder.pkl"

REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
MODEL_RESULTS_DIR = DOCS_DIR / "model_results"
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
WEATHER_CACHE_DIR = CACHE_DIR / "weather"
ASSETS_DIR = DOCS_DIR / "assets"
MODEL_REGISTRY_DIR = MODEL_DIR / "registry"
MODEL_REGISTRY_PATH = MODEL_REGISTRY_DIR / "model_registry.json"

LABEL_ORDER = ["Low", "Moderate", "High"]
LABEL_NORMALIZATION = {
    "LOW": "Low",
    "Low": "Low",
    "low": "Low",
    "MODERATE": "Moderate",
    "Moderate": "Moderate",
    "moderate": "Moderate",
    "MEDIUM": "Moderate",
    "Medium": "Moderate",
    "HIGH": "High",
    "High": "High",
    "high": "High",
}

RISK_COLORS = {
    "Low": "#22c55e",
    "Moderate": "#facc15",
    "High": "#ef4444",
}

RECOMMENDATIONS = {
    "Low": "Continue planned route with routine monitoring.",
    "Moderate": "Monitor conditions and consider altitude adjustment.",
    "High": "Avoid route section or request tactical reroute.",
}

NUMERIC_MODEL_FEATURES = [
    "latitude",
    "longitude",
    "altitude",
    "windspeed",
    "pressure",
    "temperature",
    "wind_shear",
    "temperature_gradient",
    "pressure_variation",
    "atmospheric_instability_score",
    "turbulence_intensity_score",
    "FTI",
]

CATEGORICAL_MODEL_FEATURES = ["altitude_band"]

MODEL_FEATURES = [*NUMERIC_MODEL_FEATURES, *CATEGORICAL_MODEL_FEATURES]
TARGET_COLUMN = "turbulence_label"


@dataclass(frozen=True)
class ModelSettings:
    """Training settings shared by scripts and documentation."""

    random_state: int = 42
    test_size: float = 0.2
    scoring_metric: str = "f1_weighted"


SETTINGS = ModelSettings()
