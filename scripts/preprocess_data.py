"""Build the FTIS ML training dataset from flight and weather data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLIGHTS_PATH = PROJECT_ROOT / "data" / "raw" / "flights.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "weather" / "weather.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "training_data.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "preprocess.log"

GRID_SIZE_DEGREES = 0.25
WEATHER_TIME_TOLERANCE = pd.Timedelta(hours=2)
NOISE_STD = 0.5
RANDOM_SEED = 42

WEATHER_COLUMNS = [
    "temperature",
    "windspeed",
    "winddirection",
    "pressure",
    "humidity",
]

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

OUTPUT_COLUMNS = [
    "timestamp",
    "callsign",
    "latitude",
    "longitude",
    *FEATURE_COLUMNS,
    "turbulence_score",
    "turbulence_label",
]


def setup_logging() -> None:
    """Configure preprocessing logs."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        logging.basicConfig(
            filename=LOG_PATH,
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            encoding="utf-8",
            force=True,
        )
    except OSError as exc:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        logging.warning("File logging unavailable at %s: %s", LOG_PATH, exc)


def round_to_grid(value: float, grid_size: float = GRID_SIZE_DEGREES) -> float:
    """Round coordinates to the weather grid resolution."""

    if pd.isna(value):
        return float("nan")

    return round(round(float(value) / grid_size) * grid_size, 4)


def load_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load a CSV and ensure required columns exist."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    logging.info("Loaded %s rows from %s", len(df), path)
    return df


def load_flights(path: Path = FLIGHTS_PATH) -> pd.DataFrame:
    """Load raw flight states."""

    required_columns = [
        "callsign",
        "latitude",
        "longitude",
        "altitude",
        "velocity",
        "heading",
        "timestamp",
    ]
    return load_csv(path, required_columns)


def load_weather(path: Path = WEATHER_PATH) -> pd.DataFrame:
    """Load weather records."""

    required_columns = [
        "timestamp",
        "latitude",
        "longitude",
        *WEATHER_COLUMNS,
    ]
    return load_csv(path, required_columns)


def prepare_grid_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add latitude/longitude grid columns when missing."""

    prepared = df.copy()
    prepared["latitude"] = pd.to_numeric(prepared["latitude"], errors="coerce")
    prepared["longitude"] = pd.to_numeric(prepared["longitude"], errors="coerce")

    if "latitude_grid" not in prepared.columns:
        prepared["latitude_grid"] = prepared["latitude"].apply(round_to_grid)
    else:
        prepared["latitude_grid"] = pd.to_numeric(
            prepared["latitude_grid"],
            errors="coerce",
        ).fillna(prepared["latitude"].apply(round_to_grid))

    if "longitude_grid" not in prepared.columns:
        prepared["longitude_grid"] = prepared["longitude"].apply(round_to_grid)
    else:
        prepared["longitude_grid"] = pd.to_numeric(
            prepared["longitude_grid"],
            errors="coerce",
        ).fillna(prepared["longitude"].apply(round_to_grid))

    return prepared


def prepare_flights(flights: pd.DataFrame) -> pd.DataFrame:
    """Normalize flight records for feature engineering."""

    prepared = prepare_grid_columns(flights)
    prepared["timestamp"] = pd.to_datetime(
        prepared["timestamp"],
        utc=True,
        errors="coerce",
    )

    numeric_columns = [
        "altitude",
        "velocity",
        "heading",
        "latitude",
        "longitude",
        "latitude_grid",
        "longitude_grid",
    ]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared["callsign"] = prepared["callsign"].fillna("UNKNOWN").astype(str).str.strip()
    prepared["callsign"] = prepared["callsign"].replace("", "UNKNOWN")
    prepared = prepared.dropna(
        subset=[
            "timestamp",
            "latitude",
            "longitude",
            "altitude",
            "velocity",
            "heading",
            "latitude_grid",
            "longitude_grid",
        ]
    )
    prepared = prepared[
        prepared["latitude"].between(-90, 90)
        & prepared["longitude"].between(-180, 180)
        & (prepared["altitude"] >= 0)
        & (prepared["velocity"] >= 0)
    ]

    return prepared.sort_values(["timestamp", "callsign"]).reset_index(drop=True)


def prepare_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Normalize weather records for nearest-neighbor matching."""

    prepared = prepare_grid_columns(weather)
    prepared["timestamp"] = pd.to_datetime(
        prepared["timestamp"],
        utc=True,
        errors="coerce",
    )

    if "weather_timestamp" in prepared.columns:
        prepared["match_timestamp"] = pd.to_datetime(
            prepared["weather_timestamp"],
            utc=True,
            errors="coerce",
        ).fillna(prepared["timestamp"])
    else:
        prepared["match_timestamp"] = prepared["timestamp"]

    for column in [*WEATHER_COLUMNS, "latitude_grid", "longitude_grid"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    prepared = prepared.dropna(
        subset=[
            "match_timestamp",
            "latitude_grid",
            "longitude_grid",
            "temperature",
            "windspeed",
            "pressure",
            "humidity",
        ]
    )

    if prepared.empty:
        raise ValueError("Weather data is empty after normalization")

    return prepared.sort_values("match_timestamp").reset_index(drop=True)


def merge_data(
    flights: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Merge flights with nearest weather by grid/time, then nearest fallback."""

    if flights.empty or weather.empty:
        logging.warning(
            "Cannot merge empty inputs flights=%s weather=%s",
            len(flights),
            len(weather),
        )
        return pd.DataFrame()

    flights_prepared = prepare_flights(flights).reset_index(names="flight_id")
    weather_prepared = prepare_weather(weather)
    flight_columns = list(flights_prepared.columns)
    weather_match_columns = [
        "latitude_grid",
        "longitude_grid",
        "match_timestamp",
        *WEATHER_COLUMNS,
    ]

    flights_sorted = flights_prepared.sort_values("timestamp").reset_index(drop=True)
    weather_sorted = weather_prepared[weather_match_columns].sort_values(
        "match_timestamp"
    )

    grid_matches = pd.merge_asof(
        flights_sorted,
        weather_sorted,
        left_on="timestamp",
        right_on="match_timestamp",
        by=["latitude_grid", "longitude_grid"],
        direction="nearest",
        tolerance=WEATHER_TIME_TOLERANCE,
    )
    matched_mask = grid_matches[WEATHER_COLUMNS].notna().all(axis=1)

    matched = grid_matches[matched_mask].copy()
    matched["weather_match_type"] = "grid_time"
    matched["weather_match_timestamp"] = matched["match_timestamp"]

    unmatched = grid_matches.loc[~matched_mask, flight_columns].copy()
    fallback_count = len(unmatched)

    if fallback_count:
        flight_lat = unmatched["latitude_grid"].to_numpy()[:, np.newaxis]
        flight_lon = unmatched["longitude_grid"].to_numpy()[:, np.newaxis]
        flight_time = unmatched["timestamp"].astype("int64").to_numpy()[:, np.newaxis]

        weather_lat = weather_prepared["latitude_grid"].to_numpy()[np.newaxis, :]
        weather_lon = weather_prepared["longitude_grid"].to_numpy()[np.newaxis, :]
        weather_time = (
            weather_prepared["match_timestamp"].astype("int64").to_numpy()[np.newaxis, :]
        )

        grid_distance = np.sqrt(
            (weather_lat - flight_lat) ** 2
            + (weather_lon - flight_lon) ** 2
        )
        time_delta_seconds = np.abs(weather_time - flight_time) / 1_000_000_000
        combined_distance = grid_distance * 1_000_000_000 + time_delta_seconds
        nearest_indices = combined_distance.argmin(axis=1)

        fallback_weather = weather_prepared.iloc[nearest_indices].reset_index(drop=True)
        fallback = unmatched.reset_index(drop=True)
        for column in WEATHER_COLUMNS:
            fallback[column] = fallback_weather[column].to_numpy()
        fallback["weather_match_type"] = "nearest_fallback"
        fallback["weather_match_timestamp"] = fallback_weather[
            "match_timestamp"
        ].to_numpy()
    else:
        fallback = pd.DataFrame(columns=[*flight_columns, *WEATHER_COLUMNS])

    merged = pd.concat([matched, fallback], ignore_index=True, sort=False)
    merged = merged.sort_values("flight_id").drop(columns=["flight_id"])
    logging.info(
        "Merged rows=%s grid_time_matches=%s fallback_matches=%s",
        len(merged),
        int(matched_mask.sum()),
        fallback_count,
    )
    return merged


def _circular_heading_delta(values: pd.Series) -> pd.Series:
    """Compute minimal absolute heading change in degrees."""

    delta = values.diff().abs()
    return np.minimum(delta, 360 - delta)


def _safe_time_delta_seconds(timestamps: pd.Series) -> pd.Series:
    """Return positive timestamp deltas with invalid values set to NaN."""

    delta_seconds = timestamps.diff().dt.total_seconds()
    return delta_seconds.where(delta_seconds > 0)


def _smoothed(series: pd.Series, window: int = 3) -> pd.Series:
    """Rolling mean smoothing for noisy aviation signals."""

    return series.rolling(window=window, min_periods=1).mean()


def _group_delta(
    df: pd.DataFrame,
    column: str,
    *,
    absolute: bool = True,
) -> pd.Series:
    """Compute per-callsign deltas with global fallback for sparse callsigns."""

    grouped_delta = df.groupby("callsign", group_keys=False)[column].diff()
    global_delta = df[column].diff()

    if absolute:
        grouped_delta = grouped_delta.abs()
        global_delta = global_delta.abs()

    return grouped_delta.fillna(global_delta).fillna(0.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer stable FTIS model features."""

    engineered = df.copy().sort_values(["callsign", "timestamp"]).reset_index(drop=True)

    for column in [
        "altitude",
        "velocity",
        "heading",
        *WEATHER_COLUMNS,
    ]:
        engineered[column] = pd.to_numeric(engineered[column], errors="coerce")

    engineered[WEATHER_COLUMNS] = engineered[WEATHER_COLUMNS].fillna(
        engineered[WEATHER_COLUMNS].median(numeric_only=True)
    )

    group_time_delta = engineered.groupby(
        "callsign",
        group_keys=False,
    )["timestamp"].diff().dt.total_seconds()
    global_time_delta = _safe_time_delta_seconds(engineered["timestamp"])
    time_delta = group_time_delta.where(group_time_delta > 0).fillna(global_time_delta)
    time_delta = time_delta.replace(0, np.nan)

    altitude_delta = _group_delta(engineered, "altitude", absolute=False)
    velocity_delta = _group_delta(engineered, "velocity", absolute=True)
    pressure_delta = _group_delta(engineered, "pressure", absolute=True)
    windspeed_delta = _group_delta(engineered, "windspeed", absolute=True)

    grouped_heading_delta = engineered.groupby(
        "callsign",
        group_keys=False,
    )["heading"].transform(_circular_heading_delta)
    global_heading_delta = _circular_heading_delta(engineered["heading"])
    heading_delta = grouped_heading_delta.fillna(global_heading_delta).fillna(0.0)

    engineered["vertical_speed"] = (altitude_delta / time_delta).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    engineered["turn_rate"] = (heading_delta / time_delta).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    engineered["speed_variation"] = velocity_delta
    engineered["altitude_variation"] = altitude_delta.abs()
    engineered["pressure_variation"] = pressure_delta
    engineered["wind_shear_proxy"] = windspeed_delta / engineered[
        "altitude_variation"
    ].clip(lower=1.0)

    noisy_columns = [
        "windspeed",
        "pressure_variation",
        "altitude_variation",
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    for column in noisy_columns:
        engineered[column] = (
            engineered[column].fillna(0.0)
            + rng.normal(0, NOISE_STD, size=len(engineered))
        ).clip(lower=0.0)

    smooth_columns = [
        "vertical_speed",
        "turn_rate",
        "speed_variation",
        "altitude_variation",
        "pressure_variation",
        "wind_shear_proxy",
    ]
    for column in smooth_columns:
        engineered[column] = _smoothed(engineered[column].fillna(0.0))

    engineered[FEATURE_COLUMNS] = engineered[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    engineered[FEATURE_COLUMNS] = engineered[FEATURE_COLUMNS].fillna(
        engineered[FEATURE_COLUMNS].median(numeric_only=True)
    )
    engineered[FEATURE_COLUMNS] = engineered[FEATURE_COLUMNS].fillna(0.0)

    logging.info("Engineered FTIS feature columns=%s", FEATURE_COLUMNS)
    return engineered.sort_values("timestamp").reset_index(drop=True)


def generate_turbulence_label(df: pd.DataFrame) -> pd.DataFrame:
    """Generate normalized multi-class turbulence scores and labels."""

    labeled = df.copy()
    score = (
        0.6 * labeled["windspeed"]
        + 0.5 * labeled["altitude_variation"]
        + 0.4 * labeled["pressure_variation"]
        + 0.3 * labeled["vertical_speed"].abs()
        + 0.2 * labeled["turn_rate"].abs()
        + 0.2 * labeled["humidity"]
    )

    score_min = float(score.min())
    score_max = float(score.max())
    score_range = score_max - score_min

    if score_range <= 0:
        raise ValueError(
            "Cannot normalize turbulence score because all rows have the same score"
        )

    score = (score - score_min) / score_range * 100
    labeled["turbulence_score"] = score.round(6)
    labeled["turbulence_label"] = np.select(
        [
            labeled["turbulence_score"] < 30,
            labeled["turbulence_score"] <= 60,
        ],
        [
            "LOW",
            "MODERATE",
        ],
        default="HIGH",
    )

    return labeled


def validate_training_data(df: pd.DataFrame) -> None:
    """Print and enforce production ML dataset quality checks."""

    if df.empty:
        raise ValueError("Training dataset is empty")

    missing_values = df[FEATURE_COLUMNS].isna().sum()
    missing_features = missing_values[missing_values > 0]
    if not missing_features.empty:
        raise ValueError(f"Training features contain NaNs: {missing_features.to_dict()}")

    value_counts = df["turbulence_label"].value_counts().reindex(
        ["LOW", "MODERATE", "HIGH"],
        fill_value=0,
    )
    distribution_pct = (value_counts / len(df) * 100).round(2)
    min_score = float(df["turbulence_score"].min())
    max_score = float(df["turbulence_score"].max())

    print("Turbulence label counts:")
    print(value_counts)
    print(f"Turbulence score min/max: {min_score:.2f} / {max_score:.2f}")
    print("Turbulence class distribution %:")
    print(distribution_pct)

    logging.info("Turbulence label counts=%s", value_counts.to_dict())
    logging.info("Turbulence score min=%s max=%s", min_score, max_score)
    logging.info("Turbulence class distribution pct=%s", distribution_pct.to_dict())

    if (value_counts > 0).sum() < 3:
        raise ValueError(
            "Rejected dataset: expected LOW, MODERATE, and HIGH turbulence classes"
        )

    if value_counts["HIGH"] == 0:
        raise ValueError("Rejected dataset: HIGH turbulence class is 0%")


def save_training_data(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Validate and save the processed training dataset."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = df[OUTPUT_COLUMNS].copy()
    validate_training_data(output)
    output.to_csv(output_path, index=False)
    logging.info("Saved training dataset rows=%s path=%s", len(output), output_path)


def main() -> None:
    """Run the FTIS preprocessing pipeline."""

    setup_logging()
    logging.info("Starting preprocessing")

    flights = load_flights()
    weather = load_weather()
    merged = merge_data(flights, weather)
    engineered = engineer_features(merged)
    labeled = generate_turbulence_label(engineered)
    save_training_data(labeled)

    logging.info("Finished preprocessing")


if __name__ == "__main__":
    main()
