"""Feature engineering primitives for FTIS turbulence intelligence."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ftis.config import MODEL_FEATURES


BASE_COLUMNS = [
    "latitude",
    "longitude",
    "altitude",
    "windspeed",
    "pressure",
    "temperature",
]

OUTPUT_COLUMNS = [
    "timestamp",
    "callsign",
    *BASE_COLUMNS,
    "altitude_band",
    "wind_shear",
    "temperature_gradient",
    "pressure_variation",
    "atmospheric_instability_score",
    "turbulence_intensity_score",
    "turbulence_score",
    "turbulence_label",
    "FTI",
]


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    prepared = df.copy()
    for column in columns:
        if column not in prepared.columns:
            prepared[column] = np.nan
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def _clip_ratio(series: pd.Series, denominator: float) -> pd.Series:
    return (series / denominator).clip(lower=0.0, upper=1.0)


def altitude_band(altitude: float) -> str:
    """Return an aviation-relevant altitude regime for a meter-based altitude."""

    if pd.isna(altitude):
        return "Unknown"

    altitude_value = float(altitude)
    if altitude_value < 1500:
        return "Surface"
    if altitude_value < 4500:
        return "Low"
    if altitude_value < 9000:
        return "Mid"
    if altitude_value < 13000:
        return "Cruise"
    return "Upper"


def _delta_by_track(df: pd.DataFrame, column: str) -> pd.Series:
    if "callsign" in df.columns:
        grouped = df.groupby("callsign", group_keys=False)[column].diff().abs()
    else:
        grouped = pd.Series(np.nan, index=df.index)

    return grouped.fillna(df[column].diff().abs()).fillna(0.0)


def _engineer_dynamic_terms(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    if "wind_shear_proxy" in engineered.columns:
        shear = pd.to_numeric(engineered["wind_shear_proxy"], errors="coerce").abs()
    else:
        wind_delta = _delta_by_track(engineered, "windspeed")
        altitude_delta = _delta_by_track(engineered, "altitude").clip(lower=1.0)
        shear = wind_delta / altitude_delta

    shear_fallback = engineered["windspeed"].abs() / (
        engineered["altitude"].abs() / 1000.0 + 1.0
    )
    engineered["wind_shear"] = (
        shear.replace([np.inf, -np.inf], np.nan)
        .mask(lambda values: values <= 0, shear_fallback)
        .fillna(shear_fallback)
    )

    if "pressure_variation" in engineered.columns:
        pressure_variation = pd.to_numeric(
            engineered["pressure_variation"],
            errors="coerce",
        ).abs()
    else:
        pressure_variation = _delta_by_track(engineered, "pressure")

    pressure_fallback = (1013.25 - engineered["pressure"]).abs()
    engineered["pressure_variation"] = (
        pressure_variation.fillna(pressure_fallback)
        .mask(lambda values: values <= 0, pressure_fallback)
    )

    altitude_delta = _delta_by_track(engineered, "altitude").clip(lower=1.0)
    temperature_delta = _delta_by_track(engineered, "temperature")
    engineered["temperature_gradient"] = (
        temperature_delta / altitude_delta * 1000.0
    ).replace([np.inf, -np.inf], np.nan)
    lapse_proxy = (15.0 - engineered["temperature"]).abs() / (
        engineered["altitude"].abs() / 1000.0 + 1.0
    )
    engineered["temperature_gradient"] = (
        engineered["temperature_gradient"]
        .fillna(lapse_proxy)
        .mask(lambda values: values <= 0, lapse_proxy)
    )

    return engineered


def engineer_turbulence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create FTIS features and turbulence labels from merged flight/weather data."""

    if df.empty:
        raise ValueError("Cannot engineer features from an empty dataframe")

    engineered = df.copy()
    if "timestamp" in engineered.columns:
        engineered["timestamp"] = pd.to_datetime(
            engineered["timestamp"],
            utc=True,
            errors="coerce",
        )
        sort_columns = ["timestamp"]
        if "callsign" in engineered.columns:
            sort_columns.insert(0, "callsign")
        engineered = engineered.sort_values(sort_columns).reset_index(drop=True)

    engineered = _coerce_numeric(engineered, BASE_COLUMNS)
    medians = engineered[BASE_COLUMNS].median(numeric_only=True).fillna(0.0)
    engineered[BASE_COLUMNS] = engineered[BASE_COLUMNS].fillna(medians)

    engineered["altitude_band"] = engineered["altitude"].apply(altitude_band)
    engineered = _engineer_dynamic_terms(engineered)

    wind_component = _clip_ratio(engineered["windspeed"].abs(), 35.0)
    shear_component = _clip_ratio(engineered["wind_shear"].abs(), 0.08)
    pressure_component = _clip_ratio(engineered["pressure_variation"].abs(), 12.0)
    temp_component = _clip_ratio(engineered["temperature_gradient"].abs(), 8.0)

    if "vertical_speed" in engineered.columns:
        vertical_component = _clip_ratio(
            pd.to_numeric(engineered["vertical_speed"], errors="coerce").abs().fillna(0.0),
            25.0,
        )
    else:
        vertical_component = pd.Series(0.0, index=engineered.index)

    engineered["atmospheric_instability_score"] = (
        100
        * (
            0.30 * shear_component
            + 0.25 * temp_component
            + 0.25 * pressure_component
            + 0.20 * wind_component
        )
    ).round(3)

    engineered["turbulence_intensity_score"] = (
        100
        * (
            0.34 * wind_component
            + 0.26 * shear_component
            + 0.18 * pressure_component
            + 0.14 * temp_component
            + 0.08 * vertical_component
        )
    ).round(3)

    engineered["FTI"] = (
        0.55 * engineered["turbulence_intensity_score"]
        + 0.35 * engineered["atmospheric_instability_score"]
        + 0.10 * (100 * wind_component)
    ).clip(0, 100).round(2)

    engineered["turbulence_score"] = engineered["turbulence_intensity_score"]
    engineered["turbulence_label"] = pd.cut(
        engineered["FTI"],
        bins=[-np.inf, 33.0, 66.0, np.inf],
        labels=["Low", "Moderate", "High"],
    ).astype(str)

    for column in MODEL_FEATURES:
        if column not in engineered.columns:
            raise ValueError(f"Feature engineering failed to create {column}")

    for column in [
        "wind_shear",
        "temperature_gradient",
        "pressure_variation",
        "atmospheric_instability_score",
        "turbulence_intensity_score",
        "turbulence_score",
        "FTI",
    ]:
        engineered[column] = (
            pd.to_numeric(engineered[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    return engineered


def features_from_prediction_payload(payload: dict[str, float]) -> pd.DataFrame:
    """Build the one-row feature frame used by API and dashboard inference."""

    row = {column: payload.get(column) for column in BASE_COLUMNS}
    frame = engineer_turbulence_features(pd.DataFrame([row]))
    return frame[MODEL_FEATURES]
