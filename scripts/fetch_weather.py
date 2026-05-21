from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FLIGHTS_PATH = PROJECT_ROOT / "data" / "raw" / "flights.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "weather" / "weather.csv"

LOG_PATH = PROJECT_ROOT / "logs" / "weather.log"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

MAX_RETRIES = 2
REQUEST_TIMEOUT_SECONDS = 10
GRID_SIZE_DEGREES = 0.25

# Keep enough aircraft/weather grids to preserve turbulence class diversity.
MAX_FLIGHTS_TO_PROCESS = 250


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def round_to_grid(
    value: float,
    grid_size: float = GRID_SIZE_DEGREES,
) -> float:
    if pd.isna(value):
        return float("nan")

    return round(round(float(value) / grid_size) * grid_size, 4)


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    max_retries: int = MAX_RETRIES,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:

    for attempt in range(1, max_retries + 1):

        try:
            logging.info(
                "Weather request lat=%s lon=%s attempt=%s",
                params.get("latitude"),
                params.get("longitude"),
                attempt,
            )

            response = session.get(
                url,
                params=params,
                timeout=timeout,
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:

            logging.warning(
                "Weather request failed attempt=%s error=%s",
                attempt,
                exc,
            )

            if attempt < max_retries:
                time.sleep(1)

    logging.error("Weather request failed permanently")

    return None


def parse_weather_response(
    payload: dict[str, Any] | None,
) -> dict[str, Any]:

    if not payload:

        return {
            "weather_timestamp": pd.NaT,
            "temperature": None,
            "windspeed": None,
            "winddirection": None,
            "pressure": None,
            "humidity": None,
        }

    current = payload.get("current", {})

    return {
        "weather_timestamp": current.get("time"),
        "temperature": current.get("temperature_2m"),
        "windspeed": current.get("wind_speed_10m"),
        "winddirection": current.get("wind_direction_10m"),
        "pressure": current.get("pressure_msl"),
        "humidity": current.get("relative_humidity_2m"),
    }


def get_weather(
    session: requests.Session,
    lat: float,
    lon: float,
) -> dict[str, Any]:

    grid_lat = round_to_grid(lat)
    grid_lon = round_to_grid(lon)

    params = {
        "latitude": grid_lat,
        "longitude": grid_lon,
        "current": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "pressure_msl",
                "wind_speed_10m",
                "wind_direction_10m",
            ]
        ),
        "timezone": "UTC",
    }

    payload = request_with_retries(
        session,
        OPEN_METEO_URL,
        params=params,
    )

    weather = parse_weather_response(payload)

    weather["latitude_grid"] = grid_lat
    weather["longitude_grid"] = grid_lon

    return weather


def batch_weather(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        logging.warning("No flights found")

        return pd.DataFrame()

    required_columns = {
        "latitude",
        "longitude",
        "timestamp",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    working = df.copy()

    working["latitude"] = pd.to_numeric(
        working["latitude"],
        errors="coerce",
    )

    working["longitude"] = pd.to_numeric(
        working["longitude"],
        errors="coerce",
    )

    working = working.dropna(
        subset=["latitude", "longitude", "timestamp"]
    )

    # Bound API calls while keeping a stable ML-sized weather sample.
    working = working.head(MAX_FLIGHTS_TO_PROCESS)

    records: list[dict[str, Any]] = []

    cache: dict[
        tuple[float, float],
        dict[str, Any],
    ] = {}

    session = requests.Session()

    for _, row in working.iterrows():

        grid_lat = round_to_grid(row["latitude"])
        grid_lon = round_to_grid(row["longitude"])

        cache_key = (grid_lat, grid_lon)

        if cache_key not in cache:

            cache[cache_key] = get_weather(
                session,
                grid_lat,
                grid_lon,
            )

        weather = cache[cache_key]

        records.append(
            {
                "callsign": row.get(
                    "callsign",
                    "UNKNOWN",
                ),
                "timestamp": row["timestamp"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                **weather,
            }
        )

    weather_df = pd.DataFrame(records)

    if not weather_df.empty:

        weather_df["timestamp"] = pd.to_datetime(
            weather_df["timestamp"],
            utc=True,
            errors="coerce",
        )

        weather_df["weather_timestamp"] = pd.to_datetime(
            weather_df["weather_timestamp"],
            utc=True,
            errors="coerce",
        )

    logging.info(
        "Weather ingestion complete rows=%s unique_grids=%s",
        len(weather_df),
        len(cache),
    )

    return weather_df


def save_weather(
    df: pd.DataFrame,
    output_path: Path = WEATHER_PATH,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output_path, index=False)

    logging.info(
        "Saved weather data rows=%s path=%s",
        len(df),
        output_path,
    )


def load_flights(
    input_path: Path = FLIGHTS_PATH,
) -> pd.DataFrame:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Flights file not found: {input_path}"
        )

    flights = pd.read_csv(input_path)

    if "timestamp" in flights.columns:

        flights["timestamp"] = pd.to_datetime(
            flights["timestamp"],
            utc=True,
            errors="coerce",
        )

    logging.info(
        "Loaded flight rows=%s",
        len(flights),
    )

    return flights


def main() -> None:

    setup_logging()

    logging.info("Starting weather ingestion")

    flights = load_flights()

    weather = batch_weather(flights)

    save_weather(weather)

    logging.info("Finished weather ingestion")


if __name__ == "__main__":
    main()
