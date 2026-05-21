"""Fetch live aircraft state vectors from the OpenSky Network API."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "flights.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "flights.log"
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 30
MIN_FLIGHTS_FOR_MODELING = 150
MAX_FLIGHTS_TO_PROCESS = 300
RANDOM_STATE = 42


def setup_logging() -> None:
    """Configure file-based logging for flight ingestion."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def request_with_retries(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    auth: tuple[str, str] | None = None,
    max_retries: int = MAX_RETRIES,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """Call an HTTP endpoint with basic retry/backoff behavior."""
    for attempt in range(1, max_retries + 1):
        try:
            logging.info("Calling OpenSky API: %s attempt=%s", url, attempt)
            response = requests.get(url, params=params, auth=auth, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logging.warning(
                "OpenSky API request failed attempt=%s/%s error=%s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(2**attempt)
        except ValueError as exc:
            logging.error("OpenSky API returned invalid JSON: %s", exc)
            return None

    logging.error("OpenSky API failed after %s attempts", max_retries)
    return None


def get_opensky_auth() -> tuple[str, str] | None:
    """Return optional OpenSky credentials from environment variables."""
    username = os.getenv("OPENSKY_USERNAME")
    password = os.getenv("OPENSKY_PASSWORD")
    if username and password:
        return username, password
    return None


def fetch_flights() -> pd.DataFrame:
    """Fetch live aircraft positions from OpenSky states endpoint."""
    payload = request_with_retries(OPENSKY_STATES_URL, auth=get_opensky_auth())
    if not payload or "states" not in payload:
        logging.error("No OpenSky states were returned")
        return pd.DataFrame(
            columns=[
                "callsign",
                "latitude",
                "longitude",
                "altitude",
                "velocity",
                "heading",
                "timestamp",
                "state_time_epoch",
            ]
        )

    states = payload.get("states") or []
    records: list[dict[str, Any]] = []
    for state in states:
        records.append(
            {
                "callsign": (state[1] or "").strip() if len(state) > 1 else None,
                "longitude": state[5] if len(state) > 5 else None,
                "latitude": state[6] if len(state) > 6 else None,
                "altitude": state[7] if len(state) > 7 else None,
                "velocity": state[9] if len(state) > 9 else None,
                "heading": state[10] if len(state) > 10 else None,
                "state_time_epoch": state[3] if len(state) > 3 else None,
                "last_contact_epoch": state[4] if len(state) > 4 else None,
            }
        )

    df = pd.DataFrame.from_records(records)
    logging.info("Fetched %s raw flight records", len(df))
    return df


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns and drop rows that cannot support downstream joins."""
    if df.empty:
        logging.warning("Flight dataframe is empty before cleaning")
        return df

    cleaned = df.copy()
    numeric_columns = [
        "latitude",
        "longitude",
        "altitude",
        "velocity",
        "heading",
        "state_time_epoch",
        "last_contact_epoch",
    ]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["timestamp_epoch"] = cleaned["state_time_epoch"].fillna(
        cleaned["last_contact_epoch"]
    )
    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp_epoch"], unit="s", utc=True, errors="coerce"
    )

    required_columns = [
        "latitude",
        "longitude",
        "altitude",
        "velocity",
        "heading",
        "timestamp",
    ]
    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=required_columns)
    cleaned = cleaned[
        (cleaned["latitude"].between(-90, 90))
        & (cleaned["longitude"].between(-180, 180))
        & (cleaned["altitude"] >= 0)
        & (cleaned["velocity"] >= 0)
    ]

    cleaned["callsign"] = cleaned["callsign"].replace("", pd.NA)
    cleaned["callsign"] = cleaned["callsign"].fillna("UNKNOWN")
    cleaned["state_time_epoch"] = cleaned["timestamp"].astype("int64") // 10**9

    if len(cleaned) > MAX_FLIGHTS_TO_PROCESS:
        cleaned = cleaned.sample(
            n=MAX_FLIGHTS_TO_PROCESS,
            random_state=RANDOM_STATE,
        )
        logging.info(
            "Sampled flight records to MAX_FLIGHTS_TO_PROCESS=%s",
            MAX_FLIGHTS_TO_PROCESS,
        )

    if len(cleaned) < MIN_FLIGHTS_FOR_MODELING:
        logging.warning(
            "Only %s cleaned flights available; FTIS modeling target is at least %s",
            len(cleaned),
            MIN_FLIGHTS_FOR_MODELING,
        )

    output_columns = [
        "callsign",
        "latitude",
        "longitude",
        "altitude",
        "velocity",
        "heading",
        "timestamp",
        "state_time_epoch",
    ]
    cleaned = cleaned[output_columns].sort_values("timestamp").reset_index(drop=True)

    logging.info(
        "Cleaned flight records before=%s after=%s dropped=%s",
        before_drop,
        len(cleaned),
        before_drop - len(cleaned),
    )
    return cleaned


def save_flights(df: pd.DataFrame, output_path: Path = DATA_PATH) -> None:
    """Persist cleaned flight records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Saved %s flight records to %s", len(df), output_path)


def main() -> None:
    """Run the flight ingestion pipeline."""
    setup_logging()
    logging.info("Starting flight ingestion")
    flights = fetch_flights()
    cleaned = clean_flights(flights)
    save_flights(cleaned)
    logging.info("Finished flight ingestion")


if __name__ == "__main__":
    main()
