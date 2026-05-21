# FTIS Technical Whitepaper

## Overview

FTIS is organized around a reusable core package and thin interfaces. The FastAPI backend and Streamlit dashboard both reuse `ftis.inference`, `ftis.features`, `ftis.weather`, and `ftis.routes`, keeping prediction logic centralized.

## Live Weather Integration

The weather layer normalizes provider-specific payloads from Open-Meteo, NOAA/NWS, and AviationWeather.gov into a unified schema:

- Wind speed and direction
- Pressure, humidity, and temperature
- Visibility and cloud layers
- Turbulence and jet stream indicators
- Provider metadata and warnings

The service uses async requests, retries, timeout protection, provider failover, and a JSON TTL cache for repeated route queries.

## API Layer

FastAPI includes:

- `/predict`
- `/predict/batch`
- `/route/analyze`
- `/weather/live`
- `/model/metrics`
- `/system/status`
- `/api/v1/*` versioned mirrors

Security and operations features include request validation, JWT authentication skeleton, rate limiting middleware, structured request logging, and startup validation checks.

## Dashboard Layer

The dashboard is a tabbed aviation operations center with route playback, weather overlays, risk heat zones, confidence gauges, model analytics, prediction history, and system health diagnostics.

## MLOps Layer

The local registry in `models/registry/model_registry.json` tracks model versions and active artifacts. CI, Docker, Compose, linting, tests, and pre-commit provide a reproducible development path.
