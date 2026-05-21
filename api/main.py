"""FastAPI production service for FTIS."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelMetricsResponse,
    RouteAnalyzeRequest,
    RouteAnalyzeResponse,
    SystemStatusResponse,
    TurbulencePredictionRequest,
    TurbulencePredictionResponse,
    WeatherLiveRequest,
    WeatherLiveResponse,
)
from api.services import predict_batch, predict_turbulence, route_simulator, weather_service
from ftis.config import MODEL_PATH, PROJECT_ROOT as FTIS_PROJECT_ROOT
from ftis.explain_model import feature_importance_ranking
from ftis.model_monitoring import model_metrics_summary
from ftis.routes.route_models import RouteRequest
from ftis.weather.provider_base import WeatherProviderError
from ftis.weather.weather_models import WeatherQuery


logging.basicConfig(
    level=os.getenv("FTIS_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Emit compact JSON request logs suitable for containers."""

    async def dispatch(self, request: Request, call_next: Any) -> JSONResponse:
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "client": request.client.host if request.client else None,
                }
            )
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory per-client rate limiter for local production demos."""

    def __init__(self, app: FastAPI, *, limit_per_minute: int = 120) -> None:
        super().__init__(app)
        self.limit_per_minute = limit_per_minute
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Any) -> JSONResponse:
        if request.url.path in {"/", "/health"}:
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = self.requests[client]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= self.limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={"detail": "FTIS rate limit exceeded. Retry in one minute."},
            )
        bucket.append(now)
        return await call_next(request)


security = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict[str, Any]:
    """JWT authentication skeleton, disabled unless FTIS_REQUIRE_AUTH=true."""

    require_token = os.getenv("FTIS_REQUIRE_AUTH", "false").lower() == "true"
    if not require_token:
        return {"sub": "anonymous", "auth_enabled": False}
    if credentials is None:
        raise HTTPException(status_code=401, detail="Bearer token required")

    secret = os.getenv("FTIS_JWT_SECRET")
    if not secret:
        return {"sub": "unverified", "auth_enabled": True}

    try:
        import jwt

        return jwt.decode(credentials.credentials, secret, algorithms=["HS256"])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid bearer token") from exc


app = FastAPI(
    title="FTIS Flight Turbulence Intelligence API",
    description=(
        "Aviation AI service for turbulence prediction, live weather, route "
        "simulation, model metrics, and operational health diagnostics."
    ),
    version="2.0.0",
    openapi_tags=[
        {"name": "Prediction", "description": "Single and batch turbulence inference."},
        {"name": "Routes", "description": "Airport route simulation and FTI analysis."},
        {"name": "Weather", "description": "Live aviation weather normalization."},
        {"name": "Model", "description": "Model metrics, calibration, and explainability."},
        {"name": "System", "description": "Health and deployment diagnostics."},
    ],
)
app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    limit_per_minute=int(os.getenv("FTIS_RATE_LIMIT_PER_MINUTE", "120")),
)

router = APIRouter(dependencies=[Depends(require_auth)])


def _to_dict(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


@app.on_event("startup")
def startup_validation() -> None:
    """Validate critical runtime directories and model artifact visibility."""

    logger.info("FTIS startup root=%s model_available=%s", FTIS_PROJECT_ROOT, MODEL_PATH.exists())
    weather_service.cache.cache_dir.mkdir(parents=True, exist_ok=True)


@app.get("/", tags=["System"])
def root() -> dict[str, str]:
    """Return public service metadata."""

    return {
        "service": "FTIS Flight Turbulence Intelligence System",
        "status": "online",
        "version": "2.0.0",
        "model": str(MODEL_PATH),
    }


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    """Health check endpoint for deployment probes."""

    return {
        "status": "healthy" if MODEL_PATH.exists() else "degraded",
        "model_available": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@router.post(
    "/predict",
    response_model=TurbulencePredictionResponse,
    tags=["Prediction"],
    summary="Predict turbulence risk for one aircraft/weather state",
)
def predict(payload: TurbulencePredictionRequest) -> dict[str, Any]:
    """Predict turbulence risk for a flight/weather state."""

    try:
        result = predict_turbulence(_to_dict(payload))
    except FileNotFoundError as exc:
        logger.warning("Prediction requested before model was trained: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="FTIS model is unavailable. Run scripts/train_model.py first.",
        ) from exc
    except ValueError as exc:
        logger.warning("Invalid prediction request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="FTIS prediction failed") from exc

    logger.info(
        "Prediction complete risk=%s confidence=%.3f FTI=%.2f",
        result["risk"],
        result["confidence"],
        result["FTI"],
    )
    return result


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict turbulence risk for multiple states",
)
def predict_batch_endpoint(payload: BatchPredictionRequest) -> dict[str, Any]:
    """Score a bounded batch of aircraft/weather states."""

    try:
        records = [_to_dict(record) for record in payload.records]
        predictions = predict_batch(records)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="FTIS batch prediction failed") from exc
    return {"count": len(predictions), "predictions": predictions}


@router.post(
    "/weather/live",
    response_model=WeatherLiveResponse,
    tags=["Weather"],
    summary="Fetch normalized live aviation weather",
)
async def weather_live(payload: WeatherLiveRequest) -> dict[str, Any]:
    """Fetch live weather with provider failover and cache reuse."""

    query = WeatherQuery(**_to_dict(payload))
    try:
        condition = await weather_service.get_live_weather(query)
    except WeatherProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return condition.as_dict()


@router.post(
    "/route/analyze",
    response_model=RouteAnalyzeResponse,
    tags=["Routes"],
    summary="Analyze airport-to-airport turbulence risk",
)
async def route_analyze(
    payload: RouteAnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Simulate a flight path, sample weather, and score route-level turbulence."""

    data = _to_dict(payload)
    request = RouteRequest(
        departure_airport=data["departure_airport"],
        destination_airport=data["destination_airport"],
        cruising_altitude_m=data["cruising_altitude_m"],
        aircraft_speed_kt=data["aircraft_speed_kt"],
        waypoint_count=data["waypoint_count"],
    )
    try:
        analysis = await route_simulator.analyze_route(
            request,
            use_live_weather=bool(data["use_live_weather"]),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Route analysis failed")
        raise HTTPException(status_code=500, detail="FTIS route analysis failed") from exc

    background_tasks.add_task(
        logger.info,
        "Route audit dep=%s dst=%s risk=%s",
        request.departure_airport,
        request.destination_airport,
        analysis.route_risk,
    )
    csv_text = route_simulator.to_csv(analysis)
    return {
        "analysis": analysis.as_dict(),
        "geojson": route_simulator.to_geojson(analysis),
        "csv_preview": "\n".join(csv_text.splitlines()[:8]),
        "overlay": route_simulator.to_map_overlay(analysis),
    }


@router.get(
    "/model/metrics",
    response_model=ModelMetricsResponse,
    tags=["Model"],
    summary="Return model metrics and explainability metadata",
)
def model_metrics() -> dict[str, Any]:
    """Return training metrics, artifact metadata, and feature importance ranking."""

    explainability: dict[str, Any] | None
    try:
        explainability = {"feature_importance": feature_importance_ranking(top_n=12)}
    except Exception as exc:
        explainability = {"warning": str(exc)}
    return {"model": model_metrics_summary(), "explainability": explainability}


@router.get(
    "/system/status",
    response_model=SystemStatusResponse,
    tags=["System"],
    summary="Return runtime diagnostics",
)
def system_status() -> dict[str, Any]:
    """Return API, model, cache, and provider diagnostics."""

    weather_status = weather_service.status()
    status = "healthy" if MODEL_PATH.exists() else "degraded"
    return {
        "status": status,
        "model_available": MODEL_PATH.exists(),
        "weather": weather_status,
        "cache": weather_status["cache"],
        "version": "2.0.0",
    }


app.include_router(router)
app.include_router(router, prefix="/api/v1")
