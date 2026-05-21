"""Provider abstractions and HTTP retry helpers for live weather ingestion."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from ftis.weather.weather_models import WeatherCondition, WeatherQuery


logger = logging.getLogger(__name__)


class WeatherProviderError(RuntimeError):
    """Raised when a live weather provider cannot return usable data."""


class AsyncWeatherProvider(ABC):
    """Base class for asynchronous weather provider adapters."""

    name = "provider"
    priority = 100
    base_url = ""

    def __init__(
        self,
        *,
        timeout_seconds: float = 8.0,
        max_retries: int = 2,
        user_agent: str = "FTIS/1.0 aviation-weather-intelligence",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.user_agent = user_agent

    @abstractmethod
    async def fetch(self, query: WeatherQuery) -> WeatherCondition:
        """Fetch and normalize weather for a query."""

    async def _get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """GET JSON with retries, timeout protection, and provider logging."""

        try:
            import httpx
        except ImportError as exc:
            raise WeatherProviderError("Install httpx to enable live weather requests") from exc

        request_headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/geo+json, application/json;q=0.9, */*;q=0.8",
        }
        if headers:
            request_headers.update(headers)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.get(
                        url,
                        params=params,
                        headers=request_headers,
                    )
                if response.status_code == 204:
                    raise WeatherProviderError(f"{self.name} returned no content")
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Weather provider request failed provider=%s attempt=%s error=%s",
                    self.name,
                    attempt,
                    exc,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(0.35 * attempt)

        raise WeatherProviderError(f"{self.name} request failed") from last_error
