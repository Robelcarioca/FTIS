"""High-level live weather service with cache and provider failover."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from typing import Any

from ftis.weather.aviation_weather import AviationWeatherProvider
from ftis.weather.cache import WeatherCache
from ftis.weather.noaa import NOAAProvider
from ftis.weather.open_meteo import OpenMeteoProvider
from ftis.weather.provider_base import AsyncWeatherProvider, WeatherProviderError
from ftis.weather.weather_models import WeatherCondition, WeatherQuery


logger = logging.getLogger(__name__)


class WeatherService:
    """Fetch live aviation weather using cache-first provider failover."""

    def __init__(
        self,
        providers: Iterable[AsyncWeatherProvider] | None = None,
        cache: WeatherCache | None = None,
    ) -> None:
        self.providers = sorted(
            list(providers)
            if providers is not None
            else [OpenMeteoProvider(), NOAAProvider(), AviationWeatherProvider()],
            key=lambda provider: provider.priority,
        )
        self.cache = cache or WeatherCache()

    async def get_live_weather(self, query: WeatherQuery) -> WeatherCondition:
        """Return weather for a single point from cache or provider failover."""

        cache_key = query.cache_key()
        cached = self.cache.get(cache_key)
        if cached:
            cached.warnings.append("served_from_cache")
            return cached

        errors: list[str] = []
        for provider in self.providers:
            try:
                condition = await provider.fetch(query)
            except WeatherProviderError as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            except Exception as exc:
                logger.exception("Unexpected provider failure provider=%s", provider.name)
                errors.append(f"{provider.name}: {exc}")
                continue

            if condition.wind_speed_kmh is None and condition.temperature_c is None:
                errors.append(f"{provider.name}: incomplete weather payload")
                continue
            if errors:
                condition.warnings.extend(errors)
            self.cache.set(cache_key, condition)
            return condition

        raise WeatherProviderError("All weather providers failed: " + " | ".join(errors))

    async def get_route_weather(
        self,
        queries: Iterable[WeatherQuery],
        *,
        concurrency: int = 6,
    ) -> list[WeatherCondition | None]:
        """Sample weather along a route without overwhelming provider APIs."""

        semaphore = asyncio.Semaphore(concurrency)

        async def _sample(query: WeatherQuery) -> WeatherCondition | None:
            async with semaphore:
                try:
                    return await self.get_live_weather(query)
                except WeatherProviderError as exc:
                    logger.warning(
                        "Route weather sample unavailable lat=%.3f lon=%.3f error=%s",
                        query.latitude,
                        query.longitude,
                        exc,
                    )
                    return None

        return await asyncio.gather(*[_sample(query) for query in queries])

    def get_live_weather_sync(self, query: WeatherQuery) -> WeatherCondition:
        """Synchronous adapter for Streamlit and scripts."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("Use await get_live_weather inside an active event loop")
        return asyncio.run(self.get_live_weather(query))

    def status(self) -> dict[str, Any]:
        return {
            "providers": [provider.name for provider in self.providers],
            "cache": self.cache.stats(),
        }


weather_service = WeatherService()
