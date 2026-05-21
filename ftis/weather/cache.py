"""Small JSON cache for repeated live weather route queries."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from ftis.config import WEATHER_CACHE_DIR
from ftis.weather.weather_models import WeatherCondition


logger = logging.getLogger(__name__)


class WeatherCache:
    """Filesystem-backed TTL cache safe for local Windows development."""

    def __init__(
        self,
        cache_dir: Path = WEATHER_CACHE_DIR,
        ttl_seconds: int = 900,
    ) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> WeatherCondition | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - float(payload.get("cached_at", 0)) > self.ttl_seconds:
                return None
            return WeatherCondition.from_cache(payload["condition"])
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring invalid weather cache entry path=%s error=%s", path, exc)
            return None

    def set(self, key: str, condition: WeatherCondition) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "cached_at": time.time(),
            "condition": condition.as_dict(),
        }
        path = self._path_for_key(key)
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Unable to write weather cache path=%s error=%s", path, exc)

    def stats(self) -> dict[str, Any]:
        try:
            files = list(self.cache_dir.glob("*.json"))
        except OSError:
            files = []
        return {
            "cache_dir": str(self.cache_dir),
            "entries": len(files),
            "ttl_seconds": self.ttl_seconds,
        }
