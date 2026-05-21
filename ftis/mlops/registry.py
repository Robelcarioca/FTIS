"""Lightweight JSON model registry for local and cloud deployments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ftis.config import MODEL_PATH, MODEL_REGISTRY_PATH


def load_registry(path: Path = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    """Load the local model registry."""

    if not path.exists():
        return {"active_version": None, "versions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def register_model_version(
    *,
    version: str,
    metrics: dict[str, Any] | None = None,
    artifact_path: Path = MODEL_PATH,
    registry_path: Path = MODEL_REGISTRY_PATH,
    activate: bool = True,
) -> dict[str, Any]:
    """Register a model artifact and optionally mark it active."""

    registry = load_registry(registry_path)
    entry = {
        "version": version,
        "artifact_path": str(artifact_path),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics or {},
    }
    registry["versions"] = [
        item for item in registry.get("versions", []) if item.get("version") != version
    ]
    registry["versions"].append(entry)
    if activate:
        registry["active_version"] = version
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry
