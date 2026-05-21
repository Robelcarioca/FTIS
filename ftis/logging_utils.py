"""Logging helpers for FTIS scripts and services."""

from __future__ import annotations

import logging
from pathlib import Path

from ftis.config import LOG_DIR


def configure_logging(name: str, log_file: str | None = None) -> logging.Logger:
    """Configure a consistent console and optional file logger."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            handlers.append(logging.FileHandler(LOG_DIR / log_file, encoding="utf-8"))
        except OSError:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(name)


def ensure_parent(path: Path) -> Path:
    """Create the parent directory for a path and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path
