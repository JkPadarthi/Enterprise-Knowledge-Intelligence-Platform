"""Structured logging setup for the engine."""

from __future__ import annotations

import logging
import sys

from config.settings import get_settings

_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"


def setup_logging(level: str | None = None, *, name: str = "graphrag") -> logging.Logger:
    """Configure root logging to stdout and return the engine logger.

    Args:
        level: Optional override; defaults to ``Settings.log_level``.
        name: Logger name to return.

    Returns:
        The configured engine logger.
    """
    settings = get_settings()
    log_level = (level or settings.log_level).upper()

    root = logging.getLogger()
    root.handlers = []  # avoid duplicate handlers on repeated calls
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(log_level)

    return logging.getLogger(name)
