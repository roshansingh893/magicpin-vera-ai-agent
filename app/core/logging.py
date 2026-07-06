"""Application-wide logging configuration.

Centralizes log format, level, and handler setup so every module
gets a consistent logger via ``logging.getLogger(__name__)``.
"""

import logging
import sys

from app.core.config import get_settings


def setup_logging() -> None:
    """Configure the root logger for the application.

    Called once at startup from ``app.main``. Sets a human-readable
    format for development and a structured format suitable for
    log aggregation in production.
    """
    settings = get_settings()

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # ── Format ───────────────────────────────────────────────────
    if settings.environment == "production":
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"

    # ── Handler ──────────────────────────────────────────────────
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    # ── Root logger ──────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(log_level)

    # Prevent duplicate handlers on reload
    if not root.handlers:
        root.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
