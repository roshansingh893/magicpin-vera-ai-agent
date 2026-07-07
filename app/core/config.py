"""Application configuration — loads settings from environment variables.

Uses python-dotenv to read from .env files. Settings are validated
at startup via Pydantic to fail fast on misconfiguration.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration.

    Values are loaded from environment variables (or .env file).
    Defaults are safe for local development; production deployments
    should override via environment.
    """

    # ── Application ──────────────────────────────────────────────
    app_name: str = "Vera AI Agent"
    app_version: str = "1.0.0"
    app_description: str = (
        "AI-powered merchant assistant for the magicpin platform. "
        "Composes context-aware WhatsApp messages for merchants and their customers."
    )
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # ── Server ───────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # ── LLM Configuration (Groq) ────────────────────────────────
    groq_api_key: str = ""          # GROQ_API_KEY env var
    groq_model: str = "llama-3.3-70b-versatile"  # GROQ_MODEL env var
    llm_temperature: float = 0.0    # deterministic output per challenge rules
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 30   # challenge constraint: <30s per call

    # ── Dataset Paths (Phase 2) ──────────────────────────────────
    dataset_base_path: str = "../dataset"

    # ── Cache & Rate Limiting (Phase 5) ──────────────────────────
    cache_enabled: bool = True            # CACHE_ENABLED env var
    cache_dir: str = "cache/responses"    # CACHE_DIR env var
    rate_limit_rpm: int = 28              # RATE_LIMIT_RPM env var (Groq free tier: 30)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings.

    Using ``lru_cache`` ensures the .env file is read exactly once
    and the same ``Settings`` instance is reused across the app.
    """
    return Settings()
