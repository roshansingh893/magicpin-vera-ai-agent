"""FastAPI application entrypoint.

Configures the application, attaches middleware, registers routes,
and wires up exception handlers. Start with:

    uvicorn app.main:app --reload
"""

import logging

from fastapi import FastAPI

from app.api.routes import router as v1_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    """Application factory — builds and configures the FastAPI instance.

    Using a factory function (instead of a module-level ``app``) allows
    tests to create isolated app instances with different configs.
    """
    # ── Bootstrap logging before anything else ───────────────────
    setup_logging()
    logger = logging.getLogger(__name__)

    settings = get_settings()
    logger.info("Initializing %s v%s [%s]", settings.app_name, settings.app_version, settings.environment)

    # ── Create FastAPI app ───────────────────────────────────────
    application = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── Register exception handlers ─────────────────────────────
    register_exception_handlers(application)

    # ── Include routers ──────────────────────────────────────────
    application.include_router(v1_router)

    logger.info("Application ready — routes registered, exception handlers attached")
    return application


# Module-level instance used by uvicorn
app = create_app()
