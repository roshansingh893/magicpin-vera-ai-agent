"""Global exception handlers for the FastAPI application.

Catches unhandled errors and returns consistent JSON error responses.
Stack traces are logged server-side but never exposed to the client.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class VeraAgentError(Exception):
    """Base exception for all application-specific errors."""

    def __init__(self, message: str = "An internal error occurred.", status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ContextValidationError(VeraAgentError):
    """Raised when business-level context validation fails.

    Distinct from Pydantic's ``RequestValidationError`` — this covers
    semantic issues (e.g., trigger references a merchant not in context).
    """

    def __init__(self, message: str = "Context validation failed.") -> None:
        super().__init__(message=message, status_code=422)


class ServiceUnavailableError(VeraAgentError):
    """Raised when an upstream dependency (LLM, database) is unreachable."""

    def __init__(self, message: str = "Service temporarily unavailable.") -> None:
        super().__init__(message=message, status_code=503)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI application instance."""

    @app.exception_handler(VeraAgentError)
    async def vera_error_handler(_request: Request, exc: VeraAgentError) -> JSONResponse:
        logger.error("VeraAgentError: %s (status=%d)", exc.message, exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Request validation failed: %s", exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        logger.warning("HTTP %d: %s", exc.status_code, exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error."},
        )
