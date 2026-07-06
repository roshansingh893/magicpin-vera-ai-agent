"""Service layer — business logic and message composition.

This package houses the core message composition engine.
Services are injected into API routes via FastAPI's dependency injection.
"""

from app.services.composer import compose

__all__ = ["compose"]
