"""Service layer — business logic and message composition.

This package houses the core message composition engine,
conversation management, intent detection, and reply handling.
"""

from app.services.composer import compose

__all__ = ["compose"]
