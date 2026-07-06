"""LLM integration layer — model clients and response parsing.

This package contains the LLM client abstraction and provider-specific
implementations. The LLM layer is decoupled from services and prompts
to allow swapping providers without touching business logic.
"""

from app.llm.base import BaseLLMClient
from app.llm.groq_client import GroqClient

__all__ = ["BaseLLMClient", "GroqClient"]
