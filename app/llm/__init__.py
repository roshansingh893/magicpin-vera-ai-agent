"""LLM integration layer — model clients and response parsing.

This package contains the LLM client abstraction and provider-specific
implementations. The LLM layer is decoupled from services and prompts
to allow swapping providers without touching business logic.

Imports are lazy to avoid loading the openai SDK at module level,
which speeds up startup for scripts that don't need LLM access.
"""

from app.llm.base import BaseLLMClient


def __getattr__(name: str):
    """Lazy-load GroqClient only when explicitly accessed.

    This avoids importing the ``openai`` SDK (and validating
    ``GROQ_API_KEY``) at module load time, which speeds up startup
    for CLI scripts, tests, and evaluation tools that may not need
    the LLM client.
    """
    if name == "GroqClient":
        from app.llm.groq_client import GroqClient
        return GroqClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BaseLLMClient", "GroqClient"]
