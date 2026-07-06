"""Abstract base class for LLM clients.

Defines the contract that all LLM client implementations must satisfy.
The base class enforces a single generate() method so the composition
layer never depends on a specific provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract LLM client — all providers implement this interface.

    Subclasses must implement ``generate()``, which takes a system
    prompt and a user prompt and returns the raw model output as a
    string.  No parsing, no validation — just the wire response.
    """

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return the raw text response.

        Args:
            system_prompt: The system-level instructions for the model.
            user_prompt: The user-level prompt with full context.

        Returns:
            Raw text string from the model.

        Raises:
            LLMClientError: If the API call fails for any reason.
        """
        ...
