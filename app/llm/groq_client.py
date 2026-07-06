"""Groq LLM client — OpenAI-compatible interface via the Groq API.

Reads GROQ_API_KEY and GROQ_MODEL from the application configuration.
Exposes a single ``generate()`` method; contains zero business logic.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

from app.core.config import get_settings
from app.core.exceptions import ServiceUnavailableError
from app.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class GroqClient(BaseLLMClient):
    """Groq API client using the OpenAI-compatible SDK.

    Configuration is read from ``Settings`` on first instantiation.
    The client is intentionally stateless beyond the HTTP session so
    it can be shared across concurrent requests.
    """

    def __init__(self) -> None:
        settings = get_settings()

        api_key = settings.groq_api_key
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file or "
                "export it as an environment variable."
            )

        self._model = settings.groq_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._timeout = settings.llm_timeout_seconds

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=float(self._timeout),
        )

        logger.info(
            "GroqClient initialized — model=%s temperature=%.2f max_tokens=%d timeout=%ds",
            self._model,
            self._temperature,
            self._max_tokens,
            self._timeout,
        )

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request to Groq and return the response text.

        Args:
            system_prompt: System-level instructions for the model.
            user_prompt: The fully-assembled user prompt.

        Returns:
            The model's response as a raw string.

        Raises:
            ServiceUnavailableError: On timeout, rate-limit, or connection errors.
        """
        logger.debug("GroqClient.generate — sending request (model=%s)", self._model)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except APITimeoutError as exc:
            logger.error("Groq API timeout after %ds: %s", self._timeout, exc)
            raise ServiceUnavailableError(
                f"LLM request timed out after {self._timeout}s. Please retry."
            ) from exc
        except RateLimitError as exc:
            logger.error("Groq rate limit exceeded: %s", exc)
            raise ServiceUnavailableError(
                "LLM rate limit exceeded. Please wait and retry."
            ) from exc
        except APIConnectionError as exc:
            logger.error("Groq connection error: %s", exc)
            raise ServiceUnavailableError(
                "Unable to connect to the LLM service."
            ) from exc

        content = response.choices[0].message.content
        if not content or not content.strip():
            logger.warning("Groq returned an empty response")
            raise ServiceUnavailableError("LLM returned an empty response.")

        logger.debug(
            "GroqClient.generate — received %d chars (model=%s, usage=%s)",
            len(content),
            self._model,
            response.usage,
        )
        return content.strip()
