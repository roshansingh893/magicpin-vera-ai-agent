"""Message composition service — the orchestration core of Phase 2.

compose() is the single entry point called by the API layer.  It:
1. Receives validated request contexts.
2. Determines merchant vs. customer flow.
3. Builds the prompt via the prompt layer.
4. Calls the LLM via the Groq client.
5. Parses and validates the JSON response.
6. Returns a ComposedMessage.

This module contains orchestration only — no business logic, no
prompt construction, no LLM protocol details.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.llm.groq_client import GroqClient
from app.services.prompt_builder import build_prompts
from app.services.output_validator import (
    OutputValidationError,
    parse_llm_response,
    validate_composed_message,
)

if TYPE_CHECKING:
    from app.models.requests import (
        CategoryContext,
        CustomerContext,
        MerchantContext,
        TriggerContext,
    )
    from app.models.responses import ComposedMessage

logger = logging.getLogger(__name__)

# Module-level LLM client instance — initialized lazily
_llm_client: GroqClient | None = None

# Maximum retry attempts for malformed LLM output
MAX_RETRIES = 2


def _get_llm_client() -> GroqClient:
    """Return the module-level Groq client, creating it on first use.

    Lazy initialization avoids import-time errors when GROQ_API_KEY
    is not yet configured (e.g., during testing).
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = GroqClient()
    return _llm_client


def set_llm_client(client: GroqClient | None) -> None:
    """Override the LLM client (for dependency injection in tests).

    Args:
        client: A GroqClient instance or None to reset.
    """
    global _llm_client
    _llm_client = client


async def compose(
    category: CategoryContext,
    merchant: MerchantContext,
    trigger: TriggerContext,
    customer: CustomerContext | None = None,
) -> ComposedMessage:
    """Compose a WhatsApp message from the 4-context framework.

    Orchestrates the full pipeline: prompt → LLM → parse → validate.

    Args:
        category: Vertical-level knowledge (voice, offers, peer stats).
        merchant: This specific merchant's state and history.
        trigger: The event prompting this message.
        customer: Optional customer context for customer-facing messages.

    Returns:
        A fully populated and validated ComposedMessage.

    Raises:
        OutputValidationError: If the LLM output is invalid after retries.
        ServiceUnavailableError: If the LLM call itself fails.
    """
    flow = "customer" if customer is not None else "merchant"
    logger.info(
        "compose() started — flow=%s merchant=%s trigger=%s",
        flow,
        merchant.merchant_id,
        trigger.kind,
    )

    # ── 1. Build prompts ─────────────────────────────────────────
    system_prompt, user_prompt = build_prompts(
        category, merchant, trigger, customer
    )

    # ── 2. Call LLM with retry on parse failure ──────────────────
    client = _get_llm_client()
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info("LLM call attempt %d/%d", attempt, MAX_RETRIES)

        raw_response = await client.generate(system_prompt, user_prompt)
        logger.debug("Raw LLM response (attempt %d): %.300s", attempt, raw_response)

        try:
            parsed = parse_llm_response(raw_response)
            message = validate_composed_message(parsed)
            logger.info(
                "compose() succeeded — body=%d chars, cta=%s, send_as=%s",
                len(message.body),
                message.cta,
                message.send_as,
            )
            return message
        except OutputValidationError as exc:
            last_error = exc
            logger.warning(
                "Validation failed on attempt %d/%d: %s",
                attempt,
                MAX_RETRIES,
                exc.message,
            )

    # All retries exhausted
    logger.error("compose() failed after %d attempts — last error: %s", MAX_RETRIES, last_error)
    raise last_error  # type: ignore[misc]
