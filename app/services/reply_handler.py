"""Reply handler — orchestrator for multi-turn merchant replies.

This is the main entry point for POST /v1/reply.  It:
1. Retrieves the conversation.
2. Detects the merchant's intent (with confidence).
3. Handles high-confidence deterministic intents without the LLM.
4. Falls back to the LLM for low-confidence or complex replies.
5. Updates conversation state.
6. Returns a response.

Phase 3.5 enhancements:
- Confidence-based LLM fallback.
- Goal-aware response selection.
- Richer logging (intent, confidence, latency).

Orchestration only — no state logic, no LLM protocol.
"""

from __future__ import annotations

import logging
import time

from app.models.conversation import Intent, ConversationStage
from app.models.responses import ComposedMessage
from app.prompts.reply_prompt import build_reply_prompt
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.services.conversation_manager import ConversationManager
from app.services.intent_detector import (
    detect_intent_with_confidence,
    is_deterministic,
    should_use_llm,
)
from app.services.output_validator import (
    OutputValidationError,
    parse_llm_response,
    validate_composed_message,
)

logger = logging.getLogger(__name__)

# Maximum retries for LLM reply generation
MAX_REPLY_RETRIES = 2

# ──────────────────────────────────────────────────────────────────
# Deterministic response templates
# ──────────────────────────────────────────────────────────────────

_DETERMINISTIC_RESPONSES: dict[Intent, str] = {
    Intent.AFFIRMATIVE: (
        "Great! Let me put together those suggestions for you. "
        "I'll have them ready shortly."
    ),
    Intent.NEGATIVE: (
        "No problem at all. I'll check back with you when I have "
        "something more relevant. Have a good day!"
    ),
    Intent.UNSUBSCRIBE: (
        "Understood. I've noted your preference and won't send "
        "further messages. If you ever want to reconnect, just "
        "reply HI."
    ),
    Intent.THANKS: (
        "You're welcome! Feel free to reach out anytime you need "
        "help with your business growth. I'm here."
    ),
    Intent.GREETING: (
        "Hello! Good to hear from you. Is there anything specific "
        "I can help you with today?"
    ),
    Intent.AUTORESPONDER: "",  # No reply for autoresponders
}


def _build_deterministic_reply(intent: Intent) -> ComposedMessage | None:
    """Build a pre-written reply for a deterministic intent.

    Returns None for autoresponder (we don't reply).
    """
    body = _DETERMINISTIC_RESPONSES.get(intent, "")

    if not body:
        return None  # Autoresponder — don't reply

    # Map intent to appropriate CTA type
    cta_map = {
        Intent.AFFIRMATIVE: "none",
        Intent.NEGATIVE: "none",
        Intent.UNSUBSCRIBE: "none",
        Intent.THANKS: "none",
        Intent.GREETING: "open_ended",
    }

    return ComposedMessage(
        body=body,
        cta=cta_map.get(intent, "none"),
        send_as="vera",
        suppression_key=f"reply:{intent.value}",
        rationale=f"Deterministic response to '{intent.value}' intent. No LLM needed.",
    )


async def handle_merchant_reply(
    conversation_id: str,
    merchant_message: str,
) -> tuple[ComposedMessage | None, Intent, str]:
    """Process a merchant reply and return the bot's response.

    Args:
        conversation_id: The conversation to continue.
        merchant_message: The merchant's reply text.

    Returns:
        A tuple of (reply_message, detected_intent, conversation_id).
        reply_message is None if no reply should be sent (e.g., autoresponder).

    Raises:
        ConversationNotFoundError: If the conversation doesn't exist.
        OutputValidationError: If the LLM output fails validation after retries.
    """
    logger.info(
        "Handling reply: conversation=%s message=%.60s",
        conversation_id,
        merchant_message,
    )

    # ── 1. Detect intent with confidence ─────────────────────────
    intent_result = detect_intent_with_confidence(merchant_message)
    intent = intent_result.intent
    confidence = intent_result.confidence

    logger.info(
        "Intent: %s (confidence=%.2f, source=%s) for conversation %s",
        intent.value,
        confidence,
        intent_result.source,
        conversation_id,
    )

    # ── 2. Record the merchant reply (with confidence) ───────────
    state = ConversationManager.append_merchant_reply(
        conversation_id, merchant_message, intent.value, confidence
    )

    # ── 3. Handle autoresponder — close/pause, no reply ──────────
    if intent == Intent.AUTORESPONDER:
        ConversationManager.close(conversation_id, reason="autoresponder_detected")
        logger.info("Autoresponder detected — conversation closed: %s", conversation_id)
        return None, intent, conversation_id

    # ── 4. Handle unsubscribe — reply and close ──────────────────
    if intent == Intent.UNSUBSCRIBE:
        reply = _build_deterministic_reply(intent)
        if reply:
            ConversationManager.append_bot_message(conversation_id, reply.body)
        ConversationManager.close(conversation_id, reason="merchant_unsubscribed")
        logger.info("Unsubscribe — conversation closed: %s", conversation_id)
        return reply, intent, conversation_id

    # ── 5. Confidence-based routing ──────────────────────────────
    #   High confidence deterministic → handle locally
    #   Low confidence or UNKNOWN/QUESTION → use LLM
    use_llm = should_use_llm(intent_result)

    if not use_llm and is_deterministic(intent):
        reply = _build_deterministic_reply(intent)
        if reply:
            ConversationManager.append_bot_message(conversation_id, reply.body)
        logger.info(
            "Deterministic reply sent: intent=%s confidence=%.2f",
            intent.value,
            confidence,
        )
        return reply, intent, conversation_id

    # ── 6. Fall back to LLM ──────────────────────────────────────
    logger.info(
        "LLM fallback: intent=%s confidence=%.2f requires generated reply",
        intent.value,
        confidence,
    )
    start_time = time.monotonic()
    reply = await _generate_llm_reply(state)
    elapsed_ms = (time.monotonic() - start_time) * 1000

    logger.info("LLM reply generated in %.0fms", elapsed_ms)

    if reply:
        ConversationManager.append_bot_message(conversation_id, reply.body)

    return reply, intent, conversation_id


async def _generate_llm_reply(state) -> ComposedMessage | None:
    """Generate an LLM-powered reply for complex merchant messages.

    Uses the reply prompt builder and the same Groq client as compose().
    """
    from app.services.composer import _get_llm_client

    user_prompt = build_reply_prompt(state)
    client = _get_llm_client()
    last_error: Exception | None = None

    for attempt in range(1, MAX_REPLY_RETRIES + 1):
        logger.info("LLM reply attempt %d/%d", attempt, MAX_REPLY_RETRIES)

        raw_response = await client.generate(SYSTEM_PROMPT, user_prompt)
        logger.debug("Raw LLM reply (attempt %d): %.300s", attempt, raw_response)

        try:
            parsed = parse_llm_response(raw_response)
            message = validate_composed_message(parsed)
            return message
        except OutputValidationError as exc:
            last_error = exc
            logger.warning(
                "Reply validation failed on attempt %d/%d: %s",
                attempt,
                MAX_REPLY_RETRIES,
                exc.message,
            )

    logger.error("LLM reply failed after %d attempts: %s", MAX_REPLY_RETRIES, last_error)
    raise last_error  # type: ignore[misc]
