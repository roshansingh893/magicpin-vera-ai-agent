"""Output validator — validates and parses LLM JSON responses.

Ensures the LLM output conforms to the ComposedMessage schema before
it reaches the API layer.  Raises on any violation — never silently
continues with bad data.
"""

from __future__ import annotations

import json
import logging
import re

from app.core.exceptions import VeraAgentError
from app.models.responses import ComposedMessage

logger = logging.getLogger(__name__)

# Valid values for constrained fields
VALID_CTA_VALUES = {"binary_yes_stop", "open_ended", "none"}
VALID_SEND_AS_VALUES = {"vera", "merchant_on_behalf"}

# Reasonable body length bounds
MIN_BODY_LENGTH = 10
MAX_BODY_LENGTH = 2000


class OutputValidationError(VeraAgentError):
    """Raised when the LLM output fails validation."""

    def __init__(self, message: str = "LLM output validation failed.") -> None:
        super().__init__(message=message, status_code=502)


def _clean_json_response(raw: str) -> str:
    """Strip markdown code fences and surrounding whitespace.

    Some models wrap JSON in ```json ... ``` despite instructions.
    This normalises the response before parsing.
    """
    cleaned = raw.strip()

    # Remove ```json ... ``` or ``` ... ``` wrappers
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    return cleaned.strip()


def parse_llm_response(raw: str) -> dict:
    """Parse raw LLM text into a Python dictionary.

    Args:
        raw: The raw string returned by the LLM.

    Returns:
        Parsed dictionary.

    Raises:
        OutputValidationError: If the text is not valid JSON.
    """
    cleaned = _clean_json_response(raw)

    if not cleaned:
        raise OutputValidationError("LLM returned empty output.")

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON: %s — raw: %.200s", exc, raw)
        raise OutputValidationError(
            f"LLM output is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise OutputValidationError(
            f"Expected a JSON object, got {type(data).__name__}."
        )

    return data


def validate_composed_message(data: dict) -> ComposedMessage:
    """Validate a parsed dict against the ComposedMessage schema.

    Checks required fields, correct data types, enum constraints,
    non-empty body, and reasonable body length.

    Args:
        data: Parsed JSON dictionary from the LLM.

    Returns:
        A validated ComposedMessage instance.

    Raises:
        OutputValidationError: On any validation failure.
    """
    # ── Required fields ──────────────────────────────────────────
    required = {"body", "cta", "send_as", "suppression_key", "rationale"}
    missing = required - set(data.keys())
    if missing:
        raise OutputValidationError(
            f"Missing required fields in LLM output: {', '.join(sorted(missing))}"
        )

    # ── Type checks ──────────────────────────────────────────────
    for field in required:
        if not isinstance(data[field], str):
            raise OutputValidationError(
                f"Field '{field}' must be a string, got {type(data[field]).__name__}."
            )

    # ── Non-empty body ───────────────────────────────────────────
    body = data["body"].strip()
    if not body:
        raise OutputValidationError("Field 'body' must not be empty.")

    if len(body) < MIN_BODY_LENGTH:
        raise OutputValidationError(
            f"Field 'body' is too short ({len(body)} chars, minimum {MIN_BODY_LENGTH})."
        )

    if len(body) > MAX_BODY_LENGTH:
        raise OutputValidationError(
            f"Field 'body' is too long ({len(body)} chars, maximum {MAX_BODY_LENGTH})."
        )

    # ── Enum constraints ─────────────────────────────────────────
    cta = data["cta"].strip().lower()
    if cta not in VALID_CTA_VALUES:
        raise OutputValidationError(
            f"Invalid 'cta' value: '{data['cta']}'. Must be one of: {', '.join(sorted(VALID_CTA_VALUES))}"
        )
    data["cta"] = cta  # normalise

    send_as = data["send_as"].strip().lower()
    if send_as not in VALID_SEND_AS_VALUES:
        raise OutputValidationError(
            f"Invalid 'send_as' value: '{data['send_as']}'. Must be one of: {', '.join(sorted(VALID_SEND_AS_VALUES))}"
        )
    data["send_as"] = send_as  # normalise

    # ── Suppression key ──────────────────────────────────────────
    if not data["suppression_key"].strip():
        raise OutputValidationError("Field 'suppression_key' must not be empty.")

    # ── Rationale ────────────────────────────────────────────────
    if not data["rationale"].strip():
        raise OutputValidationError("Field 'rationale' must not be empty.")

    # ── Build Pydantic model (final type safety) ─────────────────
    try:
        message = ComposedMessage(**data)
    except Exception as exc:
        raise OutputValidationError(
            f"Failed to construct ComposedMessage: {exc}"
        ) from exc

    logger.info(
        "Output validated — body=%d chars, cta=%s, send_as=%s",
        len(message.body),
        message.cta,
        message.send_as,
    )
    return message
