"""Intent detector — deterministic classification of merchant replies.

Classifies obvious intents via keyword matching before falling back
to the LLM.  This avoids wasting an LLM call for simple "yes" or
"stop" replies.

Phase 3.5 enhancements:
- Confidence scores for every detection.
- Low-confidence results trigger LLM fallback.
- Expanded autoresponder patterns.
"""

from __future__ import annotations

import logging
import re

from app.models.conversation import Intent, IntentResult

logger = logging.getLogger(__name__)

# Confidence threshold — below this, fall back to LLM
CONFIDENCE_THRESHOLD = 0.6

# ──────────────────────────────────────────────────────────────────
# Keyword patterns — order matters (first match wins)
#
# Each entry: (Intent, patterns, base_confidence)
#   - Anchored patterns (^) get higher confidence.
#   - Unanchored patterns (partial match) get lower confidence.
# ──────────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[Intent, list[str], float]] = [
    # Autoresponder — check first (longest/most specific)
    (Intent.AUTORESPONDER, [
        r"automated\s+(?:response|reply|message)",
        r"auto[\s-]?reply",
        r"out\s+of\s+(?:office|station)",
        r"we(?:'ve|.ve)?\s+received\s+your\s+message",
        r"this\s+number\s+is\s+(?:un)?attended",
        r"currently\s+(?:unavailable|not\s+available)",
        r"(?:will|'ll)\s+(?:get\s+back|respond|reply)\s+(?:to\s+you\s+)?(?:soon|shortly|later)",
        r"do\s+not\s+reply\s+to\s+this",
        r"vacation\s+(?:responder|reply|mode)",
        r"(?:our|the)\s+office\s+is\s+closed",
        r"away\s+from\s+(?:the\s+)?(?:office|desk)",
        r"(?:business|office)\s+hours?\s+(?:are|is)\s+(?:over|ended|closed)",
        r"on\s+(?:leave|holiday|vacation)",
    ], 0.95),

    # Unsubscribe
    (Intent.UNSUBSCRIBE, [
        r"\bstop\b",
        r"\bunsubscribe\b",
        r"\bopt[\s-]?out\b",
        r"\bremove\s+me\b",
        r"\bdon'?t\s+(?:send|message|contact)\b",
        r"\bno\s+more\s+messages?\b",
    ], 0.95),

    # Affirmative
    (Intent.AFFIRMATIVE, [
        r"^(?:yes|yeah|yep|yup|ya|haan|ha|ok|okay|sure|alright|go\s+ahead|sounds?\s+good|let'?s?\s+(?:do\s+it|go)|tell\s+me\s+more|show\s+me|interested|definitely|absolutely)\b",
        r"^(?:ji|ji\s+haan|bilkul|zaroor|theek\s+hai|thik\s+hai|chalega)\b",
        r"^👍$",
    ], 0.95),

    # Negative
    (Intent.NEGATIVE, [
        r"^(?:no|nope|nah|nahi|na|not?\s+interested|not?\s+(?:now|right\s+now|today)|pass|skip|later|baad\s+mein)\b",
        r"^no\s+thanks?\b",
    ], 0.95),

    # Thanks
    (Intent.THANKS, [
        r"\bthanks?\b",
        r"\bthank\s+you\b",
        r"\bdhanyavaad\b",
        r"\bshukriya\b",
        r"\bthanku\b",
        r"\bthx\b",
    ], 0.90),

    # Greeting
    (Intent.GREETING, [
        r"^(?:hi|hello|hey|namaste|namaskar|namasthe)\b",
        r"^good\s+(?:morning|afternoon|evening)\b",
    ], 0.90),

    # Question
    (Intent.QUESTION, [
        r"\?$",
        r"^(?:what|how|when|where|why|which|who|can\s+you|could\s+you|is\s+it|kya|kaise|kab|kahan)\b",
        r"^(?:tell\s+me\s+about|explain|details?|more\s+info)\b",
    ], 0.85),
]

# Compiled patterns for performance
_COMPILED_PATTERNS: list[tuple[Intent, list[re.Pattern], float]] = [
    (intent, [re.compile(p, re.IGNORECASE) for p in patterns], conf)
    for intent, patterns, conf in _PATTERNS
]


def _compute_confidence(text: str, base_confidence: float, pattern: re.Pattern) -> float:
    """Adjust confidence based on message length and match quality.

    Short, exact-match messages (e.g., "yes") get the highest confidence.
    Long messages where only a keyword matched get reduced confidence.
    """
    word_count = len(text.split())

    # Single-word or two-word messages that matched → high confidence
    if word_count <= 2:
        return min(base_confidence + 0.03, 1.0)

    # Anchored pattern (starts with ^) matching → still reliable
    if pattern.pattern.startswith("^"):
        return base_confidence

    # Long message with a partial keyword match → lower confidence
    penalty = min(0.05 * (word_count - 2), 0.30)
    return max(base_confidence - penalty, 0.40)


def detect_intent_with_confidence(text: str) -> IntentResult:
    """Classify a merchant reply with confidence scoring.

    Uses deterministic keyword/regex matching.  Returns Intent.UNKNOWN
    with low confidence if no pattern matches — the caller should then
    use the LLM.

    Args:
        text: The merchant's raw reply text.

    Returns:
        An IntentResult with intent, confidence, and source.
    """
    cleaned = text.strip()

    if not cleaned:
        logger.debug("Empty message → UNKNOWN intent, confidence=0.0")
        return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)

    for intent, compiled, base_conf in _COMPILED_PATTERNS:
        for pattern in compiled:
            if pattern.search(cleaned):
                confidence = _compute_confidence(cleaned, base_conf, pattern)
                logger.info(
                    "Intent detected: %s (confidence=%.2f) — text=%.60s",
                    intent.value,
                    confidence,
                    cleaned,
                )
                return IntentResult(
                    intent=intent,
                    confidence=confidence,
                    source="rules",
                )

    logger.info("No deterministic match → UNKNOWN — text=%.60s", cleaned)
    return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)


def detect_intent(text: str) -> Intent:
    """Classify a merchant reply into an intent.

    Backward-compatible wrapper around detect_intent_with_confidence().

    Args:
        text: The merchant's raw reply text.

    Returns:
        The detected Intent enum value.
    """
    return detect_intent_with_confidence(text).intent


def is_deterministic(intent: Intent) -> bool:
    """Return True if this intent can be handled without the LLM.

    Affirmative, negative, unsubscribe, thanks, greeting, and
    autoresponder are all handled deterministically.

    Args:
        intent: The classified intent.

    Returns:
        True if the reply handler can skip the LLM.
    """
    return intent in {
        Intent.AFFIRMATIVE,
        Intent.NEGATIVE,
        Intent.UNSUBSCRIBE,
        Intent.THANKS,
        Intent.GREETING,
        Intent.AUTORESPONDER,
    }


def should_use_llm(result: IntentResult) -> bool:
    """Determine if a detected intent is uncertain enough to need the LLM.

    Args:
        result: The IntentResult from detection.

    Returns:
        True if the LLM should handle this reply instead.
    """
    if result.intent == Intent.UNKNOWN:
        return True
    if result.confidence < CONFIDENCE_THRESHOLD:
        return True
    return False

