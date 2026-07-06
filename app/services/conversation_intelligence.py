"""Conversation intelligence — summarization and goal inference.

These utilities transform raw conversation history into compact,
structured context that makes LLM replies significantly smarter.

Phase 3.5: Rich conversation memory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.conversation import ConversationState

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Trigger → Goal mapping
# ──────────────────────────────────────────────────────────────────

_TRIGGER_GOALS: dict[str, str] = {
    "research_digest": "Help merchant apply insights from the latest research digest to improve their business.",
    "performance_drop": "Help merchant understand and reverse the recent drop in profile performance.",
    "review_milestone": "Help merchant celebrate and leverage their review milestone to attract more customers.",
    "new_offer": "Help merchant create or optimize a promotional offer to drive engagement.",
    "seasonal_opportunity": "Help merchant capitalize on the current seasonal trend.",
    "subscription_expiry": "Help merchant understand the value of renewing their subscription.",
    "profile_incomplete": "Help merchant complete their business profile for better visibility.",
    "competitor_activity": "Alert merchant to competitor movements and suggest counter-strategies.",
    "customer_lapse": "Help merchant re-engage a lapsing customer with a personalized offer.",
    "review_request": "Help merchant follow up with a recent customer for a review.",
    "content_suggestion": "Help merchant publish engaging content to their profile.",
}


def infer_goal(trigger_kind: str) -> str:
    """Infer a conversation goal from the trigger kind.

    Falls back to a generic goal if the trigger is not recognized.

    Args:
        trigger_kind: The kind of trigger that started this conversation.

    Returns:
        A human-readable goal description.
    """
    # Try exact match first
    goal = _TRIGGER_GOALS.get(trigger_kind)
    if goal:
        return goal

    # Try partial match (e.g., "performance_drop_views" → "performance_drop")
    for key, value in _TRIGGER_GOALS.items():
        if key in trigger_kind or trigger_kind in key:
            return value

    return f"Help merchant respond to '{trigger_kind.replace('_', ' ')}' and improve engagement."


def build_conversation_summary(state: ConversationState) -> str:
    """Build a structured summary of the conversation so far.

    Instead of sending every message verbatim, this compresses
    the history into what the merchant already knows, what
    Vera has offered, and what the current goal is.

    Args:
        state: The conversation state with full history.

    Returns:
        A compact summary string for the LLM prompt.
    """
    if not state.history:
        return ""

    sections: list[str] = []

    # ── What the merchant already knows ──────────────────────────
    merchant_knows: list[str] = []
    vera_offered: list[str] = []
    merchant_said: list[str] = []

    for msg in state.history:
        if msg.role.value == "vera":
            # Extract key facts Vera mentioned
            body = msg.body
            if len(body) > 80:
                body = body[:77] + "..."
            vera_offered.append(body)
        elif msg.role.value == "merchant":
            body = msg.body
            intent_label = f" [{msg.intent.value}]" if msg.intent else ""
            if len(body) > 60:
                body = body[:57] + "..."
            merchant_said.append(f"{body}{intent_label}")

    if vera_offered:
        sections.append("WHAT VERA HAS COMMUNICATED:\n" + "\n".join(
            f"  • {m}" for m in vera_offered
        ))

    if merchant_said:
        sections.append("WHAT THE MERCHANT REPLIED:\n" + "\n".join(
            f"  • {m}" for m in merchant_said
        ))

    # ── Conversation goal ────────────────────────────────────────
    if state.goal.description:
        status = "✅ Completed" if state.goal.completed else "🔄 In progress"
        sections.append(
            f"CONVERSATION GOAL: {state.goal.description}\n"
            f"  Status: {status}"
        )

    # ── Key facts ────────────────────────────────────────────────
    facts: list[str] = []
    facts.append(f"Trigger: {state.last_trigger_kind.replace('_', ' ')}")
    facts.append(f"Messages exchanged: {len(state.history)}")
    facts.append(f"Follow-ups sent: {state.follow_up_count}")

    if state.history:
        last = state.history[-1]
        if last.intent:
            facts.append(f"Last merchant intent: {last.intent.value}")

    sections.append("KEY FACTS:\n" + "\n".join(f"  • {f}" for f in facts))

    summary = "\n\n".join(sections)
    logger.debug("Conversation summary built: %d chars", len(summary))
    return summary
