"""Tick handler — follow-up decision engine for POST /v1/tick.

Evaluates all active conversations and decides whether a follow-up
message should be sent.  Maximum one follow-up per conversation.

Phase 3.5 enhancements:
- Contextual follow-ups based on trigger and conversation history.
- Expiry-based auto-close on each tick.
- Richer logging.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.models.conversation import ConversationStage, ConversationState
from app.models.responses import ComposedMessage
from app.services.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

# Minimum hours since last interaction before sending a follow-up
FOLLOW_UP_DELAY_HOURS = 24

# Maximum follow-ups per conversation (do not spam)
MAX_FOLLOW_UPS = 1

# ──────────────────────────────────────────────────────────────────
# Trigger-specific follow-up templates
# ──────────────────────────────────────────────────────────────────

_TRIGGER_FOLLOW_UPS: dict[str, str] = {
    "performance_drop": (
        "Hi! Just a quick follow-up — your profile views are still trending "
        "down. I have a few quick optimizations that could help reverse this. "
        "Reply YES to see them."
    ),
    "research_digest": (
        "Hi! I noticed you haven't had a chance to review the insights I "
        "shared. They're still relevant and could help your business this "
        "week. Want me to highlight the key takeaway? Reply YES."
    ),
    "seasonal_opportunity": (
        "Hi! The seasonal window I mentioned is still open. There's still "
        "time to launch a targeted campaign and capture this demand. "
        "Reply YES if you'd like help setting it up."
    ),
    "subscription_expiry": (
        "Hi! Just a reminder — your subscription benefits are expiring soon. "
        "Renewing now ensures your visibility doesn't drop. "
        "Reply YES to learn about renewal options."
    ),
    "review_milestone": (
        "Hi! Your review milestone is a great achievement worth sharing. "
        "I can help you create a celebratory post for your profile. "
        "Reply POST to approve it."
    ),
    "new_offer": (
        "Hi! I still have suggestions ready to optimize your offer for "
        "maximum visibility. Reply YES and I'll share them right away."
    ),
    "profile_incomplete": (
        "Hi! Completing your profile can significantly boost your "
        "visibility in local search results. I can guide you through "
        "the quick steps. Reply YES to get started."
    ),
    "customer_lapse": (
        "Hi! Your lapsing customer is still reachable. I have a "
        "personalized re-engagement message ready. Reply YES to send it."
    ),
}


def _needs_follow_up(state: ConversationState, now: datetime) -> bool:
    """Determine whether a conversation needs a follow-up.

    Criteria:
    1. Stage must be WAITING_REPLY (merchant hasn't replied yet).
    2. No follow-ups have been sent yet (max 1).
    3. Enough time has passed since the last interaction.

    Args:
        state: The conversation to evaluate.
        now: The current UTC datetime.

    Returns:
        True if a follow-up should be sent.
    """
    # Only follow up conversations that are waiting
    if state.stage != ConversationStage.WAITING_REPLY:
        return False

    # Don't exceed the follow-up limit
    if state.follow_up_count >= MAX_FOLLOW_UPS:
        return False

    # Check time elapsed
    hours_since_update = (now - state.updated_at).total_seconds() / 3600
    if hours_since_update < FOLLOW_UP_DELAY_HOURS:
        return False

    return True


def _generate_follow_up(state: ConversationState) -> str:
    """Generate a contextual follow-up message body.

    Uses trigger-specific templates when available, falls back to
    a generic template that references the original trigger and
    conversation goal.

    Args:
        state: The conversation state.

    Returns:
        The follow-up message body.
    """
    # Try trigger-specific template first
    trigger = state.last_trigger_kind
    template = _TRIGGER_FOLLOW_UPS.get(trigger)
    if template:
        return template

    # Try partial match
    for key, tmpl in _TRIGGER_FOLLOW_UPS.items():
        if key in trigger or trigger in key:
            return tmpl

    # Fallback: use goal if available
    trigger_display = trigger.replace("_", " ")
    if state.goal.description:
        return (
            f"Hi! Just following up on my earlier message. "
            f"I'd still like to {state.goal.description.lower().rstrip('.')}. "
            f"Reply YES and I'll share my suggestions right away."
        )

    return (
        f"Hi! Just following up on my earlier message about {trigger_display}. "
        f"I still have personalized suggestions ready for you if you'd "
        f"like to see them. Just reply YES and I'll share them right away."
    )


def process_tick(
    timestamp: str = "",
    merchant_ids: list[str] | None = None,
) -> list[dict]:
    """Process a tick event and generate follow-up actions.

    Also runs expiry checks to auto-close stale conversations.

    Args:
        timestamp: ISO timestamp of the tick (for logging).
        merchant_ids: Optional list of merchant IDs to check.
                      If None, checks all active conversations.

    Returns:
        A list of action dictionaries with conversation_id, action,
        and optional message.
    """
    now = datetime.now(timezone.utc)
    actions: list[dict] = []

    logger.info("Processing tick — timestamp=%s", timestamp or "now")

    # ── Phase 3.5: Auto-close expired conversations ──────────────
    closed = ConversationManager.check_and_close_expired()
    if closed:
        logger.info("Tick: auto-closed %d expired conversations", len(closed))

    if merchant_ids:
        # Check specific merchants
        conversations = []
        for mid in merchant_ids:
            conversations.extend(
                ConversationManager.find_by_merchant(mid, active_only=True)
            )
    else:
        # Check all active conversations
        conversations = list(_get_all_active())

    for state in conversations:
        if _needs_follow_up(state, now):
            follow_up_body = _generate_follow_up(state)

            # Record the follow-up and transition state
            ConversationManager.append_bot_message(
                state.conversation_id,
                follow_up_body,
                is_follow_up=True,
            )

            actions.append({
                "conversation_id": state.conversation_id,
                "merchant_id": state.merchant_id,
                "action": "send_follow_up",
                "message": ComposedMessage(
                    body=follow_up_body,
                    cta="binary_yes_stop",
                    send_as="vera",
                    suppression_key=f"follow_up:{state.conversation_id}",
                    rationale=f"Follow-up after {FOLLOW_UP_DELAY_HOURS}h with no reply. "
                              f"Original trigger: {state.last_trigger_kind}.",
                ),
            })

            logger.info(
                "Follow-up queued: conversation=%s merchant=%s trigger=%s",
                state.conversation_id,
                state.merchant_id,
                state.last_trigger_kind,
            )
        else:
            actions.append({
                "conversation_id": state.conversation_id,
                "merchant_id": state.merchant_id,
                "action": "no_action",
            })

    if not conversations:
        logger.info("Tick: no active conversations found.")

    return actions


def _get_all_active() -> list[ConversationState]:
    """Get all non-closed conversations from the store.

    This is a workaround for the in-memory store. In production,
    the store would support a native scan operation.
    """
    from app.services.conversation_manager import _conversation_store

    return [
        s for s in _conversation_store.values()
        if s.stage != ConversationStage.CLOSED
    ]
