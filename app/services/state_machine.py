"""Conversation state machine — explicit stage transitions.

All stage transitions are defined in a single transition table.
No scattered if-else logic.  The state machine is pure — it mutates
the ConversationState and nothing else.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.models.conversation import ConversationStage, Intent

if TYPE_CHECKING:
    from app.models.conversation import ConversationState

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Transition table
#   (current_stage, event) → next_stage
# ──────────────────────────────────────────────────────────────────

_TRANSITIONS: dict[tuple[ConversationStage, str], ConversationStage] = {
    # Initial message sent
    (ConversationStage.NEW, "message_sent"):          ConversationStage.MESSAGE_SENT,
    (ConversationStage.MESSAGE_SENT, "wait"):         ConversationStage.WAITING_REPLY,

    # Merchant replies
    (ConversationStage.WAITING_REPLY, "reply_received"):   ConversationStage.RESOLVED,
    (ConversationStage.FOLLOW_UP_SENT, "reply_received"):  ConversationStage.RESOLVED,

    # Follow-up sent
    (ConversationStage.WAITING_REPLY, "follow_up_sent"):   ConversationStage.FOLLOW_UP_SENT,

    # Closing events
    (ConversationStage.WAITING_REPLY, "close"):       ConversationStage.CLOSED,
    (ConversationStage.FOLLOW_UP_SENT, "close"):      ConversationStage.CLOSED,
    (ConversationStage.RESOLVED, "close"):             ConversationStage.CLOSED,
    (ConversationStage.RESOLVED, "reply_received"):    ConversationStage.RESOLVED,

    # Resolved → send another follow-up (if conversation continues)
    (ConversationStage.RESOLVED, "message_sent"):      ConversationStage.MESSAGE_SENT,
}


class InvalidTransitionError(Exception):
    """Raised when a stage transition is not allowed."""


def transition(state: ConversationState, event: str) -> ConversationStage:
    """Apply a transition event to a conversation.

    Args:
        state: The current conversation state (will be mutated).
        event: The event name triggering the transition.

    Returns:
        The new stage after transition.

    Raises:
        InvalidTransitionError: If the transition is not in the table.
    """
    key = (state.stage, event)
    next_stage = _TRANSITIONS.get(key)

    if next_stage is None:
        raise InvalidTransitionError(
            f"Invalid transition: {state.stage.value} + '{event}'. "
            f"Conversation {state.conversation_id}."
        )

    old_stage = state.stage
    state.stage = next_stage
    state.touch()

    logger.info(
        "State transition: %s → %s (event=%s, conversation=%s)",
        old_stage.value,
        next_stage.value,
        event,
        state.conversation_id,
    )
    return next_stage
