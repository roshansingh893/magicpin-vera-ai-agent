"""Conversation manager — CRUD for conversation state.

All conversation storage is encapsulated here.  No other module
accesses the underlying store directly.  The in-memory dict can
be replaced with Redis by changing only this file.

Phase 3.5 enhancements:
- Goal inference on conversation creation.
- Conversation summary updates after each message.
- Confidence tracking on merchant replies.
- Conversation expiry checks.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from app.models.conversation import (
    ConversationGoal,
    ConversationStage,
    ConversationState,
    Message,
    MessageRole,
)
from app.services.state_machine import transition

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# In-memory store — encapsulated, never imported directly
# ──────────────────────────────────────────────────────────────────
_conversation_store: dict[str, ConversationState] = {}


class ConversationNotFoundError(Exception):
    """Raised when a conversation ID is not in the store."""


class ConversationManager:
    """Manages conversation lifecycle and state persistence.

    All storage operations go through this class so the backend
    can be swapped (e.g., to Redis) without touching business logic.
    """

    # ── Create ───────────────────────────────────────────────────

    @staticmethod
    def create(
        merchant_id: str,
        trigger_id: str,
        trigger_kind: str,
        bot_message: str,
        customer_id: str | None = None,
    ) -> ConversationState:
        """Create a new conversation and persist it.

        The conversation starts in MESSAGE_SENT → WAITING_REPLY stage
        after the initial bot message is recorded.

        Args:
            merchant_id: The merchant this conversation is with.
            trigger_id: ID of the trigger that started the conversation.
            trigger_kind: Kind of trigger (e.g., 'research_digest').
            bot_message: The initial message Vera sent.
            customer_id: Optional customer ID for customer-scoped conversations.

        Returns:
            The newly created ConversationState.
        """
        # Infer goal from trigger
        from app.services.conversation_intelligence import infer_goal
        goal_description = infer_goal(trigger_kind)

        state = ConversationState(
            merchant_id=merchant_id,
            customer_id=customer_id,
            last_trigger_id=trigger_id,
            last_trigger_kind=trigger_kind,
            last_bot_message=bot_message,
            goal=ConversationGoal(description=goal_description),
        )

        # Record the bot message in history
        state.history.append(Message(
            role=MessageRole.VERA,
            body=bot_message,
        ))

        # Transition: NEW → MESSAGE_SENT → WAITING_REPLY
        transition(state, "message_sent")
        transition(state, "wait")

        _conversation_store[state.conversation_id] = state

        logger.info(
            "Conversation created: id=%s merchant=%s trigger=%s goal='%s'",
            state.conversation_id,
            merchant_id,
            trigger_kind,
            goal_description[:60],
        )
        return state

    # ── Read ─────────────────────────────────────────────────────

    @staticmethod
    def get(conversation_id: str) -> ConversationState:
        """Retrieve a conversation by ID.

        Args:
            conversation_id: The unique conversation identifier.

        Returns:
            The ConversationState.

        Raises:
            ConversationNotFoundError: If the conversation doesn't exist.
        """
        state = _conversation_store.get(conversation_id)
        if state is None:
            raise ConversationNotFoundError(
                f"Conversation not found: {conversation_id}"
            )
        return state

    @staticmethod
    def find_by_merchant(
        merchant_id: str,
        active_only: bool = True,
    ) -> list[ConversationState]:
        """Find all conversations for a merchant.

        Args:
            merchant_id: The merchant to search for.
            active_only: If True, exclude CLOSED conversations.

        Returns:
            List of matching ConversationState objects.
        """
        results = []
        for state in _conversation_store.values():
            if state.merchant_id != merchant_id:
                continue
            if active_only and state.stage == ConversationStage.CLOSED:
                continue
            results.append(state)
        return results

    # ── Update ───────────────────────────────────────────────────

    @staticmethod
    def append_merchant_reply(
        conversation_id: str,
        body: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> ConversationState:
        """Record a merchant reply and update conversation state.

        Args:
            conversation_id: The conversation to update.
            body: The merchant's reply text.
            intent: Optional classified intent string.
            confidence: Optional confidence score for the intent.

        Returns:
            The updated ConversationState.
        """
        from app.models.conversation import Intent

        state = ConversationManager.get(conversation_id)

        intent_enum = None
        if intent:
            try:
                intent_enum = Intent(intent)
            except ValueError:
                intent_enum = Intent.UNKNOWN

        state.history.append(Message(
            role=MessageRole.MERCHANT,
            body=body,
            intent=intent_enum,
            confidence=confidence,
        ))

        # Transition to RESOLVED on merchant reply
        transition(state, "reply_received")

        # Update conversation summary
        ConversationManager._update_summary(state)
        state.touch()

        logger.info(
            "Merchant reply recorded: conversation=%s intent=%s confidence=%.2f",
            conversation_id,
            intent or "none",
            confidence or 0.0,
        )
        return state

    @staticmethod
    def append_bot_message(
        conversation_id: str,
        body: str,
        is_follow_up: bool = False,
    ) -> ConversationState:
        """Record a bot response and update conversation state.

        Args:
            conversation_id: The conversation to update.
            body: The bot's reply text.
            is_follow_up: Whether this is a follow-up (tick) message.

        Returns:
            The updated ConversationState.
        """
        state = ConversationManager.get(conversation_id)

        state.history.append(Message(
            role=MessageRole.VERA,
            body=body,
        ))
        state.last_bot_message = body

        if is_follow_up:
            state.follow_up_count += 1
            transition(state, "follow_up_sent")
        else:
            transition(state, "message_sent")
            transition(state, "wait")

        state.touch()

        logger.info(
            "Bot message appended: conversation=%s follow_up=%s",
            conversation_id,
            is_follow_up,
        )
        return state

    # ── Close ────────────────────────────────────────────────────

    @staticmethod
    def close(conversation_id: str, reason: str = "") -> ConversationState:
        """Close a conversation.

        Args:
            conversation_id: The conversation to close.
            reason: Optional reason for closing.

        Returns:
            The updated ConversationState.
        """
        state = ConversationManager.get(conversation_id)
        transition(state, "close")

        if reason:
            state.metadata["close_reason"] = reason

        state.touch()
        logger.info(
            "Conversation closed: id=%s reason=%s",
            conversation_id,
            reason or "none",
        )
        return state

    # ── Summary ───────────────────────────────────────────────────

    @staticmethod
    def _update_summary(state: ConversationState) -> None:
        """Rebuild the conversation summary after a state change."""
        from app.services.conversation_intelligence import build_conversation_summary
        state.summary = build_conversation_summary(state)

    # ── Goal ─────────────────────────────────────────────────────

    @staticmethod
    def complete_goal(conversation_id: str) -> ConversationState:
        """Mark the conversation goal as completed."""
        from datetime import datetime, timezone
        state = ConversationManager.get(conversation_id)
        state.goal.completed = True
        state.goal.completed_at = datetime.now(timezone.utc)
        state.touch()
        logger.info("Goal completed: conversation=%s", conversation_id)
        return state

    # ── Expiry ───────────────────────────────────────────────────

    @staticmethod
    def check_and_close_expired() -> list[str]:
        """Close all expired conversations. Returns list of closed IDs."""
        closed_ids = []
        for state in list(_conversation_store.values()):
            should_close, reason = state.should_auto_close()
            if should_close:
                transition(state, "close")
                state.metadata["close_reason"] = reason
                state.touch()
                closed_ids.append(state.conversation_id)
                logger.info(
                    "Auto-closed conversation: id=%s reason=%s",
                    state.conversation_id,
                    reason,
                )
        return closed_ids

    # ── Store Management (testing) ───────────────────────────────

    @staticmethod
    def clear_store() -> None:
        """Clear the entire conversation store. For testing only."""
        _conversation_store.clear()

    @staticmethod
    def store_size() -> int:
        """Return the number of conversations in the store."""
        return len(_conversation_store)
