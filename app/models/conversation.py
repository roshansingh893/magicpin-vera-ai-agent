"""Conversation domain models — state, stages, and message objects.

These models represent the conversation lifecycle between Vera and
a merchant (or customer).  They are pure data — no business logic,
no storage access.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────

class ConversationStage(str, Enum):
    """Finite state machine stages for a conversation."""
    NEW = "new"
    MESSAGE_SENT = "message_sent"
    WAITING_REPLY = "waiting_reply"
    FOLLOW_UP_SENT = "follow_up_sent"
    RESOLVED = "resolved"
    CLOSED = "closed"


class MessageRole(str, Enum):
    """Who authored a message in the conversation."""
    VERA = "vera"
    MERCHANT = "merchant"
    SYSTEM = "system"


class Intent(str, Enum):
    """Classified intent of a merchant reply."""
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    QUESTION = "question"
    UNSUBSCRIBE = "unsubscribe"
    THANKS = "thanks"
    GREETING = "greeting"
    AUTORESPONDER = "autoresponder"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────────
# Intent Result (with confidence)
# ──────────────────────────────────────────────────────────────────

class IntentResult(BaseModel):
    """Result of intent classification — includes confidence score.

    A confidence below the threshold (0.6) signals that the reply
    handler should fall back to the LLM instead of using the
    deterministic response template.
    """
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "rules"  # "rules" or "llm"


# ──────────────────────────────────────────────────────────────────
# Conversation Goal
# ──────────────────────────────────────────────────────────────────

class ConversationGoal(BaseModel):
    """What Vera is trying to achieve in this conversation.

    The AI should know its objective and work toward completion.
    """
    description: str = ""
    completed: bool = False
    completed_at: Optional[datetime] = None


# ──────────────────────────────────────────────────────────────────
# Message
# ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    """A single message in a conversation thread."""
    role: MessageRole
    body: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    intent: Optional[Intent] = None
    confidence: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Conversation State
# ──────────────────────────────────────────────────────────────────

# Default expiry: 72 hours
CONVERSATION_EXPIRY_HOURS = 72

# Maximum ignored follow-ups before auto-close
MAX_IGNORED_FOLLOW_UPS = 1


class ConversationState(BaseModel):
    """Full state of a conversation — the single source of truth.

    Designed to be serializable so it can be migrated from in-memory
    storage to Redis without changing business logic.
    """
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    merchant_id: str
    customer_id: Optional[str] = None
    stage: ConversationStage = ConversationStage.NEW
    last_trigger_id: str = ""
    last_trigger_kind: str = ""
    last_bot_message: str = ""
    history: list[Message] = Field(default_factory=list)
    follow_up_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Phase 3.5 — Intelligence enhancements
    goal: ConversationGoal = Field(default_factory=ConversationGoal)
    summary: str = ""

    def touch(self) -> None:
        """Update the ``updated_at`` timestamp to now."""
        self.updated_at = datetime.now(timezone.utc)

    def is_expired(self, now: datetime | None = None) -> bool:
        """Return True if this conversation has exceeded its TTL."""
        now = now or datetime.now(timezone.utc)
        hours_alive = (now - self.created_at).total_seconds() / 3600
        return hours_alive >= CONVERSATION_EXPIRY_HOURS

    def should_auto_close(self, now: datetime | None = None) -> tuple[bool, str]:
        """Check if this conversation should be automatically closed.

        Returns:
            (should_close, reason) tuple.
        """
        now = now or datetime.now(timezone.utc)

        if self.stage == ConversationStage.CLOSED:
            return False, ""

        if self.is_expired(now):
            return True, "expired"

        if (self.stage == ConversationStage.FOLLOW_UP_SENT
                and self.follow_up_count >= MAX_IGNORED_FOLLOW_UPS):
            # Already sent max follow-ups and still no reply
            hours_since = (now - self.updated_at).total_seconds() / 3600
            if hours_since >= 48:
                return True, "follow_ups_ignored"

        if self.goal.completed:
            return True, "goal_completed"

        return False, ""
