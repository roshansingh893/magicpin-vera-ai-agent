"""Phase 3.5 tests — conversation intelligence enhancements.

Tests for:
- Conversation summaries
- Goal inference and completion
- Intent confidence scoring
- Conversation expiry
- Dynamic follow-ups
- Enhanced autoresponder detection

All previous Phase 3 tests must still pass alongside these.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from app.models.conversation import (
    CONVERSATION_EXPIRY_HOURS,
    ConversationGoal,
    ConversationStage,
    ConversationState,
    Intent,
    IntentResult,
    Message,
    MessageRole,
)
from app.services.conversation_intelligence import (
    build_conversation_summary,
    infer_goal,
)
from app.services.conversation_manager import ConversationManager
from app.services.intent_detector import (
    CONFIDENCE_THRESHOLD,
    detect_intent_with_confidence,
    should_use_llm,
)
from app.services.tick_handler import _generate_follow_up


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_store():
    """Clear the conversation store before and after each test."""
    ConversationManager.clear_store()
    yield
    ConversationManager.clear_store()


# ──────────────────────────────────────────────────────────────────
# Goal Inference Tests
# ──────────────────────────────────────────────────────────────────

class TestGoalInference:
    """Test trigger → goal mapping."""

    def test_known_trigger(self) -> None:
        goal = infer_goal("performance_drop")
        assert "reverse" in goal.lower() or "drop" in goal.lower()

    def test_research_digest_trigger(self) -> None:
        goal = infer_goal("research_digest")
        assert "research" in goal.lower() or "insights" in goal.lower()

    def test_unknown_trigger_fallback(self) -> None:
        goal = infer_goal("some_unknown_trigger")
        assert "some unknown trigger" in goal.lower()

    def test_partial_match_trigger(self) -> None:
        """Partial matches should still find a goal."""
        goal = infer_goal("performance_drop_views")
        assert len(goal) > 10

    def test_goal_set_on_create(self) -> None:
        """ConversationManager.create should auto-set a goal."""
        conv = ConversationManager.create(
            "m_001", "t1", "performance_drop", "Your views dropped."
        )
        assert conv.goal.description != ""
        assert "drop" in conv.goal.description.lower() or "reverse" in conv.goal.description.lower()


# ──────────────────────────────────────────────────────────────────
# Goal Completion Tests
# ──────────────────────────────────────────────────────────────────

class TestGoalCompletion:
    """Test goal lifecycle."""

    def test_complete_goal(self) -> None:
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        ConversationManager.complete_goal(conv.conversation_id)
        state = ConversationManager.get(conv.conversation_id)
        assert state.goal.completed is True
        assert state.goal.completed_at is not None

    def test_completed_goal_triggers_auto_close(self) -> None:
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        ConversationManager.complete_goal(conv.conversation_id)
        state = ConversationManager.get(conv.conversation_id)
        should_close, reason = state.should_auto_close()
        assert should_close is True
        assert reason == "goal_completed"


# ──────────────────────────────────────────────────────────────────
# Conversation Summary Tests
# ──────────────────────────────────────────────────────────────────

class TestConversationSummary:
    """Test conversation summarization."""

    def test_empty_history(self) -> None:
        state = ConversationState(merchant_id="m_001")
        summary = build_conversation_summary(state)
        assert summary == ""

    def test_single_bot_message(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            goal=ConversationGoal(description="Help with profile"),
        )
        state.history.append(Message(role=MessageRole.VERA, body="Hello merchant!"))
        summary = build_conversation_summary(state)
        assert "WHAT VERA HAS COMMUNICATED" in summary
        assert "Hello merchant!" in summary

    def test_bot_and_merchant_exchange(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            last_trigger_kind="performance_drop",
            goal=ConversationGoal(description="Help with profile"),
        )
        state.history.append(Message(role=MessageRole.VERA, body="Views dropped."))
        state.history.append(Message(
            role=MessageRole.MERCHANT, body="Why?", intent=Intent.QUESTION
        ))
        summary = build_conversation_summary(state)
        assert "WHAT THE MERCHANT REPLIED" in summary
        assert "Why?" in summary
        assert "[question]" in summary

    def test_summary_includes_goal(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            goal=ConversationGoal(description="Improve Google profile"),
        )
        state.history.append(Message(role=MessageRole.VERA, body="Hello!"))
        summary = build_conversation_summary(state)
        assert "CONVERSATION GOAL" in summary
        assert "Improve Google profile" in summary

    def test_summary_includes_key_facts(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            last_trigger_kind="review_milestone",
            follow_up_count=1,
        )
        state.history.append(Message(role=MessageRole.VERA, body="Congrats!"))
        summary = build_conversation_summary(state)
        assert "KEY FACTS" in summary
        assert "review milestone" in summary

    def test_summary_updated_on_merchant_reply(self) -> None:
        """Summary should update when a merchant reply is recorded."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        ConversationManager.append_merchant_reply(
            conv.conversation_id, "Yes!", "affirmative"
        )
        state = ConversationManager.get(conv.conversation_id)
        assert state.summary != ""
        assert "Yes!" in state.summary


# ──────────────────────────────────────────────────────────────────
# Intent Confidence Tests
# ──────────────────────────────────────────────────────────────────

class TestIntentConfidence:
    """Test confidence scoring in intent detection."""

    def test_short_exact_match_high_confidence(self) -> None:
        result = detect_intent_with_confidence("yes")
        assert result.intent == Intent.AFFIRMATIVE
        assert result.confidence >= 0.95

    def test_short_message_gets_boosted(self) -> None:
        result = detect_intent_with_confidence("ok")
        assert result.confidence >= 0.95

    def test_long_message_with_keyword_lower_confidence(self) -> None:
        """A long message where 'thanks' appears should have lower confidence."""
        result = detect_intent_with_confidence(
            "I was thinking about the proposal you sent and thanks for sharing but I need more details on pricing"
        )
        # 'thanks' is unanchored, so it matches but with lower confidence
        assert result.intent == Intent.THANKS
        assert result.confidence < 0.90

    def test_unknown_returns_zero_confidence(self) -> None:
        result = detect_intent_with_confidence(
            "I need to think about the strategy for next quarter"
        )
        assert result.intent == Intent.UNKNOWN
        assert result.confidence == 0.0

    def test_empty_returns_zero_confidence(self) -> None:
        result = detect_intent_with_confidence("")
        assert result.confidence == 0.0

    def test_result_includes_source(self) -> None:
        result = detect_intent_with_confidence("yes")
        assert result.source == "rules"

    def test_should_use_llm_for_unknown(self) -> None:
        result = IntentResult(intent=Intent.UNKNOWN, confidence=0.0)
        assert should_use_llm(result) is True

    def test_should_use_llm_for_low_confidence(self) -> None:
        result = IntentResult(intent=Intent.AFFIRMATIVE, confidence=0.4)
        assert should_use_llm(result) is True

    def test_should_not_use_llm_for_high_confidence(self) -> None:
        result = IntentResult(intent=Intent.AFFIRMATIVE, confidence=0.95)
        assert should_use_llm(result) is False

    def test_confidence_stored_in_message(self) -> None:
        """Confidence should be stored in the message history."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        ConversationManager.append_merchant_reply(
            conv.conversation_id, "yes", "affirmative", 0.98
        )
        state = ConversationManager.get(conv.conversation_id)
        merchant_msg = state.history[1]
        assert merchant_msg.confidence == 0.98


# ──────────────────────────────────────────────────────────────────
# Conversation Expiry Tests
# ──────────────────────────────────────────────────────────────────

class TestConversationExpiry:
    """Test conversation auto-close rules."""

    def test_not_expired_when_fresh(self) -> None:
        state = ConversationState(merchant_id="m_001")
        assert state.is_expired() is False

    def test_expired_after_ttl(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            created_at=datetime.now(timezone.utc) - timedelta(hours=CONVERSATION_EXPIRY_HOURS + 1),
        )
        assert state.is_expired() is True

    def test_auto_close_expired(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            stage=ConversationStage.WAITING_REPLY,
            created_at=datetime.now(timezone.utc) - timedelta(hours=CONVERSATION_EXPIRY_HOURS + 1),
        )
        should_close, reason = state.should_auto_close()
        assert should_close is True
        assert reason == "expired"

    def test_auto_close_ignored_follow_ups(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            stage=ConversationStage.FOLLOW_UP_SENT,
            follow_up_count=1,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=49),
        )
        should_close, reason = state.should_auto_close()
        assert should_close is True
        assert reason == "follow_ups_ignored"

    def test_no_auto_close_if_already_closed(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            stage=ConversationStage.CLOSED,
        )
        should_close, _ = state.should_auto_close()
        assert should_close is False

    def test_check_and_close_expired_manager(self) -> None:
        """Manager should close expired conversations."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        state = ConversationManager.get(conv.conversation_id)
        state.created_at = datetime.now(timezone.utc) - timedelta(hours=CONVERSATION_EXPIRY_HOURS + 1)

        closed = ConversationManager.check_and_close_expired()
        assert conv.conversation_id in closed
        updated = ConversationManager.get(conv.conversation_id)
        assert updated.stage == ConversationStage.CLOSED
        assert updated.metadata.get("close_reason") == "expired"


# ──────────────────────────────────────────────────────────────────
# Dynamic Follow-up Tests
# ──────────────────────────────────────────────────────────────────

class TestDynamicFollowUps:
    """Test contextual follow-up message generation."""

    def test_performance_drop_follow_up(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            last_trigger_kind="performance_drop",
        )
        body = _generate_follow_up(state)
        assert "views" in body.lower() or "profile" in body.lower()
        assert "YES" in body

    def test_seasonal_follow_up(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            last_trigger_kind="seasonal_opportunity",
        )
        body = _generate_follow_up(state)
        assert "seasonal" in body.lower() or "window" in body.lower()

    def test_research_digest_follow_up(self) -> None:
        state = ConversationState(
            merchant_id="m_001",
            last_trigger_kind="research_digest",
        )
        body = _generate_follow_up(state)
        assert "insights" in body.lower() or "review" in body.lower()

    def test_unknown_trigger_uses_goal(self) -> None:
        """Unknown trigger should use goal description in follow-up."""
        state = ConversationState(
            merchant_id="m_001",
            last_trigger_kind="custom_event_xyz",
            goal=ConversationGoal(description="Help merchant optimize their listing"),
        )
        body = _generate_follow_up(state)
        assert "help merchant optimize" in body.lower() or "listing" in body.lower()

    def test_unknown_trigger_generic_fallback(self) -> None:
        """Without a goal, unknown trigger uses trigger name."""
        state = ConversationState(
            merchant_id="m_001",
            last_trigger_kind="custom_event",
        )
        body = _generate_follow_up(state)
        assert "custom event" in body.lower()


# ──────────────────────────────────────────────────────────────────
# Enhanced Autoresponder Detection Tests
# ──────────────────────────────────────────────────────────────────

class TestEnhancedAutoresponder:
    """Test expanded autoresponder patterns."""

    @pytest.mark.parametrize("text", [
        "This is an automated response",
        "We've received your message",
        "Out of office",
        "auto-reply: currently unavailable",
        "This number is unattended",
        "We'll get back to you shortly",
        "Our office is closed",
        "Vacation responder",
        "I am on leave",
        "I am on holiday",
        "Business hours are over",
        "Away from the office",
        "Do not reply to this message",
    ])
    def test_autoresponder_detected(self, text: str) -> None:
        result = detect_intent_with_confidence(text)
        assert result.intent == Intent.AUTORESPONDER
        assert result.confidence >= 0.75
