"""Phase 3 tests — conversation engine, intent detection, state machine.

All tests use mocked Groq responses — NO real API calls.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.models.conversation import (
    ConversationStage,
    ConversationState,
    Intent,
    Message,
    MessageRole,
)
from app.models.responses import ComposedMessage
from app.services.conversation_manager import (
    ConversationManager,
    ConversationNotFoundError,
)
from app.services.intent_detector import detect_intent, is_deterministic
from app.services.state_machine import (
    InvalidTransitionError,
    transition,
)
from app.services.tick_handler import _needs_follow_up, process_tick
from app.services.reply_handler import handle_merchant_reply


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_store():
    """Clear the conversation store before and after each test."""
    ConversationManager.clear_store()
    yield
    ConversationManager.clear_store()


VALID_LLM_REPLY = json.dumps({
    "body": "Great question! Based on your clinic data, I'd recommend refreshing your treatment photos first.",
    "cta": "open_ended",
    "send_as": "vera",
    "suppression_key": "reply:question:m001",
    "rationale": "Merchant asked a question, responding with actionable advice.",
})


# ──────────────────────────────────────────────────────────────────
# State Machine Tests
# ──────────────────────────────────────────────────────────────────

class TestStateMachine:
    """Test explicit state transitions."""

    def test_new_to_message_sent(self) -> None:
        state = ConversationState(merchant_id="m_001")
        assert state.stage == ConversationStage.NEW
        transition(state, "message_sent")
        assert state.stage == ConversationStage.MESSAGE_SENT

    def test_message_sent_to_waiting(self) -> None:
        state = ConversationState(merchant_id="m_001", stage=ConversationStage.MESSAGE_SENT)
        transition(state, "wait")
        assert state.stage == ConversationStage.WAITING_REPLY

    def test_waiting_to_resolved_on_reply(self) -> None:
        state = ConversationState(merchant_id="m_001", stage=ConversationStage.WAITING_REPLY)
        transition(state, "reply_received")
        assert state.stage == ConversationStage.RESOLVED

    def test_waiting_to_follow_up_sent(self) -> None:
        state = ConversationState(merchant_id="m_001", stage=ConversationStage.WAITING_REPLY)
        transition(state, "follow_up_sent")
        assert state.stage == ConversationStage.FOLLOW_UP_SENT

    def test_follow_up_to_resolved(self) -> None:
        state = ConversationState(merchant_id="m_001", stage=ConversationStage.FOLLOW_UP_SENT)
        transition(state, "reply_received")
        assert state.stage == ConversationStage.RESOLVED

    def test_resolved_to_closed(self) -> None:
        state = ConversationState(merchant_id="m_001", stage=ConversationStage.RESOLVED)
        transition(state, "close")
        assert state.stage == ConversationStage.CLOSED

    def test_invalid_transition_raises(self) -> None:
        state = ConversationState(merchant_id="m_001", stage=ConversationStage.CLOSED)
        with pytest.raises(InvalidTransitionError):
            transition(state, "message_sent")

    def test_transition_updates_timestamp(self) -> None:
        state = ConversationState(merchant_id="m_001")
        old_ts = state.updated_at
        transition(state, "message_sent")
        assert state.updated_at >= old_ts


# ──────────────────────────────────────────────────────────────────
# Conversation Manager Tests
# ──────────────────────────────────────────────────────────────────

class TestConversationManager:
    """Test conversation CRUD operations."""

    def test_create_conversation(self) -> None:
        state = ConversationManager.create(
            merchant_id="m_001",
            trigger_id="trg_001",
            trigger_kind="research_digest",
            bot_message="Hello merchant!",
        )
        assert state.merchant_id == "m_001"
        assert state.stage == ConversationStage.WAITING_REPLY
        assert state.last_trigger_kind == "research_digest"
        assert len(state.history) == 1
        assert state.history[0].role == MessageRole.VERA

    def test_get_conversation(self) -> None:
        created = ConversationManager.create(
            merchant_id="m_001",
            trigger_id="trg_001",
            trigger_kind="test",
            bot_message="Hello!",
        )
        retrieved = ConversationManager.get(created.conversation_id)
        assert retrieved.conversation_id == created.conversation_id

    def test_get_nonexistent_raises(self) -> None:
        with pytest.raises(ConversationNotFoundError):
            ConversationManager.get("nonexistent_id")

    def test_find_by_merchant(self) -> None:
        ConversationManager.create("m_001", "t1", "test", "Hello 1")
        ConversationManager.create("m_001", "t2", "test", "Hello 2")
        ConversationManager.create("m_002", "t3", "test", "Hello 3")

        results = ConversationManager.find_by_merchant("m_001")
        assert len(results) == 2

    def test_append_merchant_reply(self) -> None:
        created = ConversationManager.create("m_001", "t1", "test", "Hello!")
        state = ConversationManager.append_merchant_reply(
            created.conversation_id, "Yes!", "affirmative"
        )
        assert len(state.history) == 2
        assert state.history[1].role == MessageRole.MERCHANT
        assert state.history[1].body == "Yes!"
        assert state.history[1].intent == Intent.AFFIRMATIVE
        assert state.stage == ConversationStage.RESOLVED

    def test_append_bot_message(self) -> None:
        created = ConversationManager.create("m_001", "t1", "test", "Hello!")
        # Transition to RESOLVED first (simulate merchant reply)
        ConversationManager.append_merchant_reply(created.conversation_id, "Yes!", "affirmative")
        # Now bot can reply (RESOLVED → MESSAGE_SENT → WAITING_REPLY)
        state = ConversationManager.append_bot_message(
            created.conversation_id, "Great!"
        )
        assert len(state.history) == 3
        assert state.last_bot_message == "Great!"
        assert state.stage == ConversationStage.WAITING_REPLY

    def test_close_conversation(self) -> None:
        created = ConversationManager.create("m_001", "t1", "test", "Hello!")
        state = ConversationManager.close(created.conversation_id, reason="test")
        assert state.stage == ConversationStage.CLOSED
        assert state.metadata.get("close_reason") == "test"

    def test_store_size(self) -> None:
        assert ConversationManager.store_size() == 0
        ConversationManager.create("m_001", "t1", "test", "Hello!")
        assert ConversationManager.store_size() == 1

    def test_find_active_only(self) -> None:
        c1 = ConversationManager.create("m_001", "t1", "test", "Hello 1")
        ConversationManager.create("m_001", "t2", "test", "Hello 2")
        ConversationManager.close(c1.conversation_id, reason="done")

        active = ConversationManager.find_by_merchant("m_001", active_only=True)
        all_convos = ConversationManager.find_by_merchant("m_001", active_only=False)
        assert len(active) == 1
        assert len(all_convos) == 2


# ──────────────────────────────────────────────────────────────────
# Intent Detection Tests
# ──────────────────────────────────────────────────────────────────

class TestIntentDetection:
    """Test deterministic intent classification."""

    # Affirmative
    @pytest.mark.parametrize("text", ["yes", "Yes", "YES", "ok", "sure", "haan", "ji", "bilkul", "👍", "sounds good"])
    def test_affirmative(self, text: str) -> None:
        assert detect_intent(text) == Intent.AFFIRMATIVE

    # Negative
    @pytest.mark.parametrize("text", ["no", "No", "nope", "nahi", "not interested", "no thanks", "pass"])
    def test_negative(self, text: str) -> None:
        assert detect_intent(text) == Intent.NEGATIVE

    # Unsubscribe
    @pytest.mark.parametrize("text", ["stop", "STOP", "unsubscribe", "opt out", "don't send", "remove me"])
    def test_unsubscribe(self, text: str) -> None:
        assert detect_intent(text) == Intent.UNSUBSCRIBE

    # Thanks
    @pytest.mark.parametrize("text", ["thanks", "Thank you", "dhanyavaad", "shukriya", "thx"])
    def test_thanks(self, text: str) -> None:
        assert detect_intent(text) == Intent.THANKS

    # Greeting
    @pytest.mark.parametrize("text", ["hi", "Hello", "hey", "namaste", "good morning"])
    def test_greeting(self, text: str) -> None:
        assert detect_intent(text) == Intent.GREETING

    # Autoresponder
    @pytest.mark.parametrize("text", [
        "This is an automated response",
        "We've received your message",
        "Out of office",
        "auto-reply: currently unavailable",
        "This number is unattended",
    ])
    def test_autoresponder(self, text: str) -> None:
        assert detect_intent(text) == Intent.AUTORESPONDER

    # Question
    @pytest.mark.parametrize("text", ["what is this about?", "how does it work?", "can you explain?"])
    def test_question(self, text: str) -> None:
        assert detect_intent(text) == Intent.QUESTION

    # Unknown
    def test_unknown(self) -> None:
        assert detect_intent("I need to think about the strategy for next quarter") == Intent.UNKNOWN

    def test_empty_text(self) -> None:
        assert detect_intent("") == Intent.UNKNOWN

    # is_deterministic helper
    def test_affirmative_is_deterministic(self) -> None:
        assert is_deterministic(Intent.AFFIRMATIVE) is True

    def test_question_is_not_deterministic(self) -> None:
        assert is_deterministic(Intent.QUESTION) is False

    def test_unknown_is_not_deterministic(self) -> None:
        assert is_deterministic(Intent.UNKNOWN) is False


# ──────────────────────────────────────────────────────────────────
# Reply Handler Tests
# ──────────────────────────────────────────────────────────────────

class TestReplyHandler:
    """Test the reply orchestrator with mocked LLM."""

    @pytest.mark.asyncio
    async def test_affirmative_reply(self) -> None:
        """Affirmative reply is handled deterministically."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        reply, intent, cid = await handle_merchant_reply(conv.conversation_id, "yes")

        assert intent == Intent.AFFIRMATIVE
        assert reply is not None
        assert "suggestion" in reply.body.lower() or "ready" in reply.body.lower()

    @pytest.mark.asyncio
    async def test_negative_reply(self) -> None:
        """Negative reply is handled without LLM."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        reply, intent, cid = await handle_merchant_reply(conv.conversation_id, "no thanks")

        assert intent == Intent.NEGATIVE
        assert reply is not None
        assert "no problem" in reply.body.lower() or "check back" in reply.body.lower()

    @pytest.mark.asyncio
    async def test_unsubscribe_closes_conversation(self) -> None:
        """Unsubscribe closes the conversation."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        reply, intent, cid = await handle_merchant_reply(conv.conversation_id, "stop")

        assert intent == Intent.UNSUBSCRIBE
        state = ConversationManager.get(conv.conversation_id)
        assert state.stage == ConversationStage.CLOSED
        assert state.metadata.get("close_reason") == "merchant_unsubscribed"

    @pytest.mark.asyncio
    async def test_autoresponder_closes_no_reply(self) -> None:
        """Autoresponder closes the conversation with no reply."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        reply, intent, cid = await handle_merchant_reply(
            conv.conversation_id, "This is an automated response"
        )

        assert intent == Intent.AUTORESPONDER
        assert reply is None
        state = ConversationManager.get(conv.conversation_id)
        assert state.stage == ConversationStage.CLOSED
        assert state.metadata.get("close_reason") == "autoresponder_detected"

    @pytest.mark.asyncio
    async def test_question_uses_llm(self) -> None:
        """Questions fall back to the LLM."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=VALID_LLM_REPLY)

        from app.services.composer import set_llm_client
        set_llm_client(mock_client)
        try:
            reply, intent, cid = await handle_merchant_reply(
                conv.conversation_id, "what does CTR mean?"
            )
            assert intent == Intent.QUESTION
            assert reply is not None
            mock_client.generate.assert_called_once()
        finally:
            set_llm_client(None)

    @pytest.mark.asyncio
    async def test_thanks_reply(self) -> None:
        """Thanks is handled deterministically."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        reply, intent, cid = await handle_merchant_reply(conv.conversation_id, "thank you")

        assert intent == Intent.THANKS
        assert reply is not None
        assert "welcome" in reply.body.lower()

    @pytest.mark.asyncio
    async def test_nonexistent_conversation(self) -> None:
        """Raises error for nonexistent conversation."""
        with pytest.raises(ConversationNotFoundError):
            await handle_merchant_reply("nonexistent", "hello")

    @pytest.mark.asyncio
    async def test_conversation_history_updated(self) -> None:
        """Reply updates conversation history with both messages."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        await handle_merchant_reply(conv.conversation_id, "yes")

        state = ConversationManager.get(conv.conversation_id)
        # Initial bot message + merchant reply + bot reply = 3
        assert len(state.history) == 3
        assert state.history[0].role == MessageRole.VERA
        assert state.history[1].role == MessageRole.MERCHANT
        assert state.history[2].role == MessageRole.VERA


# ──────────────────────────────────────────────────────────────────
# Tick Handler Tests
# ──────────────────────────────────────────────────────────────────

class TestTickHandler:
    """Test the follow-up decision engine."""

    def test_needs_follow_up_true(self) -> None:
        """Should follow up if waiting > 24h and no prior follow-ups."""
        state = ConversationState(
            merchant_id="m_001",
            stage=ConversationStage.WAITING_REPLY,
            follow_up_count=0,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        assert _needs_follow_up(state, datetime.now(timezone.utc)) is True

    def test_needs_follow_up_too_soon(self) -> None:
        """Should not follow up if < 24h elapsed."""
        state = ConversationState(
            merchant_id="m_001",
            stage=ConversationStage.WAITING_REPLY,
            follow_up_count=0,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=12),
        )
        assert _needs_follow_up(state, datetime.now(timezone.utc)) is False

    def test_needs_follow_up_already_sent(self) -> None:
        """Should not follow up if max follow-ups reached."""
        state = ConversationState(
            merchant_id="m_001",
            stage=ConversationStage.WAITING_REPLY,
            follow_up_count=1,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        assert _needs_follow_up(state, datetime.now(timezone.utc)) is False

    def test_needs_follow_up_wrong_stage(self) -> None:
        """Should not follow up if not in WAITING_REPLY stage."""
        state = ConversationState(
            merchant_id="m_001",
            stage=ConversationStage.RESOLVED,
            follow_up_count=0,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        assert _needs_follow_up(state, datetime.now(timezone.utc)) is False

    def test_process_tick_no_conversations(self) -> None:
        """Tick with no active conversations returns empty."""
        actions = process_tick()
        assert actions == []

    def test_process_tick_with_follow_up(self) -> None:
        """Tick generates a follow-up for eligible conversations."""
        # Create a conversation and manually age it
        conv = ConversationManager.create("m_001", "t1", "research_digest", "Hello!")
        state = ConversationManager.get(conv.conversation_id)
        state.updated_at = datetime.now(timezone.utc) - timedelta(hours=25)

        actions = process_tick()

        follow_ups = [a for a in actions if a["action"] == "send_follow_up"]
        assert len(follow_ups) == 1
        assert follow_ups[0]["conversation_id"] == conv.conversation_id
        assert follow_ups[0]["message"] is not None

    def test_process_tick_no_action_for_recent(self) -> None:
        """Tick returns no_action for recent conversations."""
        ConversationManager.create("m_001", "t1", "test", "Hello!")

        actions = process_tick()
        no_actions = [a for a in actions if a["action"] == "no_action"]
        assert len(no_actions) == 1

    def test_follow_up_transitions_state(self) -> None:
        """Follow-up transitions conversation to FOLLOW_UP_SENT."""
        conv = ConversationManager.create("m_001", "t1", "test", "Hello!")
        state = ConversationManager.get(conv.conversation_id)
        state.updated_at = datetime.now(timezone.utc) - timedelta(hours=25)

        process_tick()

        updated = ConversationManager.get(conv.conversation_id)
        assert updated.stage == ConversationStage.FOLLOW_UP_SENT
        assert updated.follow_up_count == 1


# ──────────────────────────────────────────────────────────────────
# Endpoint Integration Tests
# ──────────────────────────────────────────────────────────────────

class TestEndpointIntegration:
    """Test /v1/reply and /v1/tick endpoint wiring with mocked compose."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import create_app
        app = create_app()
        return TestClient(app)

    MOCK_COMPOSED = ComposedMessage(
        body="Hello merchant!",
        cta="open_ended",
        send_as="vera",
        suppression_key="test:key",
        rationale="Test rationale.",
    )

    VALID_CONTEXT_PAYLOAD = {
        "category": {"slug": "dentists"},
        "merchant": {
            "merchant_id": "m_001",
            "identity": {"name": "Test Clinic"},
            "subscription": {"status": "active"},
        },
        "trigger": {
            "id": "trg_001",
            "scope": "merchant",
            "kind": "test",
            "source": "external",
            "merchant_id": "m_001",
        },
    }

    @patch("app.api.routes.compose", new_callable=AsyncMock)
    def test_context_creates_conversation(self, mock_compose: AsyncMock, client) -> None:
        """POST /v1/context should create a conversation."""
        mock_compose.return_value = self.MOCK_COMPOSED
        response = client.post("/v1/context", json=self.VALID_CONTEXT_PAYLOAD)
        assert response.status_code == 200
        assert ConversationManager.store_size() >= 1

    @patch("app.api.routes.compose", new_callable=AsyncMock)
    def test_reply_endpoint(self, mock_compose: AsyncMock, client) -> None:
        """POST /v1/reply should process a merchant reply."""
        mock_compose.return_value = self.MOCK_COMPOSED
        # Create conversation via context
        client.post("/v1/context", json=self.VALID_CONTEXT_PAYLOAD)
        convos = ConversationManager.find_by_merchant("m_001")
        conv_id = convos[0].conversation_id

        reply_payload = {
            "conversation_id": conv_id,
            "merchant_id": "m_001",
            "merchant_message": "yes",
        }
        response = client.post("/v1/reply", json=reply_payload)
        data = response.json()
        assert response.status_code == 200
        assert data["intent"] == "affirmative"
        assert data["result"] is not None

    @patch("app.api.routes.compose", new_callable=AsyncMock)
    def test_reply_not_found(self, mock_compose: AsyncMock, client) -> None:
        """POST /v1/reply with invalid conversation_id."""
        reply_payload = {
            "conversation_id": "nonexistent",
            "merchant_id": "m_001",
            "merchant_message": "hello",
        }
        response = client.post("/v1/reply", json=reply_payload)
        data = response.json()
        assert response.status_code == 200
        assert "not found" in data["message"]

    def test_tick_endpoint(self, client) -> None:
        """POST /v1/tick should return actions."""
        response = client.post("/v1/tick", json={"timestamp": "2026-07-07T00:00:00Z"})
        assert response.status_code == 200
        data = response.json()
        assert "actions" in data

    @patch("app.api.routes.compose", new_callable=AsyncMock)
    def test_tick_with_follow_up(self, mock_compose: AsyncMock, client) -> None:
        """POST /v1/tick should detect follow-up opportunities."""
        mock_compose.return_value = self.MOCK_COMPOSED
        # Create a conversation
        client.post("/v1/context", json=self.VALID_CONTEXT_PAYLOAD)
        convos = ConversationManager.find_by_merchant("m_001")
        state = convos[0]
        # Age it
        state.updated_at = datetime.now(timezone.utc) - timedelta(hours=25)

        response = client.post("/v1/tick", json={"timestamp": "2026-07-07T00:00:00Z"})
        data = response.json()
        assert response.status_code == 200
        follow_ups = [a for a in data["actions"] if a["action"] == "send_follow_up"]
        assert len(follow_ups) >= 1
