"""Phase 2 tests — prompt generation, output validation, and composer.

All tests use mocked Groq responses — NO real API calls.
Run with: pytest tests/ -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models.requests import (
    CategoryContext,
    ComposeRequest,
    CustomerContext,
    CustomerIdentity,
    CustomerRelationship,
    Consent,
    MerchantContext,
    MerchantIdentity,
    PerformanceSnapshot,
    Subscription,
    TriggerContext,
    VoiceProfile,
    PeerStats,
)
from app.models.responses import ComposedMessage
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.prompts.merchant_prompt import build_merchant_prompt
from app.prompts.customer_prompt import build_customer_prompt
from app.services.prompt_builder import build_prompts
from app.services.output_validator import (
    OutputValidationError,
    parse_llm_response,
    validate_composed_message,
    _clean_json_response,
)
from app.services.composer import compose, set_llm_client


# ──────────────────────────────────────────────────────────────────
# Fixtures — reusable test contexts
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_category() -> CategoryContext:
    """Minimal dentist category context for testing."""
    return CategoryContext(
        slug="dentists",
        display_name="Dentists",
        voice=VoiceProfile(
            tone="peer_clinical",
            voice_register="respectful_collegial",
            code_mix="hindi_english_natural",
            vocab_allowed=["fluoride varnish", "scaling", "caries"],
            vocab_taboo=["guaranteed", "100% safe", "miracle"],
            tone_examples=["Worth a look — JIDA Oct 2026 p.14"],
        ),
        peer_stats=PeerStats(
            avg_rating=4.4,
            avg_review_count=62,
            avg_views_30d=1820,
            avg_calls_30d=12,
            avg_ctr=0.030,
        ),
    )


@pytest.fixture
def sample_merchant() -> MerchantContext:
    """Minimal merchant context for Dr. Meera."""
    return MerchantContext(
        merchant_id="m_001_drmeera_dentist_delhi",
        category_slug="dentists",
        identity=MerchantIdentity(
            name="Dr. Meera's Dental Clinic",
            city="Delhi",
            locality="Lajpat Nagar",
            verified=True,
            languages=["en", "hi"],
            owner_first_name="Meera",
        ),
        subscription=Subscription(
            status="active",
            plan="Pro",
            days_remaining=82,
        ),
        performance=PerformanceSnapshot(
            views=2410,
            calls=18,
            directions=45,
            ctr=0.021,
        ),
        signals=["stale_posts:22d", "ctr_below_peer_median"],
    )


@pytest.fixture
def sample_trigger() -> TriggerContext:
    """Minimal trigger context for a research digest."""
    return TriggerContext(
        id="trg_001_research_digest_dentists",
        scope="merchant",
        kind="research_digest",
        source="external",
        merchant_id="m_001_drmeera_dentist_delhi",
        urgency=2,
        suppression_key="research:dentists:2026-W17",
    )


@pytest.fixture
def sample_customer() -> CustomerContext:
    """Minimal customer context for Priya."""
    return CustomerContext(
        customer_id="c_001_priya_for_m001",
        merchant_id="m_001_drmeera_dentist_delhi",
        identity=CustomerIdentity(
            name="Priya",
            language_pref="hi-en mix",
        ),
        relationship=CustomerRelationship(
            first_visit="2025-11-04",
            last_visit="2026-05-12",
            visits_total=4,
            services_received=["cleaning", "cleaning", "whitening", "cleaning"],
        ),
        state="lapsed_soft",
        consent=Consent(
            opted_in_at="2025-11-04",
            scope=["recall_reminders", "appointment_reminders"],
        ),
    )


VALID_LLM_RESPONSE = json.dumps({
    "body": "Dr. Meera, JIDA's Oct issue has a finding relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look?",
    "cta": "open_ended",
    "send_as": "vera",
    "suppression_key": "research:dentists:2026-W17",
    "rationale": "External research digest with merchant-relevant clinical anchor. Source citation maintains credibility. Open-ended CTA invites continuation.",
})


VALID_CUSTOMER_LLM_RESPONSE = json.dumps({
    "body": "Hi Priya, Dr. Meera's clinic here 🦷 It's been a while since your last visit — your cleaning recall is due. Kya aap is week available hain?",
    "cta": "open_ended",
    "send_as": "merchant_on_behalf",
    "suppression_key": "recall:c_001_priya_for_m001:6mo",
    "rationale": "Customer-scoped recall. Hi-en mix language honored. Friendly tone, not promotional.",
})


# ──────────────────────────────────────────────────────────────────
# System Prompt Tests
# ──────────────────────────────────────────────────────────────────

class TestSystemPrompt:
    """Verify the system prompt contains critical elements."""

    def test_contains_vera_identity(self) -> None:
        assert "Vera" in SYSTEM_PROMPT

    def test_contains_json_format_instruction(self) -> None:
        assert '"body"' in SYSTEM_PROMPT
        assert '"cta"' in SYSTEM_PROMPT
        assert '"send_as"' in SYSTEM_PROMPT
        assert '"suppression_key"' in SYSTEM_PROMPT
        assert '"rationale"' in SYSTEM_PROMPT

    def test_no_hallucination_rule(self) -> None:
        assert "Never invent" in SYSTEM_PROMPT or "never invent" in SYSTEM_PROMPT.lower()

    def test_no_url_rule(self) -> None:
        assert "URL" in SYSTEM_PROMPT

    def test_contains_output_format(self) -> None:
        assert "valid JSON" in SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────
# Merchant Prompt Tests
# ──────────────────────────────────────────────────────────────────

class TestMerchantPrompt:
    """Verify merchant prompt construction."""

    def test_includes_merchant_name(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        prompt = build_merchant_prompt(sample_category, sample_merchant, sample_trigger)
        assert "Dr. Meera" in prompt

    def test_includes_trigger_kind(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        prompt = build_merchant_prompt(sample_category, sample_merchant, sample_trigger)
        assert "research_digest" in prompt

    def test_includes_performance_metrics(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        prompt = build_merchant_prompt(sample_category, sample_merchant, sample_trigger)
        assert "2410" in prompt  # views
        assert "0.021" in prompt  # CTR

    def test_includes_signals(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        prompt = build_merchant_prompt(sample_category, sample_merchant, sample_trigger)
        assert "stale_posts" in prompt

    def test_includes_voice_profile(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        prompt = build_merchant_prompt(sample_category, sample_merchant, sample_trigger)
        assert "peer_clinical" in prompt

    def test_includes_taboo_words(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        prompt = build_merchant_prompt(sample_category, sample_merchant, sample_trigger)
        assert "guaranteed" in prompt

    def test_includes_task_instruction(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        prompt = build_merchant_prompt(sample_category, sample_merchant, sample_trigger)
        assert "TASK" in prompt
        assert "send_as" in prompt

    def test_includes_location(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        prompt = build_merchant_prompt(sample_category, sample_merchant, sample_trigger)
        assert "Lajpat Nagar" in prompt
        assert "Delhi" in prompt


# ──────────────────────────────────────────────────────────────────
# Customer Prompt Tests
# ──────────────────────────────────────────────────────────────────

class TestCustomerPrompt:
    """Verify customer prompt construction."""

    def test_includes_customer_name(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext,
        sample_trigger: TriggerContext, sample_customer: CustomerContext
    ) -> None:
        prompt = build_customer_prompt(sample_category, sample_merchant, sample_trigger, sample_customer)
        assert "Priya" in prompt

    def test_includes_merchant_name(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext,
        sample_trigger: TriggerContext, sample_customer: CustomerContext
    ) -> None:
        prompt = build_customer_prompt(sample_category, sample_merchant, sample_trigger, sample_customer)
        assert "Dr. Meera" in prompt

    def test_includes_language_pref(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext,
        sample_trigger: TriggerContext, sample_customer: CustomerContext
    ) -> None:
        prompt = build_customer_prompt(sample_category, sample_merchant, sample_trigger, sample_customer)
        assert "hi-en mix" in prompt

    def test_includes_visit_history(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext,
        sample_trigger: TriggerContext, sample_customer: CustomerContext
    ) -> None:
        prompt = build_customer_prompt(sample_category, sample_merchant, sample_trigger, sample_customer)
        assert "4" in prompt  # visits_total

    def test_includes_merchant_on_behalf(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext,
        sample_trigger: TriggerContext, sample_customer: CustomerContext
    ) -> None:
        prompt = build_customer_prompt(sample_category, sample_merchant, sample_trigger, sample_customer)
        assert "merchant_on_behalf" in prompt

    def test_includes_consent_info(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext,
        sample_trigger: TriggerContext, sample_customer: CustomerContext
    ) -> None:
        prompt = build_customer_prompt(sample_category, sample_merchant, sample_trigger, sample_customer)
        assert "recall_reminders" in prompt


# ──────────────────────────────────────────────────────────────────
# Prompt Builder Tests
# ──────────────────────────────────────────────────────────────────

class TestPromptBuilder:
    """Verify prompt builder routing."""

    def test_merchant_flow_no_customer(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        system, user = build_prompts(sample_category, sample_merchant, sample_trigger, customer=None)
        assert system == SYSTEM_PROMPT
        assert 'send_as = "vera"' in user

    def test_customer_flow_with_customer(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext,
        sample_trigger: TriggerContext, sample_customer: CustomerContext
    ) -> None:
        system, user = build_prompts(sample_category, sample_merchant, sample_trigger, customer=sample_customer)
        assert system == SYSTEM_PROMPT
        assert 'send_as = "merchant_on_behalf"' in user


# ──────────────────────────────────────────────────────────────────
# JSON Parsing & Cleaning Tests
# ──────────────────────────────────────────────────────────────────

class TestJSONCleaning:
    """Test JSON response cleaning."""

    def test_clean_plain_json(self) -> None:
        raw = '{"body": "hello"}'
        assert _clean_json_response(raw) == '{"body": "hello"}'

    def test_clean_json_with_code_fence(self) -> None:
        raw = '```json\n{"body": "hello"}\n```'
        assert _clean_json_response(raw) == '{"body": "hello"}'

    def test_clean_json_with_plain_fence(self) -> None:
        raw = '```\n{"body": "hello"}\n```'
        assert _clean_json_response(raw) == '{"body": "hello"}'

    def test_clean_json_with_whitespace(self) -> None:
        raw = '  \n{"body": "hello"}\n  '
        assert _clean_json_response(raw) == '{"body": "hello"}'


class TestJSONParsing:
    """Test JSON parsing from LLM output."""

    def test_parse_valid_json(self) -> None:
        data = parse_llm_response('{"body": "test"}')
        assert data == {"body": "test"}

    def test_parse_empty_raises(self) -> None:
        with pytest.raises(OutputValidationError, match="empty"):
            parse_llm_response("")

    def test_parse_invalid_json_raises(self) -> None:
        with pytest.raises(OutputValidationError, match="not valid JSON"):
            parse_llm_response("this is not json")

    def test_parse_non_object_raises(self) -> None:
        with pytest.raises(OutputValidationError, match="JSON object"):
            parse_llm_response("[1, 2, 3]")

    def test_parse_json_with_code_fence(self) -> None:
        raw = '```json\n{"body": "hello"}\n```'
        data = parse_llm_response(raw)
        assert data == {"body": "hello"}


# ──────────────────────────────────────────────────────────────────
# Output Validator Tests
# ──────────────────────────────────────────────────────────────────

class TestOutputValidator:
    """Test the composed message validator."""

    def test_valid_message_passes(self) -> None:
        data = {
            "body": "Hi Dr. Meera, here's a quick update for you.",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": "research:dentists:2026-W17",
            "rationale": "Research digest with clinical anchor.",
        }
        result = validate_composed_message(data)
        assert isinstance(result, ComposedMessage)
        assert result.body == data["body"]
        assert result.cta == "open_ended"

    def test_missing_field_raises(self) -> None:
        data = {
            "body": "test message body here",
            "cta": "open_ended",
            # missing send_as, suppression_key, rationale
        }
        with pytest.raises(OutputValidationError, match="Missing required"):
            validate_composed_message(data)

    def test_empty_body_raises(self) -> None:
        data = {
            "body": "",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": "key",
            "rationale": "reason",
        }
        with pytest.raises(OutputValidationError, match="empty"):
            validate_composed_message(data)

    def test_short_body_raises(self) -> None:
        data = {
            "body": "Hi",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": "key",
            "rationale": "reason",
        }
        with pytest.raises(OutputValidationError, match="too short"):
            validate_composed_message(data)

    def test_invalid_cta_raises(self) -> None:
        data = {
            "body": "Hi Dr. Meera, here's a quick update for you.",
            "cta": "invalid_value",
            "send_as": "vera",
            "suppression_key": "key",
            "rationale": "reason",
        }
        with pytest.raises(OutputValidationError, match="Invalid 'cta'"):
            validate_composed_message(data)

    def test_invalid_send_as_raises(self) -> None:
        data = {
            "body": "Hi Dr. Meera, here's a quick update for you.",
            "cta": "open_ended",
            "send_as": "invalid_sender",
            "suppression_key": "key",
            "rationale": "reason",
        }
        with pytest.raises(OutputValidationError, match="Invalid 'send_as'"):
            validate_composed_message(data)

    def test_empty_suppression_key_raises(self) -> None:
        data = {
            "body": "Hi Dr. Meera, here's a quick update for you.",
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": "   ",
            "rationale": "reason",
        }
        with pytest.raises(OutputValidationError, match="suppression_key"):
            validate_composed_message(data)

    def test_non_string_field_raises(self) -> None:
        data = {
            "body": 12345,
            "cta": "open_ended",
            "send_as": "vera",
            "suppression_key": "key",
            "rationale": "reason",
        }
        with pytest.raises(OutputValidationError, match="must be a string"):
            validate_composed_message(data)

    def test_cta_case_insensitive(self) -> None:
        """CTA values should be normalised to lowercase."""
        data = {
            "body": "Hi Dr. Meera, here's a quick update for you.",
            "cta": "Open_Ended",
            "send_as": "vera",
            "suppression_key": "key",
            "rationale": "reason",
        }
        result = validate_composed_message(data)
        assert result.cta == "open_ended"


# ──────────────────────────────────────────────────────────────────
# Composer Tests (mocked Groq)
# ──────────────────────────────────────────────────────────────────

class TestComposer:
    """Test the compose() orchestrator with mocked LLM responses."""

    @pytest.mark.asyncio
    async def test_merchant_flow_success(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        """compose() returns a ComposedMessage when the LLM returns valid JSON."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=VALID_LLM_RESPONSE)

        set_llm_client(mock_client)
        try:
            result = await compose(sample_category, sample_merchant, sample_trigger)
            assert isinstance(result, ComposedMessage)
            assert result.send_as == "vera"
            assert result.cta == "open_ended"
            assert len(result.body) > 0
            mock_client.generate.assert_called_once()
        finally:
            set_llm_client(None)

    @pytest.mark.asyncio
    async def test_customer_flow_success(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext,
        sample_trigger: TriggerContext, sample_customer: CustomerContext
    ) -> None:
        """compose() handles customer flow correctly."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=VALID_CUSTOMER_LLM_RESPONSE)

        set_llm_client(mock_client)
        try:
            result = await compose(sample_category, sample_merchant, sample_trigger, customer=sample_customer)
            assert isinstance(result, ComposedMessage)
            assert result.send_as == "merchant_on_behalf"
            assert "Priya" in result.body
        finally:
            set_llm_client(None)

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        """compose() retries when the first LLM call returns invalid JSON."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(
            side_effect=["this is not json", VALID_LLM_RESPONSE]
        )

        set_llm_client(mock_client)
        try:
            result = await compose(sample_category, sample_merchant, sample_trigger)
            assert isinstance(result, ComposedMessage)
            assert mock_client.generate.call_count == 2
        finally:
            set_llm_client(None)

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        """compose() raises OutputValidationError after all retries fail."""
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value="not json at all")

        set_llm_client(mock_client)
        try:
            with pytest.raises(OutputValidationError):
                await compose(sample_category, sample_merchant, sample_trigger)
            assert mock_client.generate.call_count == 2  # MAX_RETRIES = 2
        finally:
            set_llm_client(None)

    @pytest.mark.asyncio
    async def test_handles_code_fenced_json(
        self, sample_category: CategoryContext, sample_merchant: MerchantContext, sample_trigger: TriggerContext
    ) -> None:
        """compose() handles LLM responses wrapped in code fences."""
        fenced = f"```json\n{VALID_LLM_RESPONSE}\n```"
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=fenced)

        set_llm_client(mock_client)
        try:
            result = await compose(sample_category, sample_merchant, sample_trigger)
            assert isinstance(result, ComposedMessage)
        finally:
            set_llm_client(None)
