"""Request models — Pydantic schemas for incoming API payloads.

These models mirror the 4-context framework defined in the magicpin
challenge brief (CategoryContext, MerchantContext, TriggerContext,
CustomerContext). Fields are typed from the actual dataset seed files,
with ``extra="allow"`` on leaf models so new fields in the generated
dataset don't break validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────
# Category Context
# ──────────────────────────────────────────────────────────────────

class VoiceProfile(BaseModel):
    """Tone and vocabulary rules for a business category."""
    tone: str = ""
    voice_register: str = Field(default="", alias="register")
    code_mix: str = ""
    vocab_allowed: list[str] = Field(default_factory=list)
    vocab_taboo: list[str] = Field(default_factory=list)
    salutation_examples: list[str] = Field(default_factory=list)
    tone_examples: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow", "populate_by_name": True}


class OfferTemplate(BaseModel):
    """A canonical offer for a category (e.g., "Dental Cleaning @ ₹299")."""
    id: str = ""
    title: str
    value: str = ""
    audience: str = ""
    type: str = ""

    model_config = {"extra": "allow"}


class PeerStats(BaseModel):
    """Benchmark statistics for this business vertical."""
    scope: str = ""
    avg_rating: float = 0.0
    avg_review_count: int = 0
    avg_views_30d: int = 0
    avg_calls_30d: int = 0
    avg_directions_30d: int = 0
    avg_ctr: float = 0.0
    avg_photos: int = 0
    avg_post_freq_days: int = 0

    model_config = {"extra": "allow"}


class DigestItem(BaseModel):
    """A weekly research, compliance, or trend item."""
    id: str = ""
    kind: str = ""
    title: str = ""
    source: str = ""
    summary: str = ""
    actionable: str = ""

    model_config = {"extra": "allow"}


class ContentItem(BaseModel):
    """Patient/customer education content the merchant can reshare."""
    id: str = ""
    title: str = ""
    channel: str = ""
    body: str = ""

    model_config = {"extra": "allow"}


class SeasonalBeat(BaseModel):
    """Seasonal engagement pattern for this category."""
    month_range: str = ""
    note: str = ""

    model_config = {"extra": "allow"}


class TrendSignal(BaseModel):
    """Search trend movement relevant to this category."""
    query: str = ""
    delta_yoy: float = 0.0

    model_config = {"extra": "allow"}


class CategoryContext(BaseModel):
    """Slow-changing knowledge about a business vertical.

    Shared across all merchants in the same category.
    """
    slug: str
    display_name: str = ""
    voice: VoiceProfile = Field(default_factory=VoiceProfile)
    offer_catalog: list[OfferTemplate] = Field(default_factory=list)
    peer_stats: PeerStats = Field(default_factory=PeerStats)
    digest: list[DigestItem] = Field(default_factory=list)
    patient_content_library: list[ContentItem] = Field(default_factory=list)
    seasonal_beats: list[SeasonalBeat] = Field(default_factory=list)
    trend_signals: list[TrendSignal] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# ──────────────────────────────────────────────────────────────────
# Merchant Context
# ──────────────────────────────────────────────────────────────────

class MerchantIdentity(BaseModel):
    """Basic business identity and location."""
    name: str
    city: str = ""
    locality: str = ""
    place_id: str = ""
    verified: bool = False
    languages: list[str] = Field(default_factory=list)
    owner_first_name: str = ""
    established_year: int | None = None

    model_config = {"extra": "allow"}


class Subscription(BaseModel):
    """Merchant's current plan status."""
    status: str  # "active", "expired", "trial"
    plan: str = ""
    days_remaining: int | None = None
    days_since_expiry: int | None = None
    renewed_at: str | None = None

    model_config = {"extra": "allow"}


class PerformanceDelta(BaseModel):
    """Week-over-week performance deltas."""
    views_pct: float | None = None
    calls_pct: float | None = None
    ctr_pct: float | None = None

    model_config = {"extra": "allow"}


class PerformanceSnapshot(BaseModel):
    """30-day performance metrics with 7-day deltas."""
    window_days: int = 30
    views: int = 0
    calls: int = 0
    directions: int = 0
    ctr: float = 0.0
    leads: int = 0
    delta_7d: PerformanceDelta = Field(default_factory=PerformanceDelta)

    model_config = {"extra": "allow"}


class MerchantOffer(BaseModel):
    """Active or historical offer from the merchant's catalog."""
    id: str = ""
    title: str
    status: str = ""  # "active", "expired", "paused"
    started: str | None = None
    ended: str | None = None

    model_config = {"extra": "allow"}


class ConversationTurn(BaseModel):
    """A single turn in the Vera ↔ merchant conversation history."""
    ts: str  # ISO timestamp
    body: str

    # "vera" or "merchant"
    source: str = Field(default="", alias="from")
    engagement: str = ""

    model_config = {"extra": "allow", "populate_by_name": True}


class ReviewTheme(BaseModel):
    """Emerged theme from recent customer reviews."""
    theme: str = ""
    sentiment: str = ""
    occurrences_30d: int = 0
    common_quote: str = ""

    model_config = {"extra": "allow"}


class MerchantContext(BaseModel):
    """A specific merchant's current state — refreshed daily."""
    merchant_id: str
    category_slug: str = ""
    identity: MerchantIdentity
    subscription: Subscription
    performance: PerformanceSnapshot = Field(default_factory=PerformanceSnapshot)
    offers: list[MerchantOffer] = Field(default_factory=list)
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    customer_aggregate: dict[str, Any] = Field(default_factory=dict)
    signals: list[str] = Field(default_factory=list)
    review_themes: list[ReviewTheme] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# ──────────────────────────────────────────────────────────────────
# Trigger Context
# ──────────────────────────────────────────────────────────────────

class TriggerContext(BaseModel):
    """The event that prompts this message right now."""
    id: str
    scope: Literal["merchant", "customer"]
    kind: str
    source: Literal["external", "internal"]
    merchant_id: str
    customer_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    urgency: int = Field(default=1, ge=1, le=5)
    suppression_key: str = ""
    expires_at: str = ""  # ISO datetime

    model_config = {"extra": "allow"}


# ──────────────────────────────────────────────────────────────────
# Customer Context (optional — only for customer-facing messages)
# ──────────────────────────────────────────────────────────────────

class CustomerIdentity(BaseModel):
    """Customer name, language preference, and demographics."""
    name: str
    phone_redacted: str | None = None
    language_pref: str = ""
    age_band: str = ""

    model_config = {"extra": "allow"}


class CustomerRelationship(BaseModel):
    """Visit history and lifetime value."""
    first_visit: str = ""
    last_visit: str = ""
    visits_total: int = 0
    services_received: list[str] = Field(default_factory=list)
    lifetime_value: float = 0.0

    model_config = {"extra": "allow"}


class Consent(BaseModel):
    """When and how the customer opted into outreach."""
    opted_in_at: str | None = None
    scope: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class CustomerContext(BaseModel):
    """A customer of a specific merchant — for customer-facing messages."""
    customer_id: str
    merchant_id: str
    identity: CustomerIdentity
    relationship: CustomerRelationship = Field(default_factory=CustomerRelationship)
    state: str = ""  # "new", "active", "lapsed_soft", "lapsed_hard", "churned"
    preferences: dict[str, Any] = Field(default_factory=dict)
    consent: Consent = Field(default_factory=Consent)

    model_config = {"extra": "allow"}


# ──────────────────────────────────────────────────────────────────
# API Request Bodies
# ──────────────────────────────────────────────────────────────────

class ComposeRequest(BaseModel):
    """POST /v1/context — compose a message from structured context.

    Mirrors the challenge's ``compose(category, merchant, trigger, customer?)``
    function signature.
    """
    category: CategoryContext
    merchant: MerchantContext
    trigger: TriggerContext
    customer: Optional[CustomerContext] = None


class ReplyRequest(BaseModel):
    """POST /v1/reply — handle a merchant's reply in a multi-turn conversation."""
    conversation_id: str
    merchant_id: str
    merchant_message: str

    model_config = {"extra": "allow"}


class TickRequest(BaseModel):
    """POST /v1/tick — scheduled cadence check.

    Evaluates active conversations and determines which ones
    need follow-up messages.
    """
    timestamp: str = ""  # ISO datetime of tick
    merchant_ids: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}
