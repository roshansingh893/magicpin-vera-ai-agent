"""Response models — Pydantic schemas for outgoing API payloads.

Separating response models from request models keeps the interface
contract explicit. The ``ComposedMessage`` model mirrors the challenge's
required output format.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """GET /v1/healthz"""
    status: str = "ok"


class MetadataResponse(BaseModel):
    """GET /v1/metadata — bot identification."""
    name: str
    version: str
    description: str = ""


class ComposedMessage(BaseModel):
    """The structured output of the message composer.

    Mirrors the challenge deliverable schema:
    ``{body, cta, send_as, suppression_key, rationale}``
    """
    body: str = Field(
        ...,
        description="The WhatsApp message body to send.",
    )
    cta: Literal["binary_yes_stop", "open_ended", "none"] = Field(
        ...,
        description="Call-to-action type: binary (YES/STOP), open-ended, or none.",
    )
    send_as: Literal["vera", "merchant_on_behalf"] = Field(
        ...,
        description="Whether the message comes from Vera or on behalf of the merchant.",
    )
    suppression_key: str = Field(
        ...,
        description="Deduplication key to prevent re-sending this message.",
    )
    rationale: str = Field(
        ...,
        description="Short explanation of why this message was chosen.",
    )


class ComposeResponse(BaseModel):
    """POST /v1/context — wrapper for the composed message.

    In Phase 1 this returns a placeholder. Phase 2 will populate
    the ``result`` field with the actual ``ComposedMessage``.
    """
    message: str = "Context received."
    result: Optional[ComposedMessage] = None


class ReplyResponse(BaseModel):
    """POST /v1/reply — multi-turn reply output."""
    message: str = "Reply endpoint."
    conversation_id: str = ""
    intent: str = ""
    stage: str = ""
    result: Optional[ComposedMessage] = None


class TickAction(BaseModel):
    """A single action decided by the tick handler."""
    conversation_id: str
    merchant_id: str
    action: str  # "send_follow_up" or "no_action"
    message: Optional[ComposedMessage] = None


class TickResponse(BaseModel):
    """POST /v1/tick — scheduled cadence output."""
    message: str = "Tick endpoint."
    actions: list[TickAction] = Field(default_factory=list)
    results: list[ComposedMessage] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error response body."""
    error: str
    detail: Optional[list | str] = None
