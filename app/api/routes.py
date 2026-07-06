"""API route definitions — all /v1 endpoints.

Each endpoint:
1. Logs the incoming request.
2. Validates via Pydantic (automatic).
3. Returns a typed response model.
4. Logs the outgoing response.

Business logic lives in app.services, not here.
"""

import logging

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.requests import ComposeRequest, ReplyRequest, TickRequest
from app.models.responses import (
    ComposeResponse,
    HealthResponse,
    MetadataResponse,
    ReplyResponse,
    TickResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])


# ──────────────────────────────────────────────────────────────────
# Health & Metadata
# ──────────────────────────────────────────────────────────────────

@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns a simple status to confirm the service is running.",
)
async def healthz() -> HealthResponse:
    """Liveness probe — always returns ``{"status": "ok"}``."""
    logger.info("GET /v1/healthz — health check requested")
    response = HealthResponse(status="ok")
    logger.info("GET /v1/healthz — responding: %s", response.status)
    return response


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Bot metadata",
    description="Returns identifying information about the Vera AI Agent.",
)
async def metadata() -> MetadataResponse:
    """Return bot name, version, and description from config."""
    logger.info("GET /v1/metadata — metadata requested")
    settings = get_settings()
    response = MetadataResponse(
        name=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
    )
    logger.info("GET /v1/metadata — responding: name=%s version=%s", response.name, response.version)
    return response


# ──────────────────────────────────────────────────────────────────
# Core Composition Endpoints
# ──────────────────────────────────────────────────────────────────

@router.post(
    "/context",
    response_model=ComposeResponse,
    summary="Compose a message from structured context",
    description=(
        "Accepts the 4-context framework (category, merchant, trigger, "
        "optional customer) and returns a composed WhatsApp message. "
        "Phase 1: validates input and returns a placeholder."
    ),
)
async def compose_context(request: ComposeRequest) -> ComposeResponse:
    """Accept and validate the full context payload.

    Phase 2 will pass the validated contexts to the composition
    service and return the generated message.
    """
    logger.info(
        "POST /v1/context — merchant=%s trigger=%s category=%s customer=%s",
        request.merchant.merchant_id,
        request.trigger.id,
        request.category.slug,
        request.customer.customer_id if request.customer else "none",
    )
    logger.info("POST /v1/context — validation successful")

    # Phase 2: result = await composer.compose(...)
    response = ComposeResponse(message="Context received.")
    logger.info("POST /v1/context — responding: %s", response.message)
    return response


@router.post(
    "/reply",
    response_model=ReplyResponse,
    summary="Handle a merchant reply in a multi-turn conversation",
    description=(
        "Accepts a merchant's reply message and conversation state, "
        "returns the next bot response. Phase 1: placeholder."
    ),
)
async def handle_reply(request: ReplyRequest) -> ReplyResponse:
    """Accept a merchant reply and produce the next turn.

    Phase 2 will implement conversation state management,
    auto-reply detection, and intent-handoff routing.
    """
    logger.info(
        "POST /v1/reply — merchant=%s message_length=%d",
        request.merchant_id,
        len(request.merchant_message),
    )
    logger.info("POST /v1/reply — validation successful")

    response = ReplyResponse(message="Reply endpoint.")
    logger.info("POST /v1/reply — responding: %s", response.message)
    return response


@router.post(
    "/tick",
    response_model=TickResponse,
    summary="Scheduled cadence tick",
    description=(
        "Called periodically to evaluate which merchants should receive "
        "proactive outreach. Phase 1: placeholder."
    ),
)
async def handle_tick(request: TickRequest) -> TickResponse:
    """Process a scheduled tick for proactive merchant engagement.

    Phase 2 will implement cadence planning, suppression checks,
    and batch message composition.
    """
    logger.info("POST /v1/tick — timestamp=%s", request.timestamp or "not_provided")
    logger.info("POST /v1/tick — validation successful")

    response = TickResponse(message="Tick endpoint.")
    logger.info("POST /v1/tick — responding: %s", response.message)
    return response
