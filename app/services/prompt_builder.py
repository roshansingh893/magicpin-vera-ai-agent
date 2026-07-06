"""Prompt builder — selects and assembles the correct prompt for a request.

This module is the single entry point for prompt construction.  It
inspects the request to determine whether to build a merchant-facing or
customer-facing prompt, then delegates to the appropriate builder.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.prompts.system_prompt import SYSTEM_PROMPT
from app.prompts.merchant_prompt import build_merchant_prompt
from app.prompts.customer_prompt import build_customer_prompt

if TYPE_CHECKING:
    from app.models.requests import (
        CategoryContext,
        CustomerContext,
        MerchantContext,
        TriggerContext,
    )

logger = logging.getLogger(__name__)


def build_prompts(
    category: CategoryContext,
    merchant: MerchantContext,
    trigger: TriggerContext,
    customer: CustomerContext | None = None,
) -> tuple[str, str]:
    """Build the (system_prompt, user_prompt) pair for the LLM call.

    Determines the correct prompt variant based on whether a customer
    context is present and delegates to the appropriate builder.

    Args:
        category: Vertical-level knowledge (voice, offers, peer stats).
        merchant: This specific merchant's state and history.
        trigger: The event prompting this message.
        customer: Optional customer context for customer-facing messages.

    Returns:
        A tuple of (system_prompt, user_prompt).
    """
    system = SYSTEM_PROMPT

    if customer is not None:
        logger.info(
            "Building customer-facing prompt — merchant=%s customer=%s trigger=%s",
            merchant.merchant_id,
            customer.customer_id,
            trigger.kind,
        )
        user = build_customer_prompt(category, merchant, trigger, customer)
    else:
        logger.info(
            "Building merchant-facing prompt — merchant=%s trigger=%s",
            merchant.merchant_id,
            trigger.kind,
        )
        user = build_merchant_prompt(category, merchant, trigger)

    logger.debug("System prompt: %d chars | User prompt: %d chars", len(system), len(user))
    return system, user
