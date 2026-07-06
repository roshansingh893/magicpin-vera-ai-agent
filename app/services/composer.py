"""Message composition service — placeholder for Phase 2.

This module will contain the core orchestration logic:
1. Receive validated contexts (category, merchant, trigger, customer).
2. Select the appropriate prompt template based on trigger kind.
3. Call the LLM client with the assembled prompt.
4. Parse and validate the structured response.
5. Apply post-generation checks (CTA shape, language match, anti-hallucination).

Architecture notes:
- The composer is a pure function (no side effects) — easy to test.
- It depends on the prompt layer (app.prompts) and LLM layer (app.llm).
- It does NOT know about FastAPI, HTTP, or request objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.requests import (
        CategoryContext,
        CustomerContext,
        MerchantContext,
        TriggerContext,
    )
    from app.models.responses import ComposedMessage


async def compose(
    category: CategoryContext,
    merchant: MerchantContext,
    trigger: TriggerContext,
    customer: CustomerContext | None = None,
) -> ComposedMessage:
    """Compose a WhatsApp message from the 4-context framework.

    Phase 2 implementation. Currently raises NotImplementedError.

    Args:
        category: Vertical-level knowledge (voice, offers, peer stats).
        merchant: This specific merchant's state and history.
        trigger: The event prompting this message.
        customer: Optional customer context for customer-facing messages.

    Returns:
        A fully populated ComposedMessage.

    Raises:
        NotImplementedError: Always, until Phase 2 implementation.
    """
    raise NotImplementedError("Message composition will be implemented in Phase 2.")
