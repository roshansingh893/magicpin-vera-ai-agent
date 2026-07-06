"""Customer-facing prompt builder.

Converts structured context into a natural-language prompt when the
message is sent on behalf of the merchant to one of their customers.
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


def build_customer_prompt(
    category: CategoryContext,
    merchant: MerchantContext,
    trigger: TriggerContext,
    customer: CustomerContext,
) -> str:
    """Build a natural-language user prompt for customer-facing messages.

    These messages are sent from the merchant's WhatsApp number, drafted
    by Vera on behalf of the merchant.  The tone should be friendly and
    personal — not promotional or spammy.

    Args:
        category: The business vertical's knowledge pack.
        merchant: The merchant whose customer is being contacted.
        trigger: The event prompting this outreach.
        customer: The specific customer being contacted.

    Returns:
        A fully-assembled user prompt string.
    """
    sections: list[str] = []

    # ── Merchant Identity (the sender) ───────────────────────────
    identity = merchant.identity
    merchant_name = identity.name
    locality_str = f"{identity.locality}, {identity.city}" if identity.locality else identity.city

    sections.append(
        f"MERCHANT (the sender of this message)\n"
        f"- Business name: {merchant_name}\n"
        f"- Location: {locality_str}\n"
        f"- Category: {category.display_name or category.slug}"
    )

    # ── Customer Identity (the recipient) ────────────────────────
    cust_identity = customer.identity
    lang_pref = cust_identity.language_pref or "English"

    sections.append(
        f"CUSTOMER (the recipient)\n"
        f"- Name: {cust_identity.name}\n"
        f"- Language preference: {lang_pref}\n"
        f"- Age band: {cust_identity.age_band or 'unknown'}\n"
        f"- Customer state: {customer.state or 'unknown'}"
    )

    # ── Relationship History ─────────────────────────────────────
    rel = customer.relationship
    rel_lines = []
    if rel.first_visit:
        rel_lines.append(f"- First visit: {rel.first_visit}")
    if rel.last_visit:
        rel_lines.append(f"- Last visit: {rel.last_visit}")
    if rel.visits_total:
        rel_lines.append(f"- Total visits: {rel.visits_total}")
    if rel.services_received:
        rel_lines.append(f"- Services received: {', '.join(rel.services_received)}")
    if rel.lifetime_value:
        rel_lines.append(f"- Lifetime value: ₹{rel.lifetime_value:,.0f}")

    if rel_lines:
        sections.append("CUSTOMER RELATIONSHIP\n" + "\n".join(rel_lines))

    # ── Customer Preferences ─────────────────────────────────────
    if customer.preferences:
        pref_lines = [f"- {k}: {v}" for k, v in customer.preferences.items()]
        sections.append("CUSTOMER PREFERENCES\n" + "\n".join(pref_lines))

    # ── Consent ──────────────────────────────────────────────────
    consent = customer.consent
    consent_lines = []
    if consent.opted_in_at:
        consent_lines.append(f"- Opted in: {consent.opted_in_at}")
    if consent.scope:
        consent_lines.append(f"- Consent scope: {', '.join(consent.scope)}")
    if consent_lines:
        sections.append("CONSENT\n" + "\n".join(consent_lines))

    # ── Merchant Active Offers ───────────────────────────────────
    active_offers = [o for o in merchant.offers if o.status == "active"]
    if active_offers:
        offer_lines = [f"- {o.title}" for o in active_offers]
        sections.append("MERCHANT ACTIVE OFFERS (can be mentioned)\n" + "\n".join(offer_lines))

    # ── Category Voice Constraints ───────────────────────────────
    voice = category.voice
    voice_lines = [
        f"- Tone: {voice.tone}",
        f"- Code-mix: {voice.code_mix}",
    ]
    if voice.vocab_taboo:
        voice_lines.append(f"- TABOO words (never use): {', '.join(voice.vocab_taboo)}")
    sections.append("VOICE CONSTRAINTS\n" + "\n".join(voice_lines))

    # ── Trigger ──────────────────────────────────────────────────
    trigger_lines = [
        f"- Trigger ID: {trigger.id}",
        f"- Kind: {trigger.kind}",
        f"- Source: {trigger.source}",
        f"- Scope: {trigger.scope}",
        f"- Urgency: {trigger.urgency}/5",
    ]
    if trigger.suppression_key:
        trigger_lines.append(f"- Suppression key: {trigger.suppression_key}")
    if trigger.payload:
        for k, v in trigger.payload.items():
            if isinstance(v, dict):
                trigger_lines.append(f"- {k}:")
                for sk, sv in v.items():
                    trigger_lines.append(f"    - {sk}: {sv}")
            else:
                trigger_lines.append(f"- {k}: {v}")

    sections.append("TRIGGER (why this message is being sent NOW)\n" + "\n".join(trigger_lines))

    # ── Final Instruction ────────────────────────────────────────
    sections.append(
        "TASK\n"
        "Compose a WhatsApp message to this customer on behalf of the merchant. Requirements:\n"
        f"- Address the customer as {cust_identity.name}.\n"
        f"- The message should appear to come from {merchant_name}.\n"
        "- Explain WHY the customer is receiving this message in the first sentence, based on the exact trigger.\n"
        f"- Respect the customer's language preference strictly: {lang_pref}.\n"
        "- Be friendly and warm — NOT promotional or spammy.\n"
        "- Mention the merchant naturally (e.g., \"{merchant_name} here\").\n"
        "- Maximize personalization! Use visit history, services, and timing if relevant to the trigger.\n"
        "- Offer ONE concrete action or relevant offer to drive engagement.\n"
        "- End with a clear CTA.\n"
        "- Keep the message between 50–80 words (never exceed 100).\n"
        "- Use send_as = \"merchant_on_behalf\" (this is a customer-facing message).\n"
        "- Return ONLY valid JSON. No markdown, no code fences, no explanations."
    )

    return "\n\n".join(sections)
