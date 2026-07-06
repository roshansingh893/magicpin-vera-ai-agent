"""Merchant-facing prompt builder.

Converts structured context into a natural-language prompt when
the message is directed at the merchant (customer == None).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.requests import (
        CategoryContext,
        MerchantContext,
        TriggerContext,
    )


def build_merchant_prompt(
    category: CategoryContext,
    merchant: MerchantContext,
    trigger: TriggerContext,
) -> str:
    """Build a natural-language user prompt for merchant-facing messages.

    Converts the structured JSON contexts into readable prose so the
    LLM receives rich, organized information rather than raw data dumps.

    Args:
        category: The business vertical's knowledge pack.
        merchant: This specific merchant's current state.
        trigger: The event prompting this message.

    Returns:
        A fully-assembled user prompt string.
    """
    sections: list[str] = []

    # ── Merchant Identity ────────────────────────────────────────
    identity = merchant.identity
    salutation = identity.owner_first_name or identity.name
    locality_str = f"{identity.locality}, {identity.city}" if identity.locality else identity.city
    lang_str = ", ".join(identity.languages) if identity.languages else "English"

    sections.append(
        f"MERCHANT IDENTITY\n"
        f"- Name: {identity.name}\n"
        f"- Owner/Doctor first name: {salutation}\n"
        f"- Location: {locality_str}\n"
        f"- Verified: {'Yes' if identity.verified else 'No'}\n"
        f"- Preferred languages: {lang_str}\n"
        f"- Subscription: {merchant.subscription.status} ({merchant.subscription.plan})"
        + (f", {merchant.subscription.days_remaining} days remaining" if merchant.subscription.days_remaining else "")
    )

    # ── Category & Voice ─────────────────────────────────────────
    voice = category.voice
    voice_lines = [
        f"- Category: {category.display_name or category.slug}",
        f"- Tone: {voice.tone}",
        f"- Register: {voice.voice_register}",
        f"- Code-mix style: {voice.code_mix}",
    ]
    if voice.vocab_allowed:
        voice_lines.append(f"- Allowed vocabulary: {', '.join(voice.vocab_allowed[:10])}")
    if voice.vocab_taboo:
        voice_lines.append(f"- TABOO words (never use these): {', '.join(voice.vocab_taboo)}")
    if voice.tone_examples:
        voice_lines.append(f"- Tone examples: {' | '.join(voice.tone_examples)}")

    sections.append("CATEGORY VOICE PROFILE\n" + "\n".join(voice_lines))

    # ── Performance Metrics ──────────────────────────────────────
    perf = merchant.performance
    delta = perf.delta_7d
    perf_lines = [
        f"- 30-day views: {perf.views}",
        f"- 30-day calls: {perf.calls}",
        f"- 30-day directions: {perf.directions}",
        f"- CTR: {perf.ctr:.3f}",
        f"- Leads: {perf.leads}",
    ]
    if delta.views_pct is not None:
        perf_lines.append(f"- 7-day views change: {delta.views_pct:+.0%}")
    if delta.calls_pct is not None:
        perf_lines.append(f"- 7-day calls change: {delta.calls_pct:+.0%}")
    if delta.ctr_pct is not None:
        perf_lines.append(f"- 7-day CTR change: {delta.ctr_pct:+.0%}")

    sections.append("MERCHANT PERFORMANCE (last 30 days)\n" + "\n".join(perf_lines))

    # ── Peer Benchmarks ──────────────────────────────────────────
    peer = category.peer_stats
    peer_lines = [
        f"- Peer avg rating: {peer.avg_rating}",
        f"- Peer avg reviews: {peer.avg_review_count}",
        f"- Peer avg views (30d): {peer.avg_views_30d}",
        f"- Peer avg calls (30d): {peer.avg_calls_30d}",
        f"- Peer avg CTR: {peer.avg_ctr:.3f}",
    ]
    sections.append("PEER BENCHMARKS (same category)\n" + "\n".join(peer_lines))

    # ── Active Offers ────────────────────────────────────────────
    if merchant.offers:
        offer_lines = [
            f"- {o.title} (status: {o.status})" for o in merchant.offers
        ]
        sections.append("MERCHANT ACTIVE OFFERS\n" + "\n".join(offer_lines))

    # ── Customer Aggregate ───────────────────────────────────────
    if merchant.customer_aggregate:
        agg = merchant.customer_aggregate
        agg_lines = [f"- {k}: {v}" for k, v in agg.items()]
        sections.append("CUSTOMER AGGREGATE DATA\n" + "\n".join(agg_lines))

    # ── Signals ──────────────────────────────────────────────────
    if merchant.signals:
        sections.append("MERCHANT SIGNALS (issues/opportunities detected)\n" + "\n".join(f"- {s}" for s in merchant.signals))

    # ── Review Themes ────────────────────────────────────────────
    if merchant.review_themes:
        review_lines = [
            f"- \"{rt.theme}\" ({rt.sentiment}, {rt.occurrences_30d} mentions)"
            + (f" — common quote: \"{rt.common_quote}\"" if rt.common_quote else "")
            for rt in merchant.review_themes
        ]
        sections.append("RECENT REVIEW THEMES\n" + "\n".join(review_lines))

    # ── Conversation History ─────────────────────────────────────
    if merchant.conversation_history:
        history_lines = []
        for turn in merchant.conversation_history[-5:]:
            speaker = "Vera" if turn.source == "vera" else "Merchant"
            history_lines.append(f"  [{speaker}] {turn.body}")
        sections.append("RECENT CONVERSATION HISTORY (last few turns)\n" + "\n".join(history_lines))

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

    # ── Relevant Digest Items ────────────────────────────────────
    if category.digest:
        digest_lines = []
        for item in category.digest:
            line = f"- [{item.kind.upper()}] \"{item.title}\""
            if item.source:
                line += f" (source: {item.source})"
            if item.summary:
                line += f"\n    Summary: {item.summary}"
            if item.actionable:
                line += f"\n    Actionable: {item.actionable}"
            digest_lines.append(line)
        sections.append("CATEGORY DIGEST (recent research/news/compliance)\n" + "\n".join(digest_lines))

    # ── Seasonal Beats ───────────────────────────────────────────
    if category.seasonal_beats:
        beat_lines = [f"- {b.month_range}: {b.note}" for b in category.seasonal_beats]
        sections.append("SEASONAL PATTERNS\n" + "\n".join(beat_lines))

    # ── Offer Catalog (category-level) ───────────────────────────
    if category.offer_catalog:
        catalog_lines = [f"- {o.title}" for o in category.offer_catalog[:6]]
        sections.append("CATEGORY OFFER CATALOG (approved formats)\n" + "\n".join(catalog_lines))

    # ── Final Instruction ────────────────────────────────────────
    sections.append(
        "TASK\n"
        "Compose a WhatsApp message for this merchant. Requirements:\n"
        f"- Address the merchant as {salutation}.\n"
        "- Follow this 5-step reasoning pattern:\n"
        "  1. Explain WHY the merchant is receiving this message (mention the trigger in the first sentence).\n"
        "  2. Mention the exact metric, fact, or trigger.\n"
        "  3. Explain why it matters (use engagement principles like loss aversion, social proof).\n"
        "  4. Offer ONE concrete action based on context (e.g., refresh photos, launch an offer).\n"
        "  5. End with a compelling CTA.\n"
        "- Consider ALL merchant personalization fields (metrics, history, offers, signals) and naturally weave them into the message if useful.\n"
        "- Keep the message between 50–80 words (never exceed 100).\n"
        "- Adapt strictly to the category language.\n"
        "- Use send_as = \"vera\" (this is a merchant-facing message).\n"
        "- Return ONLY valid JSON. No markdown, no code fences, no explanations."
    )

    return "\n\n".join(sections)
