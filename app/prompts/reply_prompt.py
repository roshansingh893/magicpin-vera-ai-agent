"""Reply prompt builder — multi-turn conversation context for the LLM.

Builds a user prompt that includes conversation summary, goals,
merchant context, and instructions to continue the conversation
naturally.

Phase 3.5 enhancements:
- Conversation summary instead of raw history dump.
- Goal-aware instructions.
- Stage-specific task framing.
- Anti-repetition rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.services.conversation_intelligence import build_conversation_summary

if TYPE_CHECKING:
    from app.models.conversation import ConversationState


def build_reply_prompt(state: ConversationState) -> str:
    """Build a user prompt for generating a reply in an ongoing conversation.

    Uses a structured conversation summary instead of dumping every
    message verbatim.  Includes the conversation goal so the LLM
    knows what it is trying to achieve.

    Args:
        state: The current conversation state with history.

    Returns:
        A fully-assembled user prompt string.
    """
    sections: list[str] = []

    # ── Conversation metadata ────────────────────────────────────
    sections.append(
        f"CONVERSATION CONTEXT\n"
        f"- Conversation ID: {state.conversation_id}\n"
        f"- Merchant ID: {state.merchant_id}\n"
        f"- Current stage: {state.stage.value}\n"
        f"- Original trigger: {state.last_trigger_kind}\n"
        f"- Follow-ups sent: {state.follow_up_count}"
    )

    # ── Conversation summary (replaces raw history) ──────────────
    summary = build_conversation_summary(state)
    if summary:
        sections.append(summary)

    # ── Last merchant message (always include verbatim) ──────────
    last_merchant_msg = None
    for msg in reversed(state.history):
        if msg.role.value == "merchant":
            last_merchant_msg = msg
            break

    if last_merchant_msg:
        intent_str = ""
        if last_merchant_msg.intent:
            conf_str = ""
            if last_merchant_msg.confidence is not None:
                conf_str = f", confidence={last_merchant_msg.confidence:.2f}"
            intent_str = f"\n  Detected intent: {last_merchant_msg.intent.value}{conf_str}"
        sections.append(
            f"LATEST MERCHANT MESSAGE\n"
            f"  \"{last_merchant_msg.body}\"{intent_str}"
        )

    # ── Conversation goal ────────────────────────────────────────
    if state.goal.description:
        sections.append(
            f"YOUR GOAL IN THIS CONVERSATION\n"
            f"  {state.goal.description}\n"
            f"  Work toward this goal. If the merchant has declined, respect that."
        )

    # ── Previous bot messages (anti-repetition) ──────────────────
    prev_bot_msgs = [
        m.body[:80] for m in state.history
        if m.role.value == "vera"
    ]
    if prev_bot_msgs:
        sections.append(
            "PREVIOUS VERA MESSAGES (do NOT repeat these):\n" + "\n".join(
                f"  • {m}" for m in prev_bot_msgs[-3:]  # Last 3 only
            )
        )

    # ── Task instruction (stage-aware) ───────────────────────────
    task = _build_task_instruction(state)
    sections.append(task)

    return "\n\n".join(sections)


def _build_task_instruction(state: ConversationState) -> str:
    """Generate stage-specific instructions for the LLM.

    Different stages require different response strategies.
    """
    base = (
        "TASK\n"
        "Continue this conversation naturally as Vera. Requirements:\n"
        "- You are responding to the merchant's latest message.\n"
        "- Do NOT restart or repeat any previous message.\n"
        "- Reference what the merchant said to show you understood.\n"
    )

    stage = state.stage.value

    if stage == "resolved":
        # Merchant replied — continue toward goal
        base += (
            "- The merchant has engaged. Move the conversation forward.\n"
            "- If they showed interest or asked for suggestions, provide 1-2 actionable tips immediately.\n"
            "- Do NOT string the user along or ask for permission to share the tips (e.g., don't say 'Shall I share them?'). Just share them.\n"
            "- If they asked a specific question, answer it using only known context.\n"
            "- Offer a concrete next step.\n"
        )
    elif stage == "follow_up_sent":
        # Follow-up was sent — be more concise
        base += (
            "- This is after a follow-up. Be brief and direct.\n"
            "- Acknowledge the merchant's response warmly.\n"
            "- Get straight to the value you can provide.\n"
        )
    else:
        base += (
            "- If the merchant showed interest, provide the specific help you offered.\n"
            "- If the merchant asked a question, answer it using only known context.\n"
        )

    base += (
        "- Keep your reply concise (50–80 words, never exceed 100).\n"
        "- End with a clear next step or CTA.\n"
        "- Use send_as = \"vera\".\n"
        "- Return ONLY valid JSON. No markdown, no code fences."
    )

    return base
