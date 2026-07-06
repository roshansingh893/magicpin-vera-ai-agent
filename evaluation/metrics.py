"""Evaluation metrics — heuristic scoring for composed messages.

Scores messages on 5 dimensions (0-10 each), matching the AI judge
criteria from the magicpin challenge brief:

1. Specificity — concrete, verifiable facts (numbers, dates, names)
2. Merchant Fit — personalized to this merchant's state
3. Category Fit — vocabulary and voice match the vertical
4. Trigger Relevance — message clearly explains *why now*
5. Engagement — compulsion levers (CTA, curiosity, loss aversion)

All scoring is heuristic (no LLM calls). Fast and deterministic.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MetricScores:
    """Evaluation scores for a single composed message."""
    specificity: float
    merchant_fit: float
    category_fit: float
    trigger_relevance: float
    engagement: float

    @property
    def overall(self) -> float:
        """Weighted average across all dimensions."""
        return round(
            (self.specificity + self.merchant_fit + self.category_fit
             + self.trigger_relevance + self.engagement) / 5,
            2,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "specificity": round(self.specificity, 2),
            "merchant_fit": round(self.merchant_fit, 2),
            "category_fit": round(self.category_fit, 2),
            "trigger_relevance": round(self.trigger_relevance, 2),
            "engagement": round(self.engagement, 2),
            "overall": self.overall,
        }


# ──────────────────────────────────────────────────────────────────
# Category-specific vocabulary banks
# ──────────────────────────────────────────────────────────────────

_CATEGORY_VOCAB: dict[str, list[str]] = {
    "dentists": [
        "dental", "teeth", "cleaning", "fluoride", "caries", "scaling",
        "whitening", "bruxism", "aligner", "root canal", "implant",
        "periodontal", "endodontic", "rct", "opg", "varnish", "recall",
        "patient", "clinic", "appointment", "checkup", "oral",
    ],
    "salons": [
        "salon", "haircut", "styling", "hair", "beauty", "spa",
        "balayage", "keratin", "facial", "manicure", "pedicure",
        "bridal", "makeup", "stylist", "treatment", "color",
    ],
    "restaurants": [
        "restaurant", "menu", "delivery", "order", "dine", "dish",
        "thali", "cuisine", "food", "kitchen", "chef", "table",
        "reservation", "takeaway", "combo", "pizza", "lunch",
    ],
    "gyms": [
        "gym", "fitness", "workout", "training", "member", "membership",
        "exercise", "weight", "strength", "cardio", "yoga", "class",
        "session", "trial", "coach", "trainer", "studio",
    ],
    "pharmacies": [
        "pharmacy", "medicine", "prescription", "refill", "delivery",
        "otc", "drug", "dose", "molecule", "chronic", "senior",
        "recall", "batch", "stock", "health", "pharmacist",
    ],
}

# Engagement keywords and patterns
_ENGAGEMENT_PATTERNS = {
    "cta_binary": re.compile(r"\b(reply\s+(yes|stop|go|1|2))\b", re.IGNORECASE),
    "question_mark": re.compile(r"\?"),
    "loss_aversion": re.compile(
        r"\b(miss(?:ing|ed)?|lose|losing|lost|drop(?:ped)?|dip|before\s+it|window\s+closes?)\b",
        re.IGNORECASE,
    ),
    "social_proof": re.compile(
        r"\b(\d+\s+(?:dentists?|salons?|merchants?|businesses?|gyms?|pharmacies?|restaurants?)|peer|others?\s+in)\b",
        re.IGNORECASE,
    ),
    "curiosity": re.compile(
        r"\b(want\s+to\s+see|want\s+me\s+to|curious|interested\s+in|worth\s+a\s+look|check\s+this)\b",
        re.IGNORECASE,
    ),
    "effort_external": re.compile(
        r"\b(i('ve|.ve)?\s+(drafted|prepared|created|pulled)|ready\s+for\s+you|just\s+say|5[\s-]?min)\b",
        re.IGNORECASE,
    ),
    "urgency": re.compile(
        r"\b(today|tonight|this\s+week|right\s+now|limited|expires?|deadline|hurry|last\s+chance)\b",
        re.IGNORECASE,
    ),
}

# Generic phrases to penalize
_GENERIC_PHRASES = [
    "increase your sales",
    "boost your business",
    "grow your business",
    "improve your profile",
    "flat.*off",
    "amazing deal",
    "special offer",
    "don't miss out",
    "i hope you're doing well",
    "i'm reaching out",
    "we are pleased",
]


def score_specificity(body: str, merchant: dict[str, Any], trigger: dict[str, Any]) -> float:
    """Score how specific and fact-anchored the message is.

    Rewards: numbers, dates, merchant names, percentages, prices,
    source citations. Penalizes: generic language.
    """
    score = 5.0  # Base

    # Reward numbers (₹299, 2100-patient, 38%, etc.)
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", body)
    score += min(len(numbers) * 0.6, 2.5)

    # Reward currency amounts
    if re.search(r"₹[\d,]+", body):
        score += 0.5

    # Reward percentages
    pct_count = len(re.findall(r"\d+%", body))
    score += min(pct_count * 0.4, 1.0)

    # Reward source citations (JIDA, DCI, etc.)
    if re.search(r"\b(JIDA|DCI|IDA|p\.\d+|page\s+\d+)\b", body, re.IGNORECASE):
        score += 0.8

    # Reward specific names
    merchant_name = merchant.get("identity", {}).get("name", "")
    if merchant_name and merchant_name.lower() in body.lower():
        score += 0.3

    # Reward locality/city references
    locality = merchant.get("identity", {}).get("locality", "")
    city = merchant.get("identity", {}).get("city", "")
    if locality and locality.lower() in body.lower():
        score += 0.3
    if city and city.lower() in body.lower():
        score += 0.2

    # Penalize generic language
    for phrase in _GENERIC_PHRASES:
        if re.search(phrase, body, re.IGNORECASE):
            score -= 0.7

    return max(0.0, min(10.0, round(score, 2)))


def score_merchant_fit(body: str, merchant: dict[str, Any]) -> float:
    """Score how well the message is personalized to this merchant.

    Checks for references to: name, owner, city, subscription,
    performance numbers, offers, signals, conversation history.
    """
    score = 4.0  # Base

    identity = merchant.get("identity", {})
    name = identity.get("name", "")
    owner = identity.get("owner_first_name", "")
    city = identity.get("city", "")
    locality = identity.get("locality", "")

    body_lower = body.lower()

    # Name reference
    if name and name.lower() in body_lower:
        score += 1.0
    elif owner and owner.lower() in body_lower:
        score += 0.8

    # Location reference
    if locality and locality.lower() in body_lower:
        score += 0.5
    if city and city.lower() in body_lower:
        score += 0.3

    # Performance metrics mentioned
    perf = merchant.get("performance", {})
    if perf:
        views = str(perf.get("views", ""))
        calls = str(perf.get("calls", ""))
        ctr = str(perf.get("ctr", ""))
        for val in [views, calls, ctr]:
            if val and val in body:
                score += 0.6

    # Offers referenced
    for offer in merchant.get("offers", []):
        title = offer.get("title", "")
        if title and title.lower() in body_lower:
            score += 0.8

    # Signals mentioned
    for signal in merchant.get("signals", []):
        clean_signal = signal.split(":")[0].replace("_", " ")
        if clean_signal in body_lower:
            score += 0.4

    # Subscription status
    sub = merchant.get("subscription", {})
    if sub.get("status") and sub["status"] in body_lower:
        score += 0.3

    return max(0.0, min(10.0, round(score, 2)))


def score_category_fit(body: str, category: dict[str, Any]) -> float:
    """Score category-appropriate vocabulary and voice usage.

    Checks for category-specific vocabulary and penalizes
    voice taboo violations.
    """
    score = 5.0  # Base
    slug = category.get("slug", "")
    body_lower = body.lower()

    # Category vocabulary match
    vocab = _CATEGORY_VOCAB.get(slug, [])
    matches = sum(1 for word in vocab if word in body_lower)
    score += min(matches * 0.4, 3.0)

    # Voice taboo violations
    voice = category.get("voice", {})
    taboos = voice.get("vocab_taboo", [])
    for taboo in taboos:
        if taboo.lower() in body_lower:
            score -= 1.5

    # Tone check — promotional language in clinical categories
    if slug in ("dentists", "pharmacies"):
        promo_patterns = [r"amazing", r"incredible", r"best\s+deal", r"limited\s+time\s+offer"]
        for p in promo_patterns:
            if re.search(p, body, re.IGNORECASE):
                score -= 1.0

    return max(0.0, min(10.0, round(score, 2)))


def score_trigger_relevance(body: str, trigger: dict[str, Any]) -> float:
    """Score how clearly the message explains *why now*.

    The message should anchor on the trigger event.
    """
    score = 4.0  # Base
    kind = trigger.get("kind", "").lower().replace("_", " ")
    payload = trigger.get("payload", {})
    body_lower = body.lower()

    # Kind reference (e.g., "research digest" → "research")
    kind_words = kind.split()
    for word in kind_words:
        if len(word) > 3 and word in body_lower:
            score += 0.8

    # Payload data referenced
    for key, value in payload.items():
        if isinstance(value, str) and len(value) > 3 and value.lower() in body_lower:
            score += 0.6
        elif isinstance(value, (int, float)) and str(value) in body:
            score += 0.5

    # Urgency alignment
    urgency = trigger.get("urgency", 1)
    if urgency >= 4 and re.search(r"\b(urgent|important|critical|immediately|asap)\b", body, re.IGNORECASE):
        score += 0.5
    elif urgency >= 3 and re.search(r"\b(soon|this\s+week|today|time.?sensitive)\b", body, re.IGNORECASE):
        score += 0.3

    return max(0.0, min(10.0, round(score, 2)))


def score_engagement(body: str, cta: str) -> float:
    """Score engagement compulsion — would the merchant want to reply?

    Rewards: strong CTA, curiosity, urgency, social proof, loss
    aversion, effort externalization.
    """
    score = 4.0  # Base

    # CTA quality
    if cta == "binary_yes_stop":
        score += 1.0
    elif cta == "open_ended":
        score += 0.5

    # Engagement pattern matches
    for name, pattern in _ENGAGEMENT_PATTERNS.items():
        if pattern.search(body):
            score += 0.6

    # Question at end (drives reply)
    sentences = body.strip().split(".")
    last_sentence = sentences[-1] if sentences else ""
    if "?" in last_sentence:
        score += 0.5

    # Penalize weak endings
    weak_endings = [
        r"let\s+me\s+know",
        r"feel\s+free\s+to",
        r"don't\s+hesitate",
        r"if\s+you\s+have\s+any\s+questions",
    ]
    for pattern in weak_endings:
        if re.search(pattern, body, re.IGNORECASE):
            score -= 0.5

    # Message length check (too long = lower engagement)
    word_count = len(body.split())
    if word_count > 120:
        score -= 1.0
    elif word_count > 80:
        score -= 0.3

    return max(0.0, min(10.0, round(score, 2)))


def evaluate_message(
    body: str,
    cta: str,
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
) -> MetricScores:
    """Run all 5 metrics on a composed message.

    Args:
        body: The message body text.
        cta: The CTA type string.
        category: Category context dict.
        merchant: Merchant context dict.
        trigger: Trigger context dict.

    Returns:
        MetricScores with all 5 dimensions scored 0-10.
    """
    scores = MetricScores(
        specificity=score_specificity(body, merchant, trigger),
        merchant_fit=score_merchant_fit(body, merchant),
        category_fit=score_category_fit(body, category),
        trigger_relevance=score_trigger_relevance(body, trigger),
        engagement=score_engagement(body, cta),
    )

    logger.debug(
        "Evaluation: specificity=%.1f merchant=%.1f category=%.1f trigger=%.1f engagement=%.1f overall=%.1f",
        scores.specificity,
        scores.merchant_fit,
        scores.category_fit,
        scores.trigger_relevance,
        scores.engagement,
    )

    return scores
