"""Evaluator — wraps metrics scoring with output validation.

Combines output validation (structural correctness) with
quality evaluation (the 5-dimension scoring).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from evaluation.metrics import MetricScores, evaluate_message

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Full evaluation of a single composed message."""
    test_id: str
    merchant_id: str
    trigger_kind: str
    category_slug: str

    # The composed output
    body: str
    cta: str
    send_as: str
    suppression_key: str
    rationale: str

    # Quality scores
    scores: MetricScores

    # Metadata
    latency_ms: float = 0.0
    prompt_version: str = "default"
    valid: bool = True
    validation_errors: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "merchant_id": self.merchant_id,
            "trigger_kind": self.trigger_kind,
            "category_slug": self.category_slug,
            "body": self.body,
            "cta": self.cta,
            "send_as": self.send_as,
            "suppression_key": self.suppression_key,
            "rationale": self.rationale,
            "scores": self.scores.to_dict(),
            "latency_ms": round(self.latency_ms, 1),
            "prompt_version": self.prompt_version,
            "valid": self.valid,
        }

    def to_submission_line(self) -> dict[str, str]:
        """Format for submission.jsonl (challenge spec)."""
        return {
            "test_id": self.test_id,
            "body": self.body,
            "cta": self.cta,
            "send_as": self.send_as,
            "suppression_key": self.suppression_key,
            "rationale": self.rationale,
        }


def validate_output(result: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a composed message output for structural correctness.

    Checks:
    - body exists and is non-empty
    - cta is valid
    - send_as is valid
    - suppression_key exists
    - rationale exists
    - No null values
    - No markdown/code fences

    Returns:
        (is_valid, list_of_errors)
    """
    errors: list[str] = []

    body = result.get("body", "")
    if not body or not isinstance(body, str):
        errors.append("body is missing or empty")
    elif len(body.strip()) < 10:
        errors.append(f"body too short: {len(body.strip())} chars")

    cta = result.get("cta", "")
    valid_ctas = {"binary_yes_stop", "open_ended", "none"}
    if cta not in valid_ctas:
        errors.append(f"Invalid cta: '{cta}'. Must be one of {valid_ctas}")

    send_as = result.get("send_as", "")
    valid_send_as = {"vera", "merchant_on_behalf"}
    if send_as not in valid_send_as:
        errors.append(f"Invalid send_as: '{send_as}'. Must be one of {valid_send_as}")

    suppression_key = result.get("suppression_key", "")
    if not suppression_key or not isinstance(suppression_key, str):
        errors.append("suppression_key is missing or empty")

    rationale = result.get("rationale", "")
    if not rationale or not isinstance(rationale, str):
        errors.append("rationale is missing or empty")

    # Check for null values
    for key in ["body", "cta", "send_as", "suppression_key", "rationale"]:
        if result.get(key) is None:
            errors.append(f"{key} is null")

    # Check for markdown / code fences in body
    if body:
        if "```" in body:
            errors.append("body contains code fences")
        if body.startswith("#") or "**" in body:
            errors.append("body contains markdown formatting")

    return len(errors) == 0, errors


def evaluate_composed_output(
    test_id: str,
    output: dict[str, Any],
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    latency_ms: float = 0.0,
    prompt_version: str = "default",
) -> EvaluationResult:
    """Validate and score a composed message output.

    Args:
        test_id: Identifier for this test case.
        output: The ComposedMessage as a dict.
        category: Category context.
        merchant: Merchant context.
        trigger: Trigger context.
        latency_ms: Time taken to generate.
        prompt_version: Which prompt version was used.

    Returns:
        An EvaluationResult with validation status and quality scores.
    """
    is_valid, validation_errors = validate_output(output)

    body = output.get("body", "")
    cta = output.get("cta", "")

    # Score even if invalid (for diagnostics)
    scores = evaluate_message(body, cta, category, merchant, trigger)

    return EvaluationResult(
        test_id=test_id,
        merchant_id=merchant.get("merchant_id", "unknown"),
        trigger_kind=trigger.get("kind", "unknown"),
        category_slug=category.get("slug", "unknown"),
        body=body,
        cta=cta,
        send_as=output.get("send_as", ""),
        suppression_key=output.get("suppression_key", ""),
        rationale=output.get("rationale", ""),
        scores=scores,
        latency_ms=latency_ms,
        prompt_version=prompt_version,
        valid=is_valid,
        validation_errors=validation_errors if not is_valid else None,
    )
