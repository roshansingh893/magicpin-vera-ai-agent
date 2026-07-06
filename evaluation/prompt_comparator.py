"""Prompt comparator — A/B/C test prompt versions on identical scenarios.

Runs the same scenarios with different prompt configurations and
compares average scores, latency, and failure rates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from evaluation.batch_runner import BatchResult, run_batch
from evaluation.dataset_loader import EvaluationScenario

logger = logging.getLogger(__name__)


@dataclass
class PromptComparison:
    """Side-by-side comparison of prompt versions."""
    versions: dict[str, BatchResult] = field(default_factory=dict)

    @property
    def best_version(self) -> str:
        """Return the version with the highest average overall score."""
        if not self.versions:
            return "none"
        return max(self.versions, key=lambda v: self.versions[v].avg_overall_score)

    def comparison_table(self) -> list[dict[str, Any]]:
        """Generate a comparison table across all versions."""
        rows = []
        for version, batch in self.versions.items():
            rows.append({
                "version": version,
                "avg_overall": round(batch.avg_overall_score, 2),
                "avg_specificity": round(batch.avg_metric("specificity"), 2),
                "avg_merchant_fit": round(batch.avg_metric("merchant_fit"), 2),
                "avg_category_fit": round(batch.avg_metric("category_fit"), 2),
                "avg_trigger_relevance": round(batch.avg_metric("trigger_relevance"), 2),
                "avg_engagement": round(batch.avg_metric("engagement"), 2),
                "avg_latency_ms": round(batch.avg_latency_ms, 1),
                "failure_rate": round(batch.failure_rate, 4),
                "total": batch.total,
                "successful": batch.successful,
            })
        return sorted(rows, key=lambda r: r["avg_overall"], reverse=True)


async def compare_prompts(
    scenarios: list[EvaluationScenario],
    versions: list[str],
    setup_fn: Callable[[str], Awaitable[None]] | None = None,
    delay_seconds: float = 1.0,
) -> PromptComparison:
    """Run the same scenarios across multiple prompt versions.

    Args:
        scenarios: Scenarios to evaluate.
        versions: List of prompt version labels.
        setup_fn: Optional async function called before each version
                  to configure the prompt (e.g., swap system prompt).
        delay_seconds: Delay between LLM calls.

    Returns:
        PromptComparison with results for all versions.
    """
    comparison = PromptComparison()

    for version in versions:
        logger.info("Running prompt version: %s", version)

        if setup_fn:
            await setup_fn(version)

        batch = await run_batch(
            scenarios,
            prompt_version=version,
            delay_seconds=delay_seconds,
        )
        comparison.versions[version] = batch

        logger.info(
            "Version %s complete: avg_score=%.2f avg_latency=%.0fms",
            version,
            batch.avg_overall_score,
            batch.avg_latency_ms,
        )

    best = comparison.best_version
    logger.info("Best prompt version: %s (score=%.2f)", best, comparison.versions[best].avg_overall_score)

    return comparison
