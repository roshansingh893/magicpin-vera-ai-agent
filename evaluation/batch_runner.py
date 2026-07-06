"""Batch runner — runs compose() across evaluation scenarios.

Drives the production compose() pipeline over an entire dataset,
collecting outputs, evaluating quality, and producing summaries.

This is the orchestration engine for offline evaluation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.dataset_loader import EvaluationScenario
from evaluation.evaluator import EvaluationResult, evaluate_composed_output

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Aggregated results from a batch run."""
    results: list[EvaluationResult] = field(default_factory=list)
    total: int = 0
    successful: int = 0
    failed: int = 0
    invalid: int = 0
    total_latency_ms: float = 0.0
    prompt_version: str = "default"

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.successful, 1)

    @property
    def avg_overall_score(self) -> float:
        scores = [r.scores.overall for r in self.results if r.valid]
        return sum(scores) / max(len(scores), 1)

    @property
    def failure_rate(self) -> float:
        return self.failed / max(self.total, 1)

    def avg_metric(self, metric: str) -> float:
        values = [getattr(r.scores, metric) for r in self.results if r.valid]
        return sum(values) / max(len(values), 1)

    def top_results(self, n: int = 10) -> list[EvaluationResult]:
        sorted_results = sorted(
            [r for r in self.results if r.valid],
            key=lambda r: r.scores.overall,
            reverse=True,
        )
        return sorted_results[:n]

    def worst_results(self, n: int = 10) -> list[EvaluationResult]:
        sorted_results = sorted(
            [r for r in self.results if r.valid],
            key=lambda r: r.scores.overall,
        )
        return sorted_results[:n]

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "invalid": self.invalid,
            "failure_rate": round(self.failure_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_overall_score": round(self.avg_overall_score, 2),
            "avg_specificity": round(self.avg_metric("specificity"), 2),
            "avg_merchant_fit": round(self.avg_metric("merchant_fit"), 2),
            "avg_category_fit": round(self.avg_metric("category_fit"), 2),
            "avg_trigger_relevance": round(self.avg_metric("trigger_relevance"), 2),
            "avg_engagement": round(self.avg_metric("engagement"), 2),
            "prompt_version": self.prompt_version,
        }


async def run_single_scenario(
    scenario: EvaluationScenario,
    prompt_version: str = "default",
    max_retries: int = 3,
) -> EvaluationResult:
    """Run compose() on a single scenario and evaluate the result.

    Args:
        scenario: The evaluation scenario.
        prompt_version: Label for the prompt variant.
        max_retries: How many times to retry on ServiceUnavailableError.

    Returns:
        An EvaluationResult.
    """
    from app.models.requests import (
        CategoryContext,
        ComposeRequest,
        CustomerContext,
        MerchantContext,
        TriggerContext,
    )
    from app.services.composer import compose
    from app.core.exceptions import ServiceUnavailableError

    logger.info(
        "Running scenario %s: merchant=%s trigger=%s",
        scenario.test_id,
        scenario.merchant_id,
        scenario.trigger_kind,
    )

    try:
        # Build Pydantic models from raw dicts
        category = CategoryContext(**scenario.category)
        merchant = MerchantContext(**scenario.merchant)
        trigger = TriggerContext(**scenario.trigger)
        customer = CustomerContext(**scenario.customer) if scenario.customer else None

        for attempt in range(1, max_retries + 1):
            start = time.monotonic()
            try:
                result = await compose(category, merchant, trigger, customer)
                latency_ms = (time.monotonic() - start) * 1000
                break
            except ServiceUnavailableError as e:
                if attempt < max_retries:
                    logger.warning("Scenario %s encountered LLM error (%s). Retrying %d/%d...", scenario.test_id, e, attempt, max_retries)
                    await asyncio.sleep(2.0 * attempt)
                else:
                    raise

        output = result.model_dump()

        eval_result = evaluate_composed_output(
            test_id=scenario.test_id,
            output=output,
            category=scenario.category,
            merchant=scenario.merchant,
            trigger=scenario.trigger,
            latency_ms=latency_ms,
            prompt_version=prompt_version,
        )

        logger.info(
            "Scenario %s complete: score=%.1f latency=%.0fms",
            scenario.test_id,
            eval_result.scores.overall,
            latency_ms,
        )
        return eval_result

    except Exception as exc:
        logger.error("Scenario %s failed: %s", scenario.test_id, exc)
        from evaluation.metrics import MetricScores

        return EvaluationResult(
            test_id=scenario.test_id,
            merchant_id=scenario.merchant_id,
            trigger_kind=scenario.trigger_kind,
            category_slug=scenario.category_slug,
            body="",
            cta="",
            send_as="",
            suppression_key="",
            rationale=f"FAILED: {exc}",
            scores=MetricScores(0, 0, 0, 0, 0),
            latency_ms=0,
            prompt_version=prompt_version,
            valid=False,
            validation_errors=[str(exc)],
        )


async def run_batch(
    scenarios: list[EvaluationScenario],
    prompt_version: str = "default",
    concurrency: int = 1,
    delay_seconds: float = 1.0,
) -> BatchResult:
    """Run compose() on a batch of scenarios.

    Processes sequentially by default to respect LLM rate limits.
    Set concurrency > 1 for parallel execution (use with caution).

    Args:
        scenarios: List of evaluation scenarios.
        prompt_version: Label for the prompt variant.
        concurrency: Number of concurrent executions.
        delay_seconds: Delay between sequential calls.

    Returns:
        Aggregated BatchResult.
    """
    batch = BatchResult(prompt_version=prompt_version)
    batch.total = len(scenarios)

    logger.info("Starting batch run: %d scenarios, concurrency=%d", len(scenarios), concurrency)

    for i, scenario in enumerate(scenarios, start=1):
        logger.info("Progress: %d/%d", i, batch.total)

        result = await run_single_scenario(scenario, prompt_version)
        batch.results.append(result)

        if result.valid:
            batch.successful += 1
            batch.total_latency_ms += result.latency_ms
        else:
            if result.validation_errors and any("FAILED" in e for e in result.validation_errors):
                batch.failed += 1
            else:
                batch.invalid += 1

        # Rate limiting delay
        if i < batch.total and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

    logger.info(
        "Batch complete: %d/%d successful, avg_score=%.2f, avg_latency=%.0fms",
        batch.successful,
        batch.total,
        batch.avg_overall_score,
        batch.avg_latency_ms,
    )

    return batch


def save_batch_results(batch: BatchResult, output_dir: Path) -> None:
    """Save batch results to disk.

    Writes:
    - results.json — all individual results
    - summary.json — aggregate metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Individual results
    results_file = output_dir / "results.json"
    results_data = [r.to_dict() for r in batch.results]
    results_file.write_text(json.dumps(results_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    summary_file = output_dir / "summary.json"
    summary_file.write_text(json.dumps(batch.summary(), indent=2), encoding="utf-8")

    logger.info("Batch results saved to %s", output_dir)
