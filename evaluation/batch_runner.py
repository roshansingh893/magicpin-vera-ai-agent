"""Batch runner — runs compose() across evaluation scenarios.

Drives the production compose() pipeline over an entire dataset,
collecting outputs, evaluating quality, and producing summaries.

This is the orchestration engine for offline evaluation.

Phase 5 additions:
- Response cache integration (never re-call Groq for identical requests)
- Resume support (--resume skips already-completed scenarios)
- Dry-run mode (validate dataset + estimate API calls without calling Groq)
- Smart rate limiting (auto-pause when approaching Groq RPM limits)
- Improved retry strategy (retry only transient errors, max 3 attempts)
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
    cache_hits: int = 0
    cache_misses: int = 0
    skipped: int = 0

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
            "skipped": self.skipped,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
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


def _is_retryable_error(exc: Exception) -> bool:
    """Determine if an error is transient and worth retrying.

    Retryable: timeout, rate limit, connection failures.
    NOT retryable: auth errors, invalid prompts, persistent validation failures.
    """
    from app.core.exceptions import ServiceUnavailableError

    if isinstance(exc, ServiceUnavailableError):
        msg = str(exc).lower()
        # Auth errors should NOT be retried
        if "auth" in msg or "api key" in msg or "unauthorized" in msg:
            return False
        return True

    # Connection and timeout errors from the openai SDK
    exc_name = type(exc).__name__
    retryable_types = {"APITimeoutError", "APIConnectionError", "RateLimitError"}
    return exc_name in retryable_types


async def run_single_scenario(
    scenario: EvaluationScenario,
    prompt_version: str = "default",
    max_retries: int = 3,
    cache: Any = None,
    rate_limiter: Any = None,
) -> EvaluationResult:
    """Run compose() on a single scenario and evaluate the result.

    Args:
        scenario: The evaluation scenario.
        prompt_version: Label for the prompt variant.
        max_retries: How many times to retry on transient errors.
        cache: Optional ResponseCache instance.
        rate_limiter: Optional RateLimiter instance.

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
        # ── Check cache first ────────────────────────────────────
        if cache is not None:
            cached = cache.get(
                merchant_id=scenario.merchant_id,
                category_slug=scenario.category_slug,
                trigger_kind=scenario.trigger_kind,
                customer_id=scenario.customer.get("customer_id") if scenario.customer else None,
                prompt_version=prompt_version,
            )
            if cached is not None:
                logger.info("Cache HIT for scenario %s — skipping LLM call", scenario.test_id)
                eval_result = evaluate_composed_output(
                    test_id=scenario.test_id,
                    output=cached,
                    category=scenario.category,
                    merchant=scenario.merchant,
                    trigger=scenario.trigger,
                    latency_ms=0.0,
                    prompt_version=prompt_version,
                )
                return eval_result

        # Build Pydantic models from raw dicts
        category = CategoryContext(**scenario.category)
        merchant = MerchantContext(**scenario.merchant)
        trigger = TriggerContext(**scenario.trigger)
        customer = CustomerContext(**scenario.customer) if scenario.customer else None

        for attempt in range(1, max_retries + 1):
            # ── Rate limiting ────────────────────────────────────
            if rate_limiter is not None:
                rate_limiter.wait_if_needed()

            start = time.monotonic()
            try:
                result = await compose(category, merchant, trigger, customer)
                latency_ms = (time.monotonic() - start) * 1000

                # Record the request in the rate limiter
                if rate_limiter is not None:
                    rate_limiter.record_request()

                break
            except ServiceUnavailableError as e:
                if rate_limiter is not None:
                    rate_limiter.record_request()

                if not _is_retryable_error(e):
                    logger.error("Non-retryable error for scenario %s: %s", scenario.test_id, e)
                    raise

                if attempt < max_retries:
                    wait = 2.0 * attempt
                    logger.warning(
                        "Scenario %s: transient error (%s). Retrying %d/%d after %.1fs…",
                        scenario.test_id, e, attempt, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise

        output = result.model_dump()

        # ── Store in cache ───────────────────────────────────────
        if cache is not None:
            cache.put(
                merchant_id=scenario.merchant_id,
                category_slug=scenario.category_slug,
                trigger_kind=scenario.trigger_kind,
                response=output,
                customer_id=scenario.customer.get("customer_id") if scenario.customer else None,
                prompt_version=prompt_version,
                test_id=scenario.test_id,
            )

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


def dry_run_batch(
    scenarios: list[EvaluationScenario],
    prompt_version: str = "default",
    cache: Any = None,
) -> dict[str, Any]:
    """Validate dataset and estimate API calls without calling Groq.

    Args:
        scenarios: List of evaluation scenarios.
        prompt_version: Label for the prompt variant.
        cache: Optional ResponseCache to check for existing results.

    Returns:
        A dict with validation results and estimates.
    """
    total = len(scenarios)
    cached_count = 0
    needs_api = 0
    validation_errors: list[str] = []

    for scenario in scenarios:
        # Validate scenario structure
        if not scenario.merchant_id:
            validation_errors.append(f"{scenario.test_id}: missing merchant_id")
        if not scenario.trigger_kind:
            validation_errors.append(f"{scenario.test_id}: missing trigger_kind")
        if not scenario.category_slug:
            validation_errors.append(f"{scenario.test_id}: missing category_slug")

        # Check cache
        if cache is not None:
            has_cached = cache.has(
                merchant_id=scenario.merchant_id,
                category_slug=scenario.category_slug,
                trigger_kind=scenario.trigger_kind,
                customer_id=scenario.customer.get("customer_id") if scenario.customer else None,
                prompt_version=prompt_version,
            )
            # Also check by test_id (from submission warmup)
            has_tid = cache.get_by_test_id(scenario.test_id) is not None
            if has_cached or has_tid:
                cached_count += 1
            else:
                needs_api += 1
        else:
            needs_api += 1

    # Estimate time: ~3 seconds per API call (including rate limiting delay)
    est_seconds = needs_api * 3.0
    est_minutes = est_seconds / 60.0

    report = {
        "total_scenarios": total,
        "cached": cached_count,
        "needs_api_calls": needs_api,
        "validation_errors": validation_errors,
        "valid": len(validation_errors) == 0,
        "estimated_api_calls": needs_api,
        "estimated_time_seconds": round(est_seconds, 1),
        "estimated_time_minutes": round(est_minutes, 1),
        "prompt_version": prompt_version,
    }

    logger.info(
        "DRY RUN: %d scenarios, %d cached, %d need API calls, est. %.1f min",
        total, cached_count, needs_api, est_minutes,
    )

    return report


async def run_batch(
    scenarios: list[EvaluationScenario],
    prompt_version: str = "default",
    concurrency: int = 1,
    delay_seconds: float = 1.0,
    cache: Any = None,
    rate_limiter: Any = None,
    resume_from: int = 0,
    completed_test_ids: set[str] | None = None,
) -> BatchResult:
    """Run compose() on a batch of scenarios.

    Processes sequentially by default to respect LLM rate limits.
    Set concurrency > 1 for parallel execution (use with caution).

    Args:
        scenarios: List of evaluation scenarios.
        prompt_version: Label for the prompt variant.
        concurrency: Number of concurrent executions.
        delay_seconds: Delay between sequential calls.
        cache: Optional ResponseCache instance.
        rate_limiter: Optional RateLimiter instance.
        resume_from: Skip the first N scenarios (0-indexed).
        completed_test_ids: Set of test IDs already completed (for resume).

    Returns:
        Aggregated BatchResult.
    """
    batch = BatchResult(prompt_version=prompt_version)
    batch.total = len(scenarios)

    if completed_test_ids is None:
        completed_test_ids = set()

    logger.info(
        "Starting batch run: %d scenarios, concurrency=%d, resume_from=%d, pre-completed=%d",
        len(scenarios), concurrency, resume_from, len(completed_test_ids),
    )

    for i, scenario in enumerate(scenarios, start=1):
        # ── Resume support: skip already-completed scenarios ─────
        if i <= resume_from:
            batch.skipped += 1
            logger.debug("Skipping scenario %d (resume_from=%d)", i, resume_from)
            continue

        if scenario.test_id in completed_test_ids:
            batch.skipped += 1
            logger.debug("Skipping scenario %s (already completed)", scenario.test_id)

            # Try to load the cached result for inclusion in batch results
            if cache is not None:
                cached = cache.get_by_test_id(scenario.test_id)
                if cached is not None:
                    eval_result = evaluate_composed_output(
                        test_id=scenario.test_id,
                        output=cached,
                        category=scenario.category,
                        merchant=scenario.merchant,
                        trigger=scenario.trigger,
                        latency_ms=0.0,
                        prompt_version=prompt_version,
                    )
                    batch.results.append(eval_result)
                    batch.successful += 1
                    batch.cache_hits += 1
            continue

        logger.info("Progress: %d/%d", i, batch.total)

        result = await run_single_scenario(
            scenario, prompt_version, cache=cache, rate_limiter=rate_limiter,
        )
        batch.results.append(result)

        if result.valid:
            batch.successful += 1
            batch.total_latency_ms += result.latency_ms
            if result.latency_ms == 0.0:
                batch.cache_hits += 1
            else:
                batch.cache_misses += 1
        else:
            if result.validation_errors and any("FAILED" in e for e in result.validation_errors):
                batch.failed += 1
            else:
                batch.invalid += 1

        # Rate limiting delay (only for non-cached results)
        if i < batch.total and delay_seconds > 0 and result.latency_ms > 0:
            await asyncio.sleep(delay_seconds)

    logger.info(
        "Batch complete: %d/%d successful, avg_score=%.2f, avg_latency=%.0fms, "
        "cache_hits=%d, cache_misses=%d, skipped=%d",
        batch.successful,
        batch.total,
        batch.avg_overall_score,
        batch.avg_latency_ms,
        batch.cache_hits,
        batch.cache_misses,
        batch.skipped,
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
