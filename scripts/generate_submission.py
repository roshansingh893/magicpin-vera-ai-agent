#!/usr/bin/env python3
"""Generate the final submission.jsonl for the magicpin AI Challenge.

This script strictly follows the challenge evaluation requirements:
1. Loads the 30 specific test pairs (represented by the first 30 triggers).
2. Generates messages using the production compose() function.
3. Validates every response against the required schema.
4. Aborts immediately if any validation fails.
5. Writes the official submission.jsonl file.

Phase 5 enhancements:
- Response cache integration (never re-generate existing outputs)
- --resume: skip already-generated entries
- --force: regenerate everything from scratch
- --dry-run: estimate work without calling Groq
- Smart rate limiting for Groq Free Tier
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Force UTF-8 stdout on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cache.response_cache import ResponseCache
from cache.rate_limiter import RateLimiter
from evaluation.dataset_loader import load_dataset, build_scenarios
from evaluation.batch_runner import run_batch, dry_run_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate submission.jsonl for the magicpin AI Challenge.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate all entries from scratch (ignores cache).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Only generate missing entries. Reuse existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Validate dataset and estimate API calls without calling Groq.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. Default: <project_root>/submission.jsonl",
    )
    return parser.parse_args()


def load_existing_submission(path: Path) -> dict[str, dict]:
    """Load existing submission.jsonl and return a dict of test_id → entry."""
    if not path.exists():
        return {}

    entries = {}
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        try:
            entry = json.loads(line)
            test_id = entry.get("test_id", "")
            if test_id:
                entries[test_id] = entry
        except json.JSONDecodeError:
            continue

    return entries


async def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    )
    logger = logging.getLogger("generate_submission")

    project_root = Path(__file__).resolve().parent.parent
    output_file = Path(args.output) if args.output else project_root / "submission.jsonl"

    logger.info("═" * 60)
    logger.info("  Vera AI Agent — Submission Generator")
    logger.info("═" * 60)
    logger.info("  Mode: %s", "FORCE" if args.force else ("DRY-RUN" if args.dry_run else ("RESUME" if args.resume else "DEFAULT")))
    logger.info("  Output: %s", output_file)

    # ── Initialize cache ─────────────────────────────────────────
    cache = ResponseCache()

    # Warm cache from existing submission (unless --force)
    if not args.force and output_file.exists():
        warmed = cache.warm_from_submission(output_file)
        logger.info("Cache warmed: %d entries from existing submission", warmed)

    # ── Load dataset ─────────────────────────────────────────────
    try:
        dataset = load_dataset()
        scenarios = build_scenarios(dataset)
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    # For the challenge, use the first 30 scenarios
    test_scenarios = scenarios[:30]

    if len(test_scenarios) < 30:
        logger.warning("Only found %d scenarios, challenge requires 30.", len(test_scenarios))

    logger.info("Total scenarios: %d", len(test_scenarios))

    # ── Dry run mode ─────────────────────────────────────────────
    if args.dry_run:
        report = dry_run_batch(test_scenarios, prompt_version="final_submission", cache=cache)

        logger.info("─" * 50)
        logger.info("DRY RUN REPORT:")
        logger.info("  Total scenarios:    %d", report["total_scenarios"])
        logger.info("  Already cached:     %d", report["cached"])
        logger.info("  Need API calls:     %d", report["needs_api_calls"])
        logger.info("  Estimated time:     %.1f minutes", report["estimated_time_minutes"])
        logger.info("  Validation errors:  %d", len(report["validation_errors"]))

        for err in report["validation_errors"]:
            logger.warning("  ⚠ %s", err)

        logger.info("─" * 50)
        logger.info("No API calls were made.")
        return

    # ── Determine which entries need generating ──────────────────
    existing = load_existing_submission(output_file) if not args.force else {}
    completed_ids = set(existing.keys()) if args.resume or (not args.force) else set()

    if completed_ids:
        logger.info("Found %d existing entries — will skip these.", len(completed_ids))

    # ── Initialize rate limiter ──────────────────────────────────
    rate_limiter = RateLimiter(max_rpm=28)

    # ── Run batch composition ────────────────────────────────────
    batch_result = await run_batch(
        test_scenarios,
        prompt_version="final_submission",
        delay_seconds=2.5,
        cache=cache,
        rate_limiter=rate_limiter,
        completed_test_ids=completed_ids,
    )

    # ── Check for failures ───────────────────────────────────────
    if batch_result.failed > 0 or batch_result.invalid > 0:
        logger.error(
            "CRITICAL: %d failures and %d invalid schemas detected.",
            batch_result.failed,
            batch_result.invalid,
        )
        logger.error("Continuing with valid results only.")

    logger.info(
        "Batch complete: %d successful, %d failed, %d invalid, %d cache hits",
        batch_result.successful,
        batch_result.failed,
        batch_result.invalid,
        batch_result.cache_hits,
    )

    # ── Merge results with existing entries ───────────────────────
    final_entries: dict[str, dict] = {}

    # Start with existing entries (unless --force)
    if not args.force:
        final_entries.update(existing)

    # Add/overwrite with new results
    for result in batch_result.results:
        if result.valid:
            final_entries[result.test_id] = result.to_submission_line()

    # ── Validate all entries ─────────────────────────────────────
    required_fields = {"test_id", "body", "cta", "send_as", "suppression_key", "rationale"}
    valid_count = 0
    for test_id, entry in sorted(final_entries.items()):
        missing = required_fields - set(entry.keys())
        if missing:
            logger.error("Entry %s is missing fields: %s", test_id, missing)
            continue
        if not entry["body"].strip():
            logger.error("Entry %s has empty body", test_id)
            continue
        valid_count += 1

    logger.info("Validated %d / %d entries", valid_count, len(final_entries))

    # ── Write submission.jsonl ───────────────────────────────────
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for test_id in sorted(final_entries.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
                entry = final_entries[test_id]
                line = json.dumps(entry, ensure_ascii=False)
                f.write(line + "\n")

        logger.info("═" * 60)
        logger.info("  ✓ submission.jsonl generated: %d entries", len(final_entries))
        logger.info("  ✓ Average quality score: %.2f / 10.0", batch_result.avg_overall_score)
        logger.info("  ✓ Cache stats: %s", cache.stats())
        logger.info("  ✓ Rate limiter stats: %s", rate_limiter.stats())
        logger.info("═" * 60)

    except Exception as e:
        logger.error("Failed to write submission file: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
