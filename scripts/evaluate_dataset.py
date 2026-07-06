#!/usr/bin/env python3
"""Run evaluation on the full dataset and generate a markdown report.

This script executes the offline evaluation pipeline without affecting
the production API. It loads the dataset, builds scenarios, runs
batch composition, evaluates quality, and writes the report to disk.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to Python path so we can run this from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset_loader import load_dataset, build_scenarios
from evaluation.batch_runner import run_batch, save_batch_results
from evaluation.report_generator import write_report


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("evaluate_dataset")

    logger.info("Initializing evaluation pipeline...")

    # Load full dataset
    try:
        dataset = load_dataset()
        scenarios = build_scenarios(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    if not scenarios:
        logger.error("No scenarios found to evaluate. Exiting.")
        sys.exit(1)

    # We will test on a small subset by default unless --full is passed
    # Just for demonstration, let's take the first 5 if not specified.
    # Note: In a real run we might parse argparse for --full.
    test_scenarios = scenarios[:5]
    if "--full" in sys.argv:
        test_scenarios = scenarios

    logger.info(f"Running evaluation on {len(test_scenarios)} scenarios...")

    # Run batch composition and evaluation
    batch_result = await run_batch(
        test_scenarios,
        prompt_version="v2_phase3_5",
        delay_seconds=1.5,  # Avoid hitting LLM rate limits
    )

    # Output paths
    output_dir = Path(__file__).resolve().parent.parent / "reports" / "generated"
    
    # Save raw JSON results
    save_batch_results(batch_result, output_dir)
    
    # Generate human-readable Markdown report
    report_path = write_report(batch_result, output_dir)
    
    logger.info("Evaluation complete!")
    logger.info(f"Summary: Score: {batch_result.avg_overall_score:.2f} | Failures: {batch_result.failed}")
    logger.info(f"Report available at: {report_path}")

    if batch_result.failed > 0 or batch_result.invalid > 0:
        logger.warning(f"Warning: {batch_result.failed} failed and {batch_result.invalid} invalid outputs.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
