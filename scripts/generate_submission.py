#!/usr/bin/env python3
"""Generate the final submission.jsonl for the magicpin AI Challenge.

This script strictly follows the challenge evaluation requirements:
1. Loads the 30 specific test pairs (represented by the first 30 triggers).
2. Generates messages using the production compose() function.
3. Validates every response against the required schema.
4. Aborts immediately if any validation fails.
5. Writes the official submission.jsonl file.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset_loader import load_dataset, build_scenarios
from evaluation.batch_runner import run_batch


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    )
    logger = logging.getLogger("generate_submission")

    logger.info("Initializing submission generation pipeline...")

    # Load full dataset
    try:
        dataset = load_dataset()
        scenarios = build_scenarios(dataset)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # For the challenge, there is a canonical test set of 30 specific pairs.
    # In this mock environment, we assume the first 30 scenarios represent this set.
    # If there are fewer than 30, we use what's available.
    test_scenarios = scenarios[:30]
    
    if len(test_scenarios) < 30:
        logger.warning(f"Only found {len(test_scenarios)} scenarios, challenge requires 30.")

    logger.info(f"Generating submission for {len(test_scenarios)} test scenarios...")

    # Run batch composition
    batch_result = await run_batch(
        test_scenarios,
        prompt_version="final_submission",
        delay_seconds=2.5,
    )

    # Abort if any failures or invalid outputs
    if batch_result.failed > 0 or batch_result.invalid > 0:
        logger.error(
            f"CRITICAL ABORT: {batch_result.failed} failures and {batch_result.invalid} invalid schemas detected."
        )
        logger.error("Submission generation aborted to prevent malformed JSON.")
        sys.exit(1)

    logger.info("All outputs validated successfully. Writing submission.jsonl...")

    # Write submission.jsonl
    output_dir = Path(__file__).resolve().parent.parent
    output_file = output_dir / "submission.jsonl"
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in batch_result.results:
                line = json.dumps(result.to_submission_line(), ensure_ascii=False)
                f.write(line + "\n")
                
        logger.info(f"Successfully generated {output_file.name} with {len(batch_result.results)} lines.")
        logger.info(f"Final Average Quality Score: {batch_result.avg_overall_score:.2f} / 10.0")
        
    except Exception as e:
        logger.error(f"Failed to write submission file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
