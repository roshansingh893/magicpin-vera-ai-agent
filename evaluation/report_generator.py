"""Markdown report generator for evaluation batches.

Generates human-readable evaluation reports (like evaluation_report.md)
with summary statistics, top/worst responses, and recommendations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

from evaluation.batch_runner import BatchResult

logger = logging.getLogger(__name__)


def generate_markdown_report(batch: BatchResult, title: str = "Evaluation Report") -> str:
    """Generate a Markdown report from a batch result.

    Args:
        batch: The evaluated batch result.
        title: Title of the report.

    Returns:
        The markdown content as a string.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine weakest metric
    metrics = {
        "Specificity": batch.avg_metric("specificity"),
        "Merchant Fit": batch.avg_metric("merchant_fit"),
        "Category Fit": batch.avg_metric("category_fit"),
        "Trigger Relevance": batch.avg_metric("trigger_relevance"),
        "Engagement": batch.avg_metric("engagement"),
    }
    weakest_metric = min(metrics.items(), key=lambda x: x[1])

    # Simple heuristic recommendations
    recommendations = {
        "Specificity": "Add more explicit numbers, dates, and names from the context payload.",
        "Merchant Fit": "Reference the merchant's location, performance stats, or active offers.",
        "Category Fit": "Ensure vocabulary strictly aligns with the category voice guidelines.",
        "Trigger Relevance": "Explicitly state why this message is being sent right now.",
        "Engagement": "Improve CTAs, ask questions, or use social proof to compel replies.",
    }

    report = [
        f"# {title}",
        f"**Generated:** {now}",
        f"**Prompt Version:** `{batch.prompt_version}`",
        f"**Latency:** {batch.avg_latency_ms:.1f}ms/call",
        "",
        "## Evaluation Summary",
        "",
        "| Metric | Average Score |",
        "|---|---|",
        f"| **Overall Score** | **{batch.avg_overall_score:.2f}** / 10 |",
        f"| Specificity | {metrics['Specificity']:.2f} |",
        f"| Merchant Fit | {metrics['Merchant Fit']:.2f} |",
        f"| Category Fit | {metrics['Category Fit']:.2f} |",
        f"| Trigger Relevance | {metrics['Trigger Relevance']:.2f} |",
        f"| Engagement | {metrics['Engagement']:.2f} |",
        "",
        f"**Weakest Metric:** {weakest_metric[0]} ({weakest_metric[1]:.2f})",
        "",
        "**Recommendation:**",
        f"> {recommendations.get(weakest_metric[0], 'Review prompt.')}",
        "",
        "## Execution Stats",
        f"- **Total Scenarios:** {batch.total}",
        f"- **Successful:** {batch.successful}",
        f"- **Failed:** {batch.failed}",
        f"- **Invalid JSON/Schema:** {batch.invalid}",
        f"- **Failure Rate:** {batch.failure_rate:.2%}",
        "",
        "---",
        "",
        "## Top 3 Responses",
    ]

    # Top results
    for i, res in enumerate(batch.top_results(3), start=1):
        report.extend([
            f"### {i}. Test `{res.test_id}` — Score: **{res.scores.overall:.2f}**",
            f"*Merchant: {res.merchant_id} | Trigger: {res.trigger_kind}*",
            "",
            "```text",
            res.body,
            "```",
            f"- **CTA:** `{res.cta}`",
            f"- **Rationale:** {res.rationale}",
            "",
        ])

    report.extend([
        "---",
        "",
        "## Worst 3 Responses",
    ])

    # Worst results
    for i, res in enumerate(batch.worst_results(3), start=1):
        report.extend([
            f"### {i}. Test `{res.test_id}` — Score: **{res.scores.overall:.2f}**",
            f"*Merchant: {res.merchant_id} | Trigger: {res.trigger_kind}*",
            "",
            "```text",
            res.body,
            "```",
            f"- **CTA:** `{res.cta}`",
            f"- **Rationale:** {res.rationale}",
            "",
        ])

    return "\n".join(report)


def write_report(batch: BatchResult, output_dir: Path, filename: str = "evaluation_report.md") -> Path:
    """Generate and save the markdown report to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename
    
    content = generate_markdown_report(batch)
    report_path.write_text(content, encoding="utf-8")
    
    logger.info("Evaluation report saved to %s", report_path)
    return report_path
