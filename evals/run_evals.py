"""
evals/run_evals.py

CLI eval harness — runs the full pipeline against the eval dataset and
produces a scored report.

Usage:
    python evals/run_evals.py                          # all AAPL evals
    python evals/run_evals.py --ticker AAPL            # specific ticker
    python evals/run_evals.py --id aapl_001            # single eval item
    python evals/run_evals.py --output results.json    # save results
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from agents.orchestrator import OrchestratorAgent
from core.models import ResearchQuery, SynthesisResult
from evals.dataset import ALL_EVALS, get_evals_for_ticker, get_eval_by_id, EvalItem
from evals.metrics import score_eval_result, EvalResult

app = typer.Typer(help="Run the multi-agent research eval harness")
console = Console()
logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_for_eval(item: EvalItem) -> tuple[str, float, bool, SynthesisResult | None]:
    """
    Runs the full pipeline for one eval item.
    Returns (final_answer, final_confidence, passed_critic, synthesis).
    """
    rq = ResearchQuery(
        query=item.query,
        ticker=item.ticker,
        company_name=item.company_name,
    )
    orchestrator = OrchestratorAgent()

    try:
        report, memory = orchestrator.run_full_pipeline(rq)
        passed = report.critic_review.passed if report.critic_review else True
        return report.final_answer, report.final_confidence, passed, report.synthesis
    except Exception as exc:
        logger.error("eval_pipeline_error", eval_id=item.id, error=str(exc))
        return f"Pipeline error: {exc}", 0.0, False, None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_eval_result(result: EvalResult) -> None:
    score_color = (
        "green" if result.aggregate_score >= 0.7
        else "yellow" if result.aggregate_score >= 0.4
        else "red"
    )
    passed_label = "[green]PASS[/green]" if result.passed_critic else "[red]FAIL[/red]"

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_column("Notes", max_width=60)

    for metric in [
        result.citation_accuracy,
        result.answer_relevance,
        result.faithfulness,
        result.topic_coverage,
    ]:
        if metric:
            color = "green" if metric.score >= 0.7 else "yellow" if metric.score >= 0.4 else "red"
            table.add_row(
                metric.name.replace("_", " ").title(),
                f"[{color}]{metric.score:.2f}[/{color}]",
                metric.explanation[:80],
            )

    console.print(Panel(
        table,
        title=(
            f"[bold]{result.eval_id}[/bold] — "
            f"[{score_color}]Aggregate: {result.aggregate_score:.2f}[/{score_color}] | "
            f"Confidence: {result.final_confidence:.0%} | Critic: {passed_label}"
        ),
    ))


def print_summary(results: list[EvalResult]) -> None:
    if not results:
        return

    avg_aggregate = sum(r.aggregate_score for r in results) / len(results)
    avg_confidence = sum(r.final_confidence for r in results) / len(results)
    n_passed = sum(1 for r in results if r.passed_critic)

    def avg_metric(attr: str) -> str:
        vals = [getattr(r, attr).score for r in results if getattr(r, attr)]
        return f"{sum(vals)/len(vals):.2f}" if vals else "n/a"

    table = Table(title="Eval Summary", box=box.ROUNDED, show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Average", justify="right")

    table.add_row("Aggregate Score", f"{avg_aggregate:.2f}")
    table.add_row("Pipeline Confidence", f"{avg_confidence:.0%}")
    table.add_row("Critic Pass Rate", f"{n_passed}/{len(results)}")
    table.add_row("Citation Accuracy", avg_metric("citation_accuracy"))
    table.add_row("Answer Relevance", avg_metric("answer_relevance"))
    table.add_row("Faithfulness", avg_metric("faithfulness"))
    table.add_row("Topic Coverage", avg_metric("topic_coverage"))

    console.print(table)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    ticker: str = typer.Option("AAPL", help="Ticker to run evals for"),
    eval_id: str = typer.Option(None, "--id", help="Run a single eval item by ID"),
    output: Path = typer.Option(None, "--output", help="Save results to JSON file"),
    skip_pipeline: bool = typer.Option(
        False, "--skip-pipeline",
        help="Skip pipeline, use placeholder answers (for metric testing only)"
    ),
) -> None:
    """Run the eval harness against the research pipeline."""

    # Select eval items
    if eval_id:
        item = get_eval_by_id(eval_id)
        if not item:
            console.print(f"[red]Eval ID '{eval_id}' not found.[/red]")
            raise typer.Exit(1)
        items = [item]
    else:
        items = get_evals_for_ticker(ticker)
        if not items:
            console.print(f"[red]No eval items found for ticker '{ticker}'.[/red]")
            raise typer.Exit(1)

    console.print(f"\n[bold cyan]Running {len(items)} eval(s) for {ticker}[/bold cyan]\n")

    results: list[EvalResult] = []
    total_start = time.monotonic()

    for i, item in enumerate(items, 1):
        console.print(f"[bold]({i}/{len(items)}) {item.id}:[/bold] {item.query[:80]}")

        if skip_pipeline:
            final_answer = f"Placeholder answer for {item.id}"
            final_confidence = 0.5
            passed_critic = True
            synthesis = None
        else:
            pipeline_start = time.monotonic()
            final_answer, final_confidence, passed_critic, synthesis = run_pipeline_for_eval(item)
            pipeline_ms = int((time.monotonic() - pipeline_start) * 1000)
            console.print(f"  Pipeline done in {pipeline_ms:,}ms | confidence={final_confidence:.0%}")

        result = score_eval_result(
            eval_item=item,
            final_answer=final_answer,
            final_confidence=final_confidence,
            passed_critic=passed_critic,
            synthesis=synthesis,
        )
        results.append(result)
        print_eval_result(result)

    total_ms = int((time.monotonic() - total_start) * 1000)
    console.print(f"\n[dim]Total eval time: {total_ms/1000:.1f}s[/dim]\n")
    print_summary(results)

    # Save results
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps([r.to_dict() for r in results], indent=2))
        console.print(f"\n[green]Results saved to {output}[/green]")


if __name__ == "__main__":
    app()
