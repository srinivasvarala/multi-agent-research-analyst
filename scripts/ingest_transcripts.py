"""
scripts/ingest_transcripts.py

Ingest earnings call transcript files into ChromaDB.
Usage: python scripts/ingest_transcripts.py --ticker AAPL --file transcript.txt --date 2024-01-30
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import get_settings
from core.models import DocType
from ingestion.pipeline import ingest_documents

logger = structlog.get_logger()
console = Console()
app = typer.Typer()

settings = get_settings()


@app.command()
def ingest(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Company ticker e.g. AAPL"),
    files: List[Path] = typer.Option(..., "--file", "-f", help="Path to transcript file (repeat for multiple)"),
    date: str = typer.Option(..., "--date", "-d", help="Date of the earnings call e.g. 2024-01-30"),
    data_dir: str = typer.Option("./data/raw", "--data-dir", help="Base data directory"),
) -> None:
    """Ingest earnings call transcript files into ChromaDB vector store."""

    ticker = ticker.upper()

    # Validate all files exist
    for f in files:
        if not f.exists():
            console.print(f"[red]File not found: {f}[/red]")
            raise typer.Exit(1)

    # Ensure vectorstore directory exists
    Path(settings.chroma_persist_path).mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Multi-Agent Research Analyst — Transcript Ingestion[/bold cyan]")
    console.print(f"Ticker: [bold]{ticker}[/bold] | Date: {date} | Files: {len(files)}\n")

    source_title = f"{ticker} Earnings Call {date}"
    total_chunks = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as p:
            for file_path in files:
                console.print(f"\n[bold]Processing:[/bold] {file_path.name}")

                progress_task = p.add_task("Embedding and storing...", total=100)

                def _progress(n_done: int, n_total: int) -> None:
                    pct = int(n_done / n_total * 100) if n_total else 0
                    p.update(progress_task, completed=pct,
                             description=f"Embedding and storing... {n_done}/{n_total} chunks")

                chunks_stored = ingest_documents(
                    paths=[file_path],
                    doc_type=DocType.EARNINGS_CALL,
                    ticker=ticker,
                    collection_name=settings.chroma_collection_earnings,
                    source_title_prefix=source_title,
                    date_str=date,
                    progress_callback=_progress,
                )
                p.update(progress_task, completed=100)
                total_chunks += chunks_stored
                console.print(f"  ✓ Stored [bold green]{chunks_stored}[/bold green] chunks")

    except Exception as e:
        logger.error("transcript_ingestion_error", error=str(e))
        console.print(f"[red]Ingestion error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]✓ Ingestion complete![/bold green]")
    console.print(f"  Total chunks stored: [bold]{total_chunks}[/bold]")
    console.print(f"  Collection: [dim]{settings.chroma_collection_earnings}[/dim]")
    console.print(f"  Store location: [dim]{settings.chroma_persist_path}[/dim]")
    console.print(f"\nNext step:")
    console.print(
        f'  [cyan]python -m agents.earnings_call_agent --ticker {ticker} '
        f'--query "What did management say about revenue growth?"[/cyan]\n'
    )


if __name__ == "__main__":
    app()
