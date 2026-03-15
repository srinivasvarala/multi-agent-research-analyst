"""
scripts/ingest_sec.py

Download SEC 10-K/10-Q filings from EDGAR and ingest into ChromaDB.
Usage: python scripts/ingest_sec.py --ticker AAPL --year 2023
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    year: int = typer.Option(2023, "--year", "-y", help="Filing year"),
    filing_type: str = typer.Option("10-K", "--type", help="Filing type: 10-K or 10-Q"),
    data_dir: str = typer.Option("./data/raw", "--data-dir", help="Where to save raw filings"),
    limit: int = typer.Option(1, "--limit", help="Max number of filings to download"),
) -> None:
    """Download SEC filings and ingest into ChromaDB vector store."""

    ticker = ticker.upper()
    raw_path = Path(data_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    # Ensure vectorstore directory exists
    Path(settings.chroma_persist_path).mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Multi-Agent Research Analyst — SEC Ingestion[/bold cyan]")
    console.print(f"Ticker: [bold]{ticker}[/bold] | Filing: {filing_type} | Year: {year}\n")

    # Step 1: Download filings from EDGAR
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as p:
        task = p.add_task(f"Downloading {filing_type} filings from SEC EDGAR...", total=None)
        filing_paths = _download_sec_filings(ticker, filing_type, year, raw_path, limit)
        p.update(task, description=f"Downloaded {len(filing_paths)} filing(s)")

    if not filing_paths:
        console.print(f"[yellow]No filings found for {ticker} {filing_type} {year}[/yellow]")
        console.print("[dim]Tip: Try a different year or check the ticker symbol[/dim]")
        raise typer.Exit(1)

    # Step 2: Ingest via pipeline (load → chunk → embed → store)
    doc_type = DocType.SEC_10K if filing_type == "10-K" else DocType.SEC_10Q
    source_title = f"{ticker} {filing_type} {year}"
    total_chunks = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as p:
        for filing_path in filing_paths:
            console.print(f"\n[bold]Processing:[/bold] {filing_path.name}")

            # Track progress per file
            progress_task = p.add_task("Embedding and storing...", total=100)

            def _progress(n_done: int, n_total: int) -> None:
                pct = int(n_done / n_total * 100) if n_total else 0
                p.update(progress_task, completed=pct,
                         description=f"Embedding and storing... {n_done}/{n_total} chunks")

            chunks_stored = ingest_documents(
                paths=[filing_path],
                doc_type=doc_type,
                ticker=ticker,
                collection_name=settings.chroma_collection_sec,
                source_title_prefix=source_title,
                date_str=f"{year}-01-01",
                progress_callback=_progress,
            )
            p.update(progress_task, completed=100)
            total_chunks += chunks_stored
            console.print(f"  ✓ Stored [bold green]{chunks_stored}[/bold green] chunks")

    console.print(f"\n[bold green]✓ Ingestion complete![/bold green]")
    console.print(f"  Total chunks stored: [bold]{total_chunks}[/bold]")
    console.print(f"  Collection: [dim]{settings.chroma_collection_sec}[/dim]")
    console.print(f"  Store location: [dim]{settings.chroma_persist_path}[/dim]")
    console.print(f"\nNext step:")
    console.print(
        f'  [cyan]python -m agents.sec_filings_agent --ticker {ticker} '
        f'--query "What are the main risk factors?"[/cyan]\n'
    )


def _download_sec_filings(
    ticker: str,
    filing_type: str,
    year: int,
    output_dir: Path,
    limit: int,
) -> list[Path]:
    """Download filings using sec-edgar-downloader."""
    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        console.print("[red]sec-edgar-downloader not installed. Run: pip install sec-edgar-downloader[/red]")
        raise typer.Exit(1)

    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(exist_ok=True)

    dl = Downloader(
        company_name="ResearchAnalyst",
        email_address="research@example.com",
        download_folder=str(ticker_dir),
    )

    try:
        dl.get(
            filing_type,
            ticker,
            limit=limit,
            after=f"{year}-01-01",
            before=f"{year}-12-31",
        )
    except Exception as e:
        logger.error("sec_download_error", error=str(e), ticker=ticker)
        console.print(f"[red]Download error: {e}[/red]")
        return []

    # For each filing subdirectory, try to extract primary HTM first
    filing_paths: list[Path] = []

    # Collect all filing subdirectories (one per accession number)
    filing_dirs = [d for d in ticker_dir.rglob("*") if d.is_dir()]
    filing_dirs = filing_dirs if filing_dirs else [ticker_dir]

    for filing_dir in filing_dirs:
        # Attempt SGML extraction → produces primary_10k.htm
        extracted = _extract_primary_document(filing_dir, filing_type)
        if extracted:
            filing_paths.append(extracted)
            continue
        # Fallback: prefer PDF > HTM > HTML > TXT
        for ext in ["*.pdf", "*.htm", "*.html", "*.txt"]:
            matches = list(filing_dir.glob(ext))
            if matches:
                filing_paths.extend(matches)
                break

    return sorted(set(filing_paths))[:limit]


def _extract_primary_document(filing_dir: Path, filing_type: str) -> Path | None:
    """
    Parse EDGAR full-submission.txt and write the primary filing document
    (e.g. the 10-K HTML) to primary_10k.htm in the same directory.
    Returns the Path to the extracted file, or None if extraction fails.
    """
    import re

    submission_file = filing_dir / "full-submission.txt"
    if not submission_file.exists():
        return None

    output_path = filing_dir / "primary_10k.htm"
    if output_path.exists():
        logger.info("primary_doc_already_extracted", path=str(output_path))
        return output_path

    raw = submission_file.read_text(encoding="utf-8", errors="ignore")

    # Find the first <DOCUMENT> block whose <TYPE> matches the filing type
    doc_blocks = re.split(r"<DOCUMENT>", raw, flags=re.IGNORECASE)

    for block in doc_blocks[1:]:  # skip preamble before first <DOCUMENT>
        type_match = re.search(r"<TYPE>([^\n\r]+)", block, re.IGNORECASE)
        if not type_match:
            continue
        doc_type = type_match.group(1).strip()
        # Match exact type (10-K) or amended (10-K/A) — ignore exhibits
        if doc_type.upper().startswith(filing_type.upper()):
            text_match = re.search(
                r"<TEXT>(.*?)</TEXT>", block, re.IGNORECASE | re.DOTALL
            )
            if text_match:
                content = text_match.group(1).strip()
                output_path.write_text(content, encoding="utf-8")
                logger.info(
                    "primary_doc_extracted",
                    source=str(submission_file),
                    output=str(output_path),
                    size_kb=len(content) // 1024,
                )
                return output_path

    logger.warning("primary_doc_not_found", submission=str(submission_file))
    return None


if __name__ == "__main__":
    app()
