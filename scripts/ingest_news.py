"""
scripts/ingest_news.py

Ingest news articles into ChromaDB — either from NewsAPI or local files.
Usage:
  python scripts/ingest_news.py --ticker AAPL --days 30          # NewsAPI
  python scripts/ingest_news.py --ticker AAPL --file article.txt  # local file
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

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
    days: int = typer.Option(30, "--days", help="Number of days back to fetch from NewsAPI"),
    files: Optional[List[Path]] = typer.Option(None, "--file", "-f", help="Local article file(s) to ingest"),
    search_term: Optional[str] = typer.Option(None, "--search-term", help="Search term for NewsAPI (defaults to ticker)"),
    data_dir: str = typer.Option("./data/raw", "--data-dir", help="Base data directory"),
) -> None:
    """Ingest news articles into ChromaDB — from NewsAPI or local files."""

    ticker = ticker.upper()

    # Ensure vectorstore directory exists
    Path(settings.chroma_persist_path).mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Multi-Agent Research Analyst — News Ingestion[/bold cyan]")
    console.print(f"Ticker: [bold]{ticker}[/bold]\n")

    # Collect paths to ingest
    paths_to_ingest: list[Path] = []

    # Local files take precedence / supplement NewsAPI
    if files:
        for f in files:
            if not f.exists():
                console.print(f"[red]File not found: {f}[/red]")
                raise typer.Exit(1)
        paths_to_ingest.extend(files)
        console.print(f"  Using {len(files)} local file(s)")

    # Fetch from NewsAPI if no files provided
    if not files:
        api_key = settings.news_api_key
        if not api_key:
            console.print(
                "[yellow]NEWS_API_KEY is not set and no --file was provided.[/yellow]\n"
                "[dim]To ingest news you can either:\n"
                "  1. Set NEWS_API_KEY in your .env file (get a free key at https://newsapi.org)\n"
                "  2. Pass local article files with --file article.txt[/dim]"
            )
            raise typer.Exit(0)

        raw_path = Path(data_dir) / "news" / ticker
        raw_path.mkdir(parents=True, exist_ok=True)

        term = search_term or ticker
        console.print(f"  Fetching up to 100 articles for '{term}' (last {days} days)...")

        fetched = _fetch_newsapi(ticker, term, days, raw_path, api_key)
        if not fetched:
            console.print(f"[yellow]No articles fetched from NewsAPI for '{term}'[/yellow]")
            raise typer.Exit(0)

        paths_to_ingest.extend(fetched)
        console.print(f"  Fetched {len(fetched)} article(s) from NewsAPI")

    if not paths_to_ingest:
        console.print("[yellow]No documents to ingest.[/yellow]")
        raise typer.Exit(0)

    # Ingest all collected paths
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    source_title = f"{ticker} News {today_str}"
    total_chunks = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as p:
            for file_path in paths_to_ingest:
                console.print(f"\n[bold]Processing:[/bold] {file_path.name}")

                progress_task = p.add_task("Embedding and storing...", total=100)

                def _progress(n_done: int, n_total: int) -> None:
                    pct = int(n_done / n_total * 100) if n_total else 0
                    p.update(progress_task, completed=pct,
                             description=f"Embedding and storing... {n_done}/{n_total} chunks")

                chunks_stored = ingest_documents(
                    paths=[file_path],
                    doc_type=DocType.NEWS,
                    ticker=ticker,
                    collection_name=settings.chroma_collection_news,
                    source_title_prefix=source_title,
                    date_str=today_str,
                    progress_callback=_progress,
                )
                p.update(progress_task, completed=100)
                total_chunks += chunks_stored
                console.print(f"  ✓ Stored [bold green]{chunks_stored}[/bold green] chunks")

    except Exception as e:
        logger.error("news_ingestion_error", error=str(e))
        console.print(f"[red]Ingestion error: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold green]✓ Ingestion complete![/bold green]")
    console.print(f"  Total chunks stored: [bold]{total_chunks}[/bold]")
    console.print(f"  Collection: [dim]{settings.chroma_collection_news}[/dim]")
    console.print(f"  Store location: [dim]{settings.chroma_persist_path}[/dim]")
    console.print(f"\nNext step:")
    console.print(
        f'  [cyan]python -m agents.news_agent --ticker {ticker} '
        f'--query "What is the latest news about {ticker}?"[/cyan]\n'
    )


def _fetch_newsapi(
    ticker: str,
    search_term: str,
    days: int,
    data_dir: Path,
    api_key: str,
) -> list[Path]:
    """
    Fetch articles from NewsAPI and write each to a text file.
    Returns list of Paths to written files.
    """
    try:
        import httpx
    except ImportError:
        console.print("[red]httpx not installed. Run: pip install httpx[/red]")
        raise typer.Exit(1)

    date_from = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    date_to = datetime.utcnow().strftime("%Y-%m-%d")

    params = {
        "q": search_term,
        "from": date_from,
        "to": date_to,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }

    written_paths: list[Path] = []

    try:
        response = httpx.get("https://newsapi.org/v2/everything", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as e:
        logger.error("newsapi_fetch_error", error=str(e), search_term=search_term)
        console.print(f"[yellow]NewsAPI request failed: {e}[/yellow]")
        return written_paths

    articles = data.get("articles", [])
    logger.info("newsapi_fetched", n_articles=len(articles), search_term=search_term)

    for article in articles:
        pub_date = article.get("publishedAt", date_to)[:10]  # YYYY-MM-DD
        source_name = (article.get("source", {}).get("name") or "unknown").replace(" ", "_")
        # Sanitize filename
        safe_source = "".join(c for c in source_name if c.isalnum() or c in "_-")[:40]

        title = article.get("title") or ""
        description = article.get("description") or ""
        content = article.get("content") or ""
        url = article.get("url") or ""

        text = f"Title: {title}\nSource: {source_name}\nDate: {pub_date}\nURL: {url}\n\n{description}\n\n{content}"

        filename = f"{pub_date}_{safe_source}.txt"
        file_path = data_dir / filename

        file_path.write_text(text, encoding="utf-8")
        written_paths.append(file_path)

    return written_paths


if __name__ == "__main__":
    app()
