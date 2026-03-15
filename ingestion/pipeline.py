"""
ingestion/pipeline.py

Orchestrates the full ingestion run for any document corpus:
  load → chunk → embed → upsert into ChromaDB

Used by the ingest_* scripts to avoid duplicating logic.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from core.config import get_settings
from core.models import DocType
from core.retrieval.vectorstore import VectorStore
from ingestion.chunker import chunk_text
from ingestion.embedder import Embedder
from ingestion.loaders import load_document

logger = structlog.get_logger()

BATCH_SIZE = 20  # Chunks per embedding API call (smaller = fewer TPM spikes)


def ingest_documents(
    paths: list[Path],
    doc_type: DocType,
    ticker: str,
    collection_name: str,
    source_title_prefix: str = "",
    date_str: str | None = None,
    batch_size: int = BATCH_SIZE,
    progress_callback=None,
) -> int:
    """
    Full ingestion pipeline for a list of document paths.

    Args:
        paths: List of file paths (PDF, HTML, TXT)
        doc_type: DocType enum value for all documents
        ticker: Company ticker symbol
        collection_name: ChromaDB collection to write to
        source_title_prefix: Prefix for source_title metadata (e.g. "AAPL 10-K 2023")
        date_str: ISO date string to attach to chunks (e.g. "2023-01-01")
        batch_size: How many chunks to embed in one API call
        progress_callback: Optional callable(n_done, n_total) for progress reporting

    Returns:
        Total number of chunks stored
    """
    settings = get_settings()
    embedder = Embedder()
    store = VectorStore(collection_name)
    total_chunks = 0

    for path in paths:
        logger.info("ingesting_document", path=str(path), doc_type=doc_type.value)

        # 1. Load: returns list of (page_number, text)
        pages = load_document(path)
        if not pages:
            logger.warning("empty_document", path=str(path))
            continue

        # 2. Chunk each page
        all_chunks: list[dict] = []
        for page_num, page_text in pages:
            if len(page_text.strip()) < 50:
                continue  # Skip near-empty pages
            page_chunks = chunk_text(
                text=page_text,
                doc_type=doc_type,
                ticker=ticker,
                source_title=f"{source_title_prefix} p.{page_num}".strip() if source_title_prefix else None,
                source_url=str(path),
                page_number=page_num,
                date_str=date_str,
            )
            all_chunks.extend(page_chunks)

        if not all_chunks:
            logger.warning("no_chunks_produced", path=str(path))
            continue

        logger.info("chunks_created", n=len(all_chunks), path=str(path))

        # 3. Embed + store in batches
        n_total = len(all_chunks)
        n_done = 0
        for i in range(0, n_total, batch_size):
            batch = all_chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = embedder.embed(texts)
            store.upsert(
                ids=[c["id"] for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c["metadata"] for c in batch],
            )
            n_done += len(batch)
            if progress_callback:
                progress_callback(n_done, n_total)

        total_chunks += n_total
        logger.info("document_ingested", path=str(path), chunks=n_total)

    logger.info(
        "ingestion_complete",
        ticker=ticker,
        doc_type=doc_type.value,
        total_chunks=total_chunks,
    )
    return total_chunks
