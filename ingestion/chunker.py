"""
ingestion/chunker.py

Token-aware chunking with overlap.
Uses tiktoken for accurate token counting (same tokenizer Claude uses).
"""

from __future__ import annotations

import structlog
import tiktoken

from core.config import get_settings
from core.models import DocType

logger = structlog.get_logger()
settings = get_settings()

# Use cl100k_base — compatible with most modern LLMs
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    doc_type: DocType,
    ticker: str,
    source_url: str | None = None,
    source_title: str | None = None,
    page_number: int | None = None,
    date_str: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """
    Split text into overlapping token-aware chunks.
    Returns list of dicts ready for vectorstore upsert.

    Each dict contains:
        id, text, metadata (ticker, doc_type, source_url, etc.)
    """
    chunk_size = chunk_size or settings.chunk_size_tokens
    overlap = overlap or settings.chunk_overlap_tokens

    tokens = _TOKENIZER.encode(text)
    total_tokens = len(tokens)

    if total_tokens == 0:
        return []

    chunks = []
    start = 0
    chunk_idx = 0

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text_str = _TOKENIZER.decode(chunk_tokens)

        # Build a stable ID for deduplication
        import hashlib
        chunk_id = hashlib.md5(
            f"{ticker}:{doc_type.value}:{source_url}:{page_number}:{chunk_idx}".encode()
        ).hexdigest()

        metadata = {
            "ticker": ticker,
            "doc_type": doc_type.value,
            "chunk_index": chunk_idx,
            "chunk_size_tokens": len(chunk_tokens),
        }
        if source_url:
            metadata["source_url"] = source_url
        if source_title:
            metadata["source_title"] = source_title
        if page_number is not None:
            metadata["page_number"] = page_number
        if date_str:
            metadata["date"] = date_str

        chunks.append({
            "id": chunk_id,
            "text": chunk_text_str,
            "metadata": metadata,
        })

        chunk_idx += 1
        # Advance by (chunk_size - overlap) to create overlapping windows
        start += chunk_size - overlap

    logger.debug(
        "chunked_document",
        ticker=ticker,
        doc_type=doc_type.value,
        total_tokens=total_tokens,
        n_chunks=len(chunks),
    )
    return chunks
