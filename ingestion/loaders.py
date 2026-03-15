"""
ingestion/loaders.py

Document loaders for PDFs, HTML filings, and plain text.
Each loader returns a list of (page_number, text) tuples.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger()


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """
    Extract text page-by-page from a PDF using PyMuPDF.
    Returns list of (1-indexed page_number, page_text).
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((i, text))
        doc.close()
        logger.info("pdf_loaded", path=str(path), n_pages=len(pages))
        return pages
    except Exception as exc:
        logger.error("pdf_load_error", path=str(path), error=str(exc))
        raise


def load_html(path: Path) -> list[tuple[int, str]]:
    """
    Render an HTML filing through PyMuPDF and extract text page-by-page.
    Returns list of (1-indexed page_number, page_text).
    """
    try:
        import fitz  # PyMuPDF — already a project dependency
        doc = fitz.open(str(path), filetype="html")
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((i, text))
        doc.close()
        logger.info("html_loaded", path=str(path), n_pages=len(pages))
        return pages
    except Exception as exc:
        logger.error("html_load_error", path=str(path), error=str(exc))
        raise


def load_document(path: Path) -> list[tuple[int, str]]:
    """
    Auto-detect file type and return (page_number, text) pairs.
    Supports: .pdf, .htm, .html, .txt
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in {".htm", ".html"}:
        return load_html(path)
    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [(1, text)]
    else:
        # Best-effort: try PDF, then HTML
        try:
            return load_pdf(path)
        except Exception:
            return load_html(path)
