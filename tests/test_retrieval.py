"""
tests/test_retrieval.py

Unit tests for core retrieval components:
- VectorStore (ChromaDB wrapper)
- HybridSearcher (dense + BM25 + RRF)
- Reranker (cross-encoder mock)
- _meta_to_chunk helper
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from core.models import DocType, RetrievedChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_chunk(
    text: str = "Apple reported strong revenue growth.",
    doc_type: DocType = DocType.SEC_10K,
    ticker: str = "AAPL",
    date: datetime | None = None,
    dense_score: float = 0.9,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        doc_type=doc_type,
        ticker=ticker,
        date=date or datetime(2023, 9, 30),
        source_title="Apple 10-K 2023",
        page_number=12,
        dense_score=dense_score,
    )


# ---------------------------------------------------------------------------
# _meta_to_chunk
# ---------------------------------------------------------------------------

class TestMetaToChunk:
    def test_basic_parse(self):
        from core.retrieval.vectorstore import _meta_to_chunk

        meta = {
            "doc_type": "sec_10k",
            "ticker": "AAPL",
            "source_title": "Apple 10-K",
            "page_number": 5,
            "chunk_index": 3,
            "date": "2023-09-30T00:00:00",
            "source_url": "https://sec.gov/aapl.htm",
        }
        chunk = _meta_to_chunk("Some text", meta, dense_score=0.85)

        assert chunk.text == "Some text"
        assert chunk.doc_type == DocType.SEC_10K
        assert chunk.ticker == "AAPL"
        assert chunk.page_number == 5
        assert chunk.chunk_index == 3
        assert chunk.dense_score == pytest.approx(0.85)
        assert chunk.date is not None
        assert chunk.date.year == 2023

    def test_missing_date(self):
        from core.retrieval.vectorstore import _meta_to_chunk

        meta = {"doc_type": "news", "ticker": "AAPL"}
        chunk = _meta_to_chunk("text", meta)
        assert chunk.date is None

    def test_invalid_date_falls_back_to_none(self):
        from core.retrieval.vectorstore import _meta_to_chunk

        meta = {"doc_type": "news", "ticker": "AAPL", "date": "not-a-date"}
        chunk = _meta_to_chunk("text", meta)
        assert chunk.date is None

    def test_default_doc_type(self):
        from core.retrieval.vectorstore import _meta_to_chunk

        meta = {"ticker": "AAPL"}
        chunk = _meta_to_chunk("text", meta)
        assert chunk.doc_type == DocType.SEC_10K


# ---------------------------------------------------------------------------
# RetrievedChunk.citation_label
# ---------------------------------------------------------------------------

class TestCitationLabel:
    def test_label_format(self):
        chunk = make_chunk(ticker="AAPL", date=datetime(2023, 9, 30))
        label = chunk.citation_label
        assert "AAPL" in label
        assert "2023" in label
        assert "sec_10k" in label

    def test_label_without_date(self):
        chunk = RetrievedChunk(
            text="text", doc_type=DocType.NEWS, ticker="MSFT"
        )
        label = chunk.citation_label
        assert "MSFT" in label


# ---------------------------------------------------------------------------
# VectorStore (mocked ChromaDB)
# ---------------------------------------------------------------------------

class TestVectorStore:
    @patch("core.retrieval.vectorstore.get_chroma_client")
    def test_upsert_calls_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        from core.retrieval.vectorstore import VectorStore
        vs = VectorStore("sec_filings")
        vs.upsert(
            ids=["id1"],
            embeddings=[[0.1] * 1024],
            documents=["text"],
            metadatas=[{"ticker": "AAPL", "doc_type": "sec_10k"}],
        )

        mock_collection.upsert.assert_called_once()

    @patch("core.retrieval.vectorstore.get_chroma_client")
    def test_dense_search_returns_chunks(self, mock_get_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        # Simulate ChromaDB query response
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "documents": [["Apple faces competition in all markets."]],
            "metadatas": [[{"ticker": "AAPL", "doc_type": "sec_10k", "date": "2023-09-30"}]],
            "distances": [[0.1]],  # distance 0.1 → similarity 0.9
        }

        from core.retrieval.vectorstore import VectorStore
        vs = VectorStore("sec_filings")
        chunks = vs.dense_search(
            query_embedding=[0.1] * 1024,
            ticker="AAPL",
            top_k=5,
            doc_types=[DocType.SEC_10K],
        )

        assert len(chunks) == 1
        assert chunks[0].ticker == "AAPL"
        assert chunks[0].dense_score == pytest.approx(0.9)

    @patch("core.retrieval.vectorstore.get_chroma_client")
    def test_count(self, mock_get_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client

        from core.retrieval.vectorstore import VectorStore
        vs = VectorStore("sec_filings")
        assert vs.count() == 42


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class TestReranker:
    def test_reranker_passthrough_returns_top_k(self):
        """Without a Voyage key, reranker returns first top_k chunks (passthrough)."""
        from core.retrieval.reranker import Reranker

        with patch("core.retrieval.reranker.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                voyage_api_key=None,  # no key → passthrough
                retrieval_top_k_rerank=3,
            )
            reranker = Reranker()

        chunks = [make_chunk(text=f"chunk {i}") for i in range(5)]
        result = reranker.rerank("Apple risks", chunks, top_k=3)

        assert len(result) == 3

    def test_reranker_passthrough_fewer_chunks_than_top_k(self):
        """Passthrough with fewer chunks than top_k returns all chunks."""
        from core.retrieval.reranker import Reranker

        with patch("core.retrieval.reranker.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                voyage_api_key=None,
                retrieval_top_k_rerank=5,
            )
            reranker = Reranker()

        chunks = [make_chunk(text=f"chunk {i}") for i in range(2)]
        result = reranker.rerank("Apple risks", chunks, top_k=5)

        assert len(result) == 2

    def test_reranker_empty_input(self):
        """Empty input always returns empty list regardless of client."""
        from core.retrieval.reranker import Reranker

        with patch("core.retrieval.reranker.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                voyage_api_key=None,
                retrieval_top_k_rerank=5,
            )
            reranker = Reranker()

        result = reranker.rerank("Apple risks", [], top_k=5)
        assert result == []

    def test_reranker_voyage_rerank_called(self):
        """When Voyage client exists, _rerank_voyage is called."""
        from core.retrieval.reranker import Reranker

        mock_voyage_result = MagicMock()
        mock_voyage_result.results = [
            MagicMock(index=1, relevance_score=0.95),
            MagicMock(index=0, relevance_score=0.80),
        ]

        with patch("core.retrieval.reranker.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                voyage_api_key="voyage-test-key",
                retrieval_top_k_rerank=3,
            )
            with patch("voyageai.Client") as mock_voyage_cls:
                mock_voyage_client = MagicMock()
                mock_voyage_client.rerank.return_value = mock_voyage_result
                mock_voyage_cls.return_value = mock_voyage_client

                reranker = Reranker()
                chunks = [make_chunk(text=f"chunk {i}") for i in range(3)]
                result = reranker.rerank("Apple risks", chunks, top_k=2)

        assert len(result) == 2
        # chunk at index 1 should come first (highest score)
        assert result[0].rerank_score == pytest.approx(0.95)
