"""
core/retrieval/reranker.py

Cross-encoder reranking of retrieved chunks.
Uses Voyage AI's rerank endpoint if VOYAGE_API_KEY is set;
falls back to returning chunks as-is (ordered by RRF score).

Voyage rerank model: rerank-2
Input: query + list of document strings
Output: re-scored, re-ordered subset of the input
"""

from __future__ import annotations

import structlog

from core.config import get_settings
from core.models import RetrievedChunk

logger = structlog.get_logger()


class Reranker:
    """
    Reranks a list of RetrievedChunks against a query.

    Usage:
        reranker = Reranker()
        top_chunks = reranker.rerank(query_text, chunks, top_k=5)
    """

    VOYAGE_RERANK_MODEL = "rerank-2"

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = self._build_client()

    def _build_client(self):
        if self.settings.voyage_api_key:
            try:
                import voyageai
                client = voyageai.Client(api_key=self.settings.voyage_api_key)
                logger.info("reranker_init", provider="voyage")
                return client
            except ImportError:
                logger.warning("voyageai_not_installed", fallback="score_passthrough")
        logger.info("reranker_init", provider="passthrough")
        return None

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks by relevance to the query.
        Returns up to top_k chunks, ordered by rerank score descending.
        """
        top_k = top_k or self.settings.retrieval_top_k_rerank

        if not chunks:
            return []

        if self._client is None:
            # Passthrough: just truncate to top_k (already sorted by RRF)
            return chunks[:top_k]

        return self._rerank_voyage(query, chunks, top_k)

    def _rerank_voyage(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank using Voyage AI rerank-2 model."""
        documents = [chunk.text for chunk in chunks]
        try:
            result = self._client.rerank(
                query=query,
                documents=documents,
                model=self.VOYAGE_RERANK_MODEL,
                top_k=min(top_k, len(chunks)),
            )
            reranked = []
            for item in result.results:
                chunk = chunks[item.index]
                chunk.rerank_score = float(item.relevance_score)
                reranked.append(chunk)

            logger.debug(
                "rerank_done",
                n_input=len(chunks),
                n_output=len(reranked),
                top_score=reranked[0].rerank_score if reranked else None,
            )
            return reranked

        except Exception as exc:
            logger.error("rerank_error", error=str(exc), fallback="passthrough")
            return chunks[:top_k]
