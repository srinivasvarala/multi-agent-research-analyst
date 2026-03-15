"""
core/retrieval/hybrid_search.py

Hybrid retrieval: dense (ChromaDB cosine) + sparse (BM25) → RRF fusion.

Reciprocal Rank Fusion (RRF) formula:
    score(d) = Σ 1 / (k + rank(d))
where k=60 is the standard constant.
"""

from __future__ import annotations

import hashlib
import structlog
from rank_bm25 import BM25Okapi

from core.config import get_settings
from core.models import DocType, RetrievedChunk
from core.retrieval.vectorstore import VectorStore, _meta_to_chunk

logger = structlog.get_logger()
settings = get_settings()

RRF_K = 60  # Standard RRF constant

# Module-level BM25 cache keyed by (collection_name, ticker).
# Value is (n_docs_when_built, bm25_instance).
_bm25_cache: dict[tuple[str, str], tuple[int, BM25Okapi]] = {}


class HybridSearcher:
    """
    Combines dense vector search and BM25 keyword search via RRF.

    Usage:
        searcher = HybridSearcher(collection_name="sec_filings")
        chunks = searcher.search(query_text, query_embedding, ticker="AAPL")
    """

    def __init__(self, collection_name: str) -> None:
        self.store = VectorStore(collection_name)
        self.log = logger.bind(collection=collection_name)

    def search(
        self,
        query_text: str,
        query_embedding: list[float],
        ticker: str,
        doc_types: list[DocType] | None = None,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        top_k_final: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        1. Dense search (cosine similarity via ChromaDB)
        2. BM25 search over all docs for the ticker
        3. RRF fusion
        4. Return top_k_final chunks
        """
        top_k_dense = top_k_dense or settings.retrieval_top_k_dense
        top_k_sparse = top_k_sparse or settings.retrieval_top_k_sparse
        top_k_final = top_k_final or settings.retrieval_top_k_rrf

        # --- Dense ---
        dense_chunks = self.store.dense_search(
            query_embedding=query_embedding,
            ticker=ticker,
            top_k=top_k_dense,
            doc_types=doc_types,
        )
        self.log.debug("dense_search_done", n=len(dense_chunks))

        # --- Sparse (BM25) ---
        all_texts, all_metas = self.store.get_all_texts_for_bm25(ticker)
        if not all_texts:
            self.log.warning("no_documents_for_bm25", ticker=ticker)
            return dense_chunks[:top_k_final]

        sparse_chunks = self._bm25_search(
            query_text=query_text,
            all_texts=all_texts,
            all_metas=all_metas,
            doc_types=doc_types,
            top_k=top_k_sparse,
            ticker=ticker,
        )
        self.log.debug("sparse_search_done", n=len(sparse_chunks))

        # --- RRF Fusion ---
        fused = self._rrf_fusion(dense_chunks, sparse_chunks, top_k=top_k_final)
        self.log.info("hybrid_search_done", n=len(fused), ticker=ticker)
        return fused

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    def _bm25_search(
        self,
        query_text: str,
        all_texts: list[str],
        all_metas: list[dict],
        doc_types: list[DocType] | None,
        top_k: int,
        ticker: str = "",
    ) -> list[RetrievedChunk]:
        # Filter by doc_type if requested
        if doc_types:
            dt_values = {dt.value for dt in doc_types}
            filtered = [
                (t, m) for t, m in zip(all_texts, all_metas)
                if m.get("doc_type") in dt_values
            ]
        else:
            filtered = list(zip(all_texts, all_metas))

        if not filtered:
            return []

        texts, metas = zip(*filtered)

        # BM25 cache: rebuild only when corpus size changes
        cache_key = (self.store.collection.name, ticker)
        n_docs = len(texts)
        cached = _bm25_cache.get(cache_key)
        if cached is not None and cached[0] == n_docs:
            bm25 = cached[1]
            self.log.debug("bm25_cache_hit", ticker=ticker, n_docs=n_docs)
        else:
            # Tokenize (simple whitespace tokenizer — good enough for BM25)
            tokenized_corpus = [t.lower().split() for t in texts]
            bm25 = BM25Okapi(tokenized_corpus)
            _bm25_cache[cache_key] = (n_docs, bm25)
            self.log.debug("bm25_cache_miss", ticker=ticker, n_docs=n_docs)

        tokenized_query = query_text.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Get top_k indices by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        chunks = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = _meta_to_chunk(texts[idx], metas[idx])
            chunk.sparse_score = float(scores[idx])
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------
    # RRF Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(
        self,
        dense_chunks: list[RetrievedChunk],
        sparse_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Fuse two ranked lists using Reciprocal Rank Fusion.
        Chunks are identified by their text (first 100 chars as key).
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        def _key(chunk: RetrievedChunk) -> str:
            return hashlib.sha256(chunk.text.encode()).hexdigest()

        # Score from dense ranking
        for rank, chunk in enumerate(dense_chunks):
            k = _key(chunk)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (RRF_K + rank + 1)
            chunk_map[k] = chunk

        # Score from sparse ranking
        for rank, chunk in enumerate(sparse_chunks):
            k = _key(chunk)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (RRF_K + rank + 1)
            if k not in chunk_map:
                chunk_map[k] = chunk
            else:
                # Enrich existing chunk with sparse score
                chunk_map[k].sparse_score = chunk.sparse_score

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        result = []
        for key in sorted_keys[:top_k]:
            chunk = chunk_map[key]
            chunk.rrf_score = rrf_scores[key]
            result.append(chunk)

        return result
