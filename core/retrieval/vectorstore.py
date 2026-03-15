"""
core/retrieval/vectorstore.py

Direct ChromaDB wrapper — one collection per corpus.
No LangChain. Explicit is better than magic.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.config import get_settings
from core.models import DocType, RetrievedChunk

_settings = get_settings()


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=_settings.chroma_persist_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


COLLECTION_MAP: dict[DocType, str] = {
    DocType.SEC_10K: _settings.chroma_collection_sec,
    DocType.SEC_10Q: _settings.chroma_collection_sec,
    DocType.EARNINGS_CALL: _settings.chroma_collection_earnings,
    DocType.NEWS: _settings.chroma_collection_news,
}


class VectorStore:
    """
    Thin wrapper around ChromaDB for upsert + dense similarity search.
    Hybrid search (dense + BM25) is handled in hybrid_search.py.
    """

    def __init__(self, collection_name: str) -> None:
        self.client = get_chroma_client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Read — dense search only (hybrid_search.py adds BM25 on top)
    # ------------------------------------------------------------------

    def dense_search(
        self,
        query_embedding: list[float],
        ticker: str,
        top_k: int = 20,
        doc_types: list[DocType] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Cosine similarity search filtered by ticker (and optionally doc_type).
        Returns top_k RetrievedChunk objects.
        """
        ticker_filter = {"ticker": {"$eq": ticker}}
        if doc_types:
            type_values = [dt.value for dt in doc_types]
            if len(type_values) == 1:
                doc_filter = {"doc_type": {"$eq": type_values[0]}}
            else:
                doc_filter = {"doc_type": {"$in": type_values}}
            where = {"$and": [ticker_filter, doc_filter]}
        else:
            where = ticker_filter

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance → similarity score
            score = 1.0 - dist
            chunks.append(_meta_to_chunk(text, meta, dense_score=score))

        return chunks

    def get_all_texts_for_bm25(self, ticker: str) -> tuple[list[str], list[dict]]:
        """Fetch all document texts for a ticker — used to build BM25 index."""
        results = self.collection.get(
            where={"ticker": ticker},
            include=["documents", "metadatas"],
        )
        return results["documents"], results["metadatas"]

    def count(self) -> int:
        return self.collection.count()


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _meta_to_chunk(text: str, meta: dict, dense_score: float = 0.0) -> RetrievedChunk:
    from datetime import datetime
    date = None
    if meta.get("date"):
        try:
            date = datetime.fromisoformat(meta["date"])
        except ValueError:
            pass

    return RetrievedChunk(
        text=text,
        doc_type=DocType(meta.get("doc_type", DocType.SEC_10K.value)),
        source_url=meta.get("source_url"),
        source_title=meta.get("source_title"),
        ticker=meta.get("ticker", ""),
        page_number=meta.get("page_number"),
        chunk_index=meta.get("chunk_index"),
        date=date,
        dense_score=dense_score,
    )
