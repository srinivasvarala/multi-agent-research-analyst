"""
ingestion/embedder.py

Embedding wrapper. Uses Voyage AI (preferred) or OpenAI as fallback.
Voyage AI `voyage-3` produces 1024-dim embeddings optimized for retrieval.
"""

from __future__ import annotations

import time
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from core.config import get_settings

logger = structlog.get_logger()


class Embedder:
    """
    Unified embedding interface.
    Auto-selects provider based on available API keys.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.embedding_provider
        self._client = self._build_client()
        logger.info("embedder_init", provider=self.provider)

    def _build_client(self):
        if self.provider == "voyage":
            import voyageai
            return voyageai.Client(api_key=self.settings.voyage_api_key)
        else:
            from openai import OpenAI
            return OpenAI(api_key=self.settings.openai_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        if not texts:
            return []

        if self.provider == "voyage":
            return self._embed_voyage(texts)
        else:
            return self._embed_openai(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        if self.provider == "voyage":
            return self._embed_voyage_query(text)
        return self.embed([text])[0]

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _embed_voyage_docs(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(
            texts,
            model=self.settings.embedding_model_voyage,
            input_type="document",
        )
        return result.embeddings

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _embed_voyage_query(self, text: str) -> list[float]:
        result = self._client.embed(
            [text],
            model=self.settings.embedding_model_voyage,
            input_type="query",
        )
        return result.embeddings[0]

    def _embed_voyage(self, texts: list[str]) -> list[list[float]]:
        return self._embed_voyage_docs(texts)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        from openai import RateLimitError

        max_retries = 6
        delay = 1.0
        for attempt in range(max_retries):
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self.settings.embedding_model_openai,
                )
                return [item.embedding for item in response.data]
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(
                    "openai_rate_limit_retry",
                    attempt=attempt + 1,
                    wait_s=delay,
                    error=str(e),
                )
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
