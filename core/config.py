"""
core/config.py

All configuration loaded from environment variables.
Never hardcode keys — always use this module.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ---- Anthropic ----
    anthropic_api_key: str
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens_retrieval: int = 2048
    claude_max_tokens_synthesis: int = 4096
    claude_max_tokens_critic: int = 2048

    # ---- Embeddings ----
    voyage_api_key: str | None = None
    openai_api_key: str | None = None           # Fallback if no Voyage key
    embedding_model_voyage: str = "voyage-3"
    embedding_model_openai: str = "text-embedding-3-small"
    embedding_dimension: int = 1024             # voyage-3 default

    # ---- ChromaDB ----
    chroma_persist_path: str = "./data/vectorstore"
    chroma_collection_sec: str = "sec_filings"
    chroma_collection_earnings: str = "earnings_calls"
    chroma_collection_news: str = "news"

    # ---- Retrieval ----
    retrieval_top_k_dense: int = 20
    retrieval_top_k_sparse: int = 20
    retrieval_top_k_rrf: int = 10
    retrieval_top_k_rerank: int = 5
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # ---- News API ----
    news_api_key: str | None = None

    # ---- API Server ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8501"]

    # ---- Logging ----
    log_level: str = "INFO"

    @property
    def embedding_provider(self) -> str:
        return "voyage" if self.voyage_api_key else "openai"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
