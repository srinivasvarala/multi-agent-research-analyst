"""
agents/news_agent.py

Retrieves and summarizes relevant passages from recent news articles.
Uses hybrid search (dense + BM25) and Claude tool use for structured output.
"""

from __future__ import annotations

import json

import structlog

from agents.base_agent import BaseAgent
from core.config import get_settings
from core.models import AgentName, AgentResult, DocType, ResearchQuery, RetrievedChunk, SharedMemory
from core.retrieval.hybrid_search import HybridSearcher
from core.retrieval.reranker import Reranker
from ingestion.embedder import Embedder

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a specialized financial analyst with deep expertise in interpreting news coverage.
Your job is to analyze retrieved news article passages and provide a focused,
evidence-based answer to the research question.

Rules:
1. Only use information from the provided passages — do not hallucinate or use outside knowledge
2. For every claim, cite the specific passage using its [CHUNK_N] label
3. Prioritize the most recent articles — recency matters in news analysis
4. Clearly distinguish factual reporting from analyst opinion or market speculation
5. Note the overall market sentiment and any consensus or divergence among sources
6. If the passages don't contain enough information, clearly state what is missing
7. Focus on what's most relevant to the question — be concise but complete
8. Use structured output via the provided tool — never respond in free text
"""

# ---------------------------------------------------------------------------
# Tool Definition (structured output via tool use)
# ---------------------------------------------------------------------------

NEWS_ANALYSIS_TOOL = {
    "name": "news_analysis",
    "description": "Structured output of news article analysis with citations",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-4 paragraph answer to the research question, grounded in the passages. Use [CHUNK_N] inline citations."
            },
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 bullet-point key findings from the news articles"
            },
            "cited_chunk_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of CHUNK_N indices actually cited in the summary"
            },
            "confidence": {
                "type": "number",
                "description": "Confidence 0.0-1.0 based on relevance and completeness of retrieved passages",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "market_sentiment": {
                "type": "string",
                "description": "Overall market sentiment from the news: e.g. 'bullish', 'bearish', 'mixed', 'neutral'"
            },
            "most_recent_date": {
                "type": "string",
                "description": "Date of the most recent article cited, in YYYY-MM-DD format if available"
            },
            "missing_information": {
                "type": "string",
                "description": "What information would improve this answer but wasn't found in the news articles"
            }
        },
        "required": ["summary", "key_findings", "cited_chunk_indices", "confidence"]
    }
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class NewsAgent(BaseAgent):
    """
    Retrieves relevant passages from news articles using hybrid search,
    then uses Claude to synthesize a grounded answer with citations and market sentiment.
    """

    name = AgentName.NEWS

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self.searcher = HybridSearcher(self.settings.chroma_collection_news)
        self.embedder = Embedder()
        self.reranker = Reranker()

    def _run(self, query: ResearchQuery, memory: SharedMemory) -> AgentResult:
        # 1. Embed the query
        query_embedding = self.embedder.embed_query(query.query)

        # 2. Hybrid retrieval (dense + BM25 + RRF → top-10)
        chunks = self.searcher.search(
            query_text=query.query,
            query_embedding=query_embedding,
            ticker=query.ticker,
            doc_types=[DocType.NEWS],
        )

        if not chunks:
            self.log.warning("no_chunks_retrieved", ticker=query.ticker)
            return AgentResult(
                agent_name=self.name,
                query=query,
                chunks=[],
                summary=(
                    f"No news articles found for {query.ticker}. "
                    "Please ingest articles first using scripts/ingest_news.py"
                ),
                confidence=0.0,
            )

        # 3. Rerank top-10 → top-5
        chunks = self.reranker.rerank(
            query=query.query,
            chunks=chunks,
            top_k=self.settings.retrieval_top_k_rerank,
        )

        # 4. Build context for Claude
        context = self._format_context(chunks)
        user_message = self._build_user_message(query, context)

        # 5. Call Claude with tool use
        raw_response, tokens = self._call_claude(
            system=SYSTEM_PROMPT,
            user_message=user_message,
            tools=[NEWS_ANALYSIS_TOOL],
            max_tokens=self.settings.claude_max_tokens_retrieval,
        )

        # 6. Parse tool output
        analysis = json.loads(raw_response)

        # 7. Filter chunks to only cited ones
        cited_indices = set(analysis.get("cited_chunk_indices", []))
        cited_chunks = [
            chunks[i] for i in cited_indices
            if i < len(chunks)
        ]

        # Write scratchpad for downstream agents
        memory.scratchpad[self.name.value] = {
            "key_findings": analysis.get("key_findings", []),
            "market_sentiment": analysis.get("market_sentiment", ""),
            "most_recent_date": analysis.get("most_recent_date", ""),
            "missing_info": analysis.get("missing_information", ""),
            "n_chunks_retrieved": len(chunks),
            "n_chunks_cited": len(cited_chunks),
        }

        return AgentResult(
            agent_name=self.name,
            query=query,
            chunks=cited_chunks,
            summary=analysis["summary"],
            confidence=analysis["confidence"],
            tokens_used=tokens,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks as numbered context blocks for Claude."""
        lines = []
        for i, chunk in enumerate(chunks):
            meta_parts = [chunk.doc_type.value, chunk.ticker]
            if chunk.date:
                meta_parts.append(chunk.date.strftime("%Y-%m-%d"))
            if chunk.page_number:
                meta_parts.append(f"page {chunk.page_number}")
            if chunk.source_title:
                meta_parts.append(chunk.source_title)

            meta_str = " | ".join(meta_parts)
            lines.append(f"[CHUNK_{i}] ({meta_str})\n{chunk.text}\n")

        return "\n---\n".join(lines)

    def _build_user_message(self, query: ResearchQuery, context: str) -> str:
        return f"""Research Question: {query.query}
Company: {query.company_name or query.ticker} ({query.ticker})

Retrieved News Article Passages:
---
{context}
---

Please analyze these passages and answer the research question using the news_analysis tool.
"""


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from core.models import SharedMemory

    parser = argparse.ArgumentParser(description="Test NewsAgent directly")
    parser.add_argument("--query", required=True, help="Research question")
    parser.add_argument("--ticker", default="AAPL", help="Company ticker")
    args = parser.parse_args()

    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()

    rq = ResearchQuery(query=args.query, ticker=args.ticker.upper())
    mem = SharedMemory(session_id=rq.session_id, query=rq)

    agent = NewsAgent()
    result = agent.run(rq, mem)

    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")
    else:
        console.print(Panel(
            Markdown(result.summary),
            title=f"[bold]News Analysis — {args.ticker.upper()}[/bold]",
            subtitle=f"Confidence: {result.confidence:.0%} | Chunks cited: {len(result.chunks)} | {result.latency_ms}ms",
        ))
        news_scratch = mem.scratchpad.get("news", {})
        sentiment = news_scratch.get("market_sentiment")
        most_recent = news_scratch.get("most_recent_date")
        if sentiment:
            console.print(f"\n[bold]Market Sentiment:[/bold] {sentiment}")
        if most_recent:
            console.print(f"[bold]Most Recent Article:[/bold] {most_recent}")
