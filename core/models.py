"""
core/models.py

Pydantic data models shared across all agents and layers.
These are the contracts — never pass raw dicts between agents.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocType(str, Enum):
    SEC_10K = "sec_10k"
    SEC_10Q = "sec_10q"
    EARNINGS_CALL = "earnings_call"
    NEWS = "news"


class AgentName(str, Enum):
    ORCHESTRATOR = "orchestrator"
    SEC_FILINGS = "sec_filings"
    EARNINGS_CALL = "earnings_call"
    NEWS = "news"
    SYNTHESIS = "synthesis"
    CRITIC = "critic"


# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class ResearchQuery(BaseModel):
    """User's research question with context."""
    query: str = Field(..., description="Natural language research question")
    ticker: str = Field(..., description="Company ticker symbol e.g. AAPL")
    company_name: str | None = Field(None, description="Full company name if known")
    date_from: datetime | None = Field(None, description="Start of date range for docs")
    date_to: datetime | None = Field(None, description="End of date range for docs")
    session_id: str = Field(default_factory=lambda: _new_id(), description="Unique session ID")

    model_config = {"json_schema_extra": {
        "example": {
            "query": "What are the biggest risks Apple faces in 2024?",
            "ticker": "AAPL",
            "company_name": "Apple Inc."
        }
    }}


# ---------------------------------------------------------------------------
# Retrieval Models
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A single retrieved document chunk with metadata."""
    text: str = Field(..., description="Chunk text content")
    doc_type: DocType
    source_url: str | None = None
    source_title: str | None = None
    ticker: str
    page_number: int | None = None
    chunk_index: int | None = None
    date: datetime | None = None

    # Retrieval scores
    dense_score: float | None = Field(None, ge=0.0, le=1.0)
    sparse_score: float | None = Field(None, ge=0.0)
    rrf_score: float | None = Field(None, ge=0.0)
    rerank_score: float | None = Field(None, ge=0.0, le=1.0)

    @property
    def citation_label(self) -> str:
        """Human-readable citation string."""
        parts = [self.doc_type.value]
        if self.date:
            parts.append(self.date.strftime("%Y"))
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        return f"[{self.ticker} {' '.join(parts)}]"


# ---------------------------------------------------------------------------
# Agent Output Models
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Output from any specialist retrieval agent."""
    agent_name: AgentName
    query: ResearchQuery
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    summary: str = Field("", description="Agent's synthesis of retrieved chunks")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Agent's self-assessed confidence")
    error: str | None = Field(None, description="Set if agent encountered an error")
    latency_ms: int | None = None
    tokens_used: int | None = None


class SubQuery(BaseModel):
    """A decomposed sub-question from the Orchestrator."""
    sub_query: str
    target_agent: AgentName
    rationale: str


class OrchestratorPlan(BaseModel):
    """Orchestrator's decomposition of the user query."""
    original_query: ResearchQuery
    sub_queries: list[SubQuery]
    reasoning: str


# ---------------------------------------------------------------------------
# Synthesis & Critic Models
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A cited source in the final answer."""
    citation_id: str = Field(..., description="e.g. [1], [2]")
    chunk: RetrievedChunk
    quote: str = Field(..., description="The specific quoted passage")
    relevance: str = Field(..., description="Why this supports the claim")


class SynthesisResult(BaseModel):
    """Output from the SynthesisAgent."""
    answer: str = Field(..., description="Full markdown answer with inline citations")
    citations: list[Citation] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list, description="Bullet-point summary")
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    agent_results_used: list[AgentName] = Field(default_factory=list)


class CriticIssue(BaseModel):
    """A single issue found by the CriticAgent."""
    claim: str = Field(..., description="The claim being critiqued")
    issue_type: str = Field(..., description="unsupported | contradicted | ambiguous | hallucination")
    explanation: str
    severity: str = Field(..., description="low | medium | high")


class CriticReview(BaseModel):
    """Output from the CriticAgent."""
    issues: list[CriticIssue] = Field(default_factory=list)
    revised_confidence: float = Field(..., ge=0.0, le=1.0)
    final_answer: str = Field(..., description="Potentially revised answer after critique")
    critique_summary: str
    passed: bool = Field(..., description="True if no high-severity issues found")


# ---------------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------------

class ResearchReport(BaseModel):
    """The complete output of a full research run."""
    query: ResearchQuery
    orchestrator_plan: OrchestratorPlan | None = None
    agent_results: dict[AgentName, AgentResult] = Field(default_factory=dict)
    synthesis: SynthesisResult | None = None
    critic_review: CriticReview | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    total_latency_ms: int | None = None

    @property
    def final_answer(self) -> str:
        """Best available answer from the report."""
        if self.critic_review:
            return self.critic_review.final_answer
        if self.synthesis:
            return self.synthesis.answer
        # Fallback: first available agent summary
        for result in self.agent_results.values():
            if result.summary:
                return result.summary
        return "No answer could be generated."

    @property
    def final_confidence(self) -> float:
        if self.critic_review:
            return self.critic_review.revised_confidence
        if self.synthesis:
            return self.synthesis.overall_confidence
        return 0.0


# ---------------------------------------------------------------------------
# Shared Memory (passed between agents in a session)
# ---------------------------------------------------------------------------

class SharedMemory(BaseModel):
    """
    Mutable scratchpad shared across all agents in a single research session.
    Each agent writes its intermediate reasoning here so downstream agents
    can benefit from prior work.
    """
    session_id: str
    query: ResearchQuery
    scratchpad: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyed by agent name. Each agent stores its notes here."
    )
    retrieved_chunks_by_agent: dict[AgentName, list[RetrievedChunk]] = Field(
        default_factory=dict
    )
    agent_summaries: dict[AgentName, str] = Field(default_factory=dict)

    def add_chunks(self, agent: AgentName, chunks: list[RetrievedChunk]) -> None:
        existing = self.retrieved_chunks_by_agent.get(agent, [])
        self.retrieved_chunks_by_agent[agent] = existing + chunks

    def all_chunks(self) -> list[RetrievedChunk]:
        """Flatten all chunks from all agents."""
        result = []
        for chunks in self.retrieved_chunks_by_agent.values():
            result.extend(chunks)
        return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _new_id() -> str:
    import uuid
    return str(uuid.uuid4())[:8]
