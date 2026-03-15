"""
api/schemas.py

HTTP boundary models for the FastAPI layer.
Translates internal core.models objects into JSON-serializable shapes.
Never expose internal Pydantic models (with nested enums/objects) directly.
"""

from __future__ import annotations

from pydantic import BaseModel

from core.models import ResearchReport


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    query: str
    ticker: str
    company_name: str | None = None


# ---------------------------------------------------------------------------
# Response building blocks
# ---------------------------------------------------------------------------

class CitationResponse(BaseModel):
    citation_id: str
    quote: str
    relevance: str
    source_title: str | None
    doc_type: str           # DocType.value — plain string for JSON consumers
    date: str | None        # ISO 8601 string or None
    ticker: str
    page_number: int | None


class CriticIssueResponse(BaseModel):
    claim: str
    issue_type: str         # unsupported | contradicted | ambiguous | hallucination
    severity: str           # low | medium | high
    explanation: str


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class ResearchResponse(BaseModel):
    query: str
    ticker: str
    final_answer: str
    final_confidence: float
    key_findings: list[str]
    citations: list[CitationResponse]
    issues: list[CriticIssueResponse]
    passed: bool
    critique_summary: str
    agent_summaries: dict[str, str]   # AgentName.value -> summary text
    total_latency_ms: int | None


# ---------------------------------------------------------------------------
# Mapper: ResearchReport → ResearchResponse
# ---------------------------------------------------------------------------

def report_to_response(report: ResearchReport) -> ResearchResponse:
    """
    Flatten internal ResearchReport into a JSON-serializable ResearchResponse.

    Note: ResearchReport.final_answer and .final_confidence are @property
    decorators — they are NOT stored fields and won't appear in model_dump().
    They must be accessed directly.
    """
    # Citations from synthesis
    citations: list[CitationResponse] = []
    if report.synthesis:
        for c in report.synthesis.citations:
            chunk = c.chunk
            citations.append(CitationResponse(
                citation_id=c.citation_id,
                quote=c.quote,
                relevance=c.relevance,
                source_title=chunk.source_title,
                doc_type=chunk.doc_type.value,
                date=chunk.date.isoformat() if chunk.date else None,
                ticker=chunk.ticker,
                page_number=chunk.page_number,
            ))

    # Critic issues
    issues: list[CriticIssueResponse] = []
    if report.critic_review:
        for issue in report.critic_review.issues:
            issues.append(CriticIssueResponse(
                claim=issue.claim,
                issue_type=issue.issue_type,
                severity=issue.severity,
                explanation=issue.explanation,
            ))

    # Agent summaries — convert AgentName enum keys to strings
    agent_summaries: dict[str, str] = {
        name.value: result.summary
        for name, result in report.agent_results.items()
        if result.summary
    }

    return ResearchResponse(
        query=report.query.query,
        ticker=report.query.ticker,
        final_answer=report.final_answer,           # @property
        final_confidence=report.final_confidence,   # @property
        key_findings=report.synthesis.key_findings if report.synthesis else [],
        citations=citations,
        issues=issues,
        passed=report.critic_review.passed if report.critic_review else True,
        critique_summary=report.critic_review.critique_summary if report.critic_review else "",
        agent_summaries=agent_summaries,
        total_latency_ms=report.total_latency_ms,
    )
