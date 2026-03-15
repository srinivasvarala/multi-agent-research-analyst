"""
tests/test_api.py

FastAPI endpoint tests.
The full pipeline is mocked — no real Claude API calls or ChromaDB access.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.models import (
    AgentName,
    AgentResult,
    Citation,
    CriticIssue,
    CriticReview,
    DocType,
    OrchestratorPlan,
    ResearchQuery,
    ResearchReport,
    RetrievedChunk,
    SharedMemory,
    SubQuery,
    SynthesisResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_report(query_str: str = "What are Apple's risks?", ticker: str = "AAPL") -> ResearchReport:
    """Build a minimal but complete ResearchReport for testing."""
    rq = ResearchQuery(query=query_str, ticker=ticker)

    chunk = RetrievedChunk(
        text="Apple faces competition in all markets.",
        doc_type=DocType.SEC_10K,
        ticker=ticker,
        date=datetime(2023, 9, 30),
        source_title="Apple 10-K 2023",
        dense_score=0.9,
    )

    citation = Citation(
        citation_id="[1]",
        chunk=chunk,
        quote="Apple faces competition in all markets",
        relevance="Supports competition risk claim",
    )

    synthesis = SynthesisResult(
        answer="Apple faces significant competition [1].",
        citations=[citation],
        key_findings=["Competition is a key risk", "Supply chain dependency"],
        overall_confidence=0.78,
        agent_results_used=[AgentName.SEC_FILINGS],
    )

    critic_review = CriticReview(
        issues=[],
        revised_confidence=0.80,
        final_answer="Apple faces significant competition [1].",
        critique_summary="Answer is well-grounded in cited sources.",
        passed=True,
    )

    plan = OrchestratorPlan(
        original_query=rq,
        sub_queries=[
            SubQuery(
                sub_query="What are Apple's risk factors?",
                target_agent=AgentName.SEC_FILINGS,
                rationale="Risk factors in 10-K",
            )
        ],
        reasoning="Best answered by SEC filings.",
    )

    sec_result = AgentResult(
        agent_name=AgentName.SEC_FILINGS,
        query=rq,
        chunks=[chunk],
        summary="Apple faces competition and supply chain risks.",
        confidence=0.78,
        latency_ms=1200,
    )

    return ResearchReport(
        query=rq,
        orchestrator_plan=plan,
        agent_results={AgentName.SEC_FILINGS: sec_result},
        synthesis=synthesis,
        critic_review=critic_review,
        total_latency_ms=4500,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """FastAPI test client with mocked settings."""
    with patch("core.config.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            anthropic_api_key="test-key",
            api_cors_origins=["http://localhost:3000"],
            api_host="0.0.0.0",
            api_port=8000,
            log_level="INFO",
        )
        from api.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_does_not_require_auth(self, client):
        response = client.get("/health")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Research endpoint
# ---------------------------------------------------------------------------

class TestResearchEndpoint:
    def _mock_pipeline(self, report: ResearchReport):
        """Context manager that patches run_full_pipeline."""
        mem = SharedMemory(session_id="test", query=report.query)
        mock_orch = MagicMock()
        mock_orch.run_full_pipeline.return_value = (report, mem)
        return patch("api.routes.research.OrchestratorAgent", return_value=mock_orch)

    def test_research_returns_200_with_valid_response(self, client):
        report = make_mock_report()

        with self._mock_pipeline(report):
            response = client.post(
                "/research",
                json={"query": "What are Apple's risks?", "ticker": "AAPL"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert data["final_confidence"] == pytest.approx(0.80)
        assert data["passed"] is True
        assert len(data["key_findings"]) == 2
        assert len(data["citations"]) == 1

    def test_research_ticker_uppercased(self, client):
        report = make_mock_report(ticker="AAPL")

        with self._mock_pipeline(report):
            response = client.post(
                "/research",
                json={"query": "Apple risks", "ticker": "aapl"},  # lowercase
            )

        assert response.status_code == 200

    def test_research_citation_shape(self, client):
        report = make_mock_report()

        with self._mock_pipeline(report):
            response = client.post(
                "/research",
                json={"query": "Apple risks", "ticker": "AAPL"},
            )

        citation = response.json()["citations"][0]
        assert "citation_id" in citation
        assert "quote" in citation
        assert "relevance" in citation
        assert "doc_type" in citation
        assert citation["doc_type"] == "sec_10k"

    def test_research_returns_500_on_pipeline_error(self, client):
        with patch("api.routes.research.OrchestratorAgent") as mock_cls:
            mock_orch = MagicMock()
            mock_orch.run_full_pipeline.side_effect = RuntimeError("pipeline blew up")
            mock_cls.return_value = mock_orch

            response = client.post(
                "/research",
                json={"query": "Apple risks", "ticker": "AAPL"},
            )

        assert response.status_code == 500
        assert "pipeline blew up" in response.json()["detail"]

    def test_research_requires_query_field(self, client):
        response = client.post("/research", json={"ticker": "AAPL"})
        assert response.status_code == 422  # Pydantic validation error

    def test_research_requires_ticker_field(self, client):
        response = client.post("/research", json={"query": "What are risks?"})
        assert response.status_code == 422

    def test_research_company_name_is_optional(self, client):
        report = make_mock_report()

        with self._mock_pipeline(report):
            response = client.post(
                "/research",
                json={"query": "Apple risks", "ticker": "AAPL"},
                # no company_name
            )

        assert response.status_code == 200

    def test_research_agent_summaries_in_response(self, client):
        report = make_mock_report()

        with self._mock_pipeline(report):
            response = client.post(
                "/research",
                json={"query": "Apple risks", "ticker": "AAPL"},
            )

        summaries = response.json()["agent_summaries"]
        assert "sec_filings" in summaries
        assert len(summaries["sec_filings"]) > 0


# ---------------------------------------------------------------------------
# Schemas: report_to_response
# ---------------------------------------------------------------------------

class TestReportToResponse:
    def test_final_answer_from_critic(self):
        from api.schemas import report_to_response

        report = make_mock_report()
        response = report_to_response(report)

        # Critic review exists — final_answer comes from critic
        assert response.final_answer == report.critic_review.final_answer

    def test_final_confidence_from_critic(self):
        from api.schemas import report_to_response

        report = make_mock_report()
        response = report_to_response(report)

        assert response.final_confidence == pytest.approx(0.80)

    def test_no_critic_fallback_to_synthesis(self):
        from api.schemas import report_to_response

        report = make_mock_report()
        report.critic_review = None
        response = report_to_response(report)

        assert response.final_answer == report.synthesis.answer
        assert response.passed is True  # default when no critic

    def test_empty_agent_results(self):
        from api.schemas import report_to_response

        report = make_mock_report()
        report.agent_results = {}
        response = report_to_response(report)

        assert response.agent_summaries == {}
