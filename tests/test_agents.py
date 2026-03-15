"""
tests/test_agents.py

Unit tests for the agent layer.
All Claude API calls are mocked — no real API calls made here.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from core.models import (
    AgentName,
    AgentResult,
    DocType,
    ResearchQuery,
    RetrievedChunk,
    SharedMemory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_query(query: str = "What are Apple's risks?", ticker: str = "AAPL") -> ResearchQuery:
    return ResearchQuery(query=query, ticker=ticker, company_name="Apple Inc.")


def make_memory(query: ResearchQuery | None = None) -> SharedMemory:
    q = query or make_query()
    return SharedMemory(session_id="test-session", query=q)


def make_chunk(text: str = "Apple faces significant competition.", ticker: str = "AAPL") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        doc_type=DocType.SEC_10K,
        ticker=ticker,
        date=datetime(2023, 9, 30),
        source_title="Apple 10-K 2023",
        dense_score=0.9,
    )


# ---------------------------------------------------------------------------
# BaseAgent error handling
# ---------------------------------------------------------------------------

class TestBaseAgentErrorHandling:
    def test_run_catches_exception_and_returns_agent_result(self):
        """BaseAgent.run() must never raise — always returns AgentResult."""
        from agents.synthesis_agent import SynthesisAgent

        agent = SynthesisAgent()
        query = make_query()
        memory = make_memory(query)

        # Patch _run to raise
        with patch.object(agent, "_run", side_effect=RuntimeError("boom")):
            result = agent.run(query, memory)

        assert result.error == "boom"
        assert result.confidence == 0.0
        assert result.agent_name == AgentName.SYNTHESIS

    def test_run_sets_latency_ms(self):
        from agents.synthesis_agent import SynthesisAgent

        agent = SynthesisAgent()
        query = make_query()
        memory = make_memory(query)

        with patch.object(agent, "_run", side_effect=RuntimeError("err")):
            result = agent.run(query, memory)

        assert result.latency_ms is not None
        assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# OrchestratorAgent
# ---------------------------------------------------------------------------

class TestOrchestratorAgent:
    def _make_decompose_response(self) -> str:
        return json.dumps({
            "sub_queries": [
                {
                    "sub_query": "What are Apple's risk factors?",
                    "target_agent": "sec_filings",
                    "rationale": "Risk factors are in 10-K filings",
                }
            ],
            "reasoning": "The question is about risks, best answered by SEC filings.",
        })

    def test_decompose_query_parses_plan(self):
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        query = make_query()

        with patch.object(orchestrator, "_call_claude", return_value=(self._make_decompose_response(), 100)):
            plan = orchestrator._decompose_query(query)

        assert len(plan.sub_queries) == 1
        assert plan.sub_queries[0].target_agent == AgentName.SEC_FILINGS
        assert "risk" in plan.sub_queries[0].sub_query.lower()

    def test_decompose_query_handles_unknown_agent(self):
        """Unknown target_agent values should fall back to sec_filings."""
        from agents.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent()
        query = make_query()

        bad_response = json.dumps({
            "sub_queries": [
                {
                    "sub_query": "Some question",
                    "target_agent": "nonexistent_agent",
                    "rationale": "test",
                }
            ],
            "reasoning": "test",
        })

        with patch.object(orchestrator, "_call_claude", return_value=(bad_response, 50)):
            plan = orchestrator._decompose_query(query)

        assert plan.sub_queries[0].target_agent == AgentName.SEC_FILINGS


# ---------------------------------------------------------------------------
# SynthesisAgent
# ---------------------------------------------------------------------------

class TestSynthesisAgent:
    def _make_synthesis_response(self) -> str:
        return json.dumps({
            "answer": "Apple faces competition [CHUNK_0] and supply chain risks [CHUNK_0].",
            "citations": [
                {
                    "citation_id": "[1]",
                    "chunk_index": 0,
                    "quote": "Apple faces significant competition",
                    "relevance": "Directly addresses competition risk.",
                }
            ],
            "key_findings": ["Competition is a key risk", "Supply chain dependency"],
            "overall_confidence": 0.78,
            "agent_results_used": ["sec_filings"],
        })

    def test_synthesis_runs_with_chunks_in_memory(self):
        from agents.synthesis_agent import SynthesisAgent

        agent = SynthesisAgent()
        query = make_query()
        memory = make_memory(query)
        memory.add_chunks(AgentName.SEC_FILINGS, [make_chunk()])
        memory.agent_summaries[AgentName.SEC_FILINGS] = "Apple faces competition risks."

        with patch.object(agent, "_call_claude", return_value=(self._make_synthesis_response(), 200)):
            result = agent.run(query, memory)

        assert result.error is None
        assert result.confidence == pytest.approx(0.78)
        assert "synthesis" in memory.scratchpad
        assert memory.scratchpad["synthesis"]["overall_confidence"] == pytest.approx(0.78)

    def test_synthesis_guard_empty_chunks(self):
        from agents.synthesis_agent import SynthesisAgent

        agent = SynthesisAgent()
        query = make_query()
        memory = make_memory(query)  # no chunks added

        result = agent.run(query, memory)

        assert result.error is not None
        assert "No chunks" in result.error
        assert result.confidence == 0.0

    def test_synthesis_skips_out_of_range_chunk_index(self):
        from agents.synthesis_agent import SynthesisAgent

        agent = SynthesisAgent()
        query = make_query()
        memory = make_memory(query)
        memory.add_chunks(AgentName.SEC_FILINGS, [make_chunk()])  # only 1 chunk (index 0)

        bad_response = json.dumps({
            "answer": "Some answer [CHUNK_99].",
            "citations": [
                {
                    "citation_id": "[1]",
                    "chunk_index": 99,  # out of range
                    "quote": "some quote",
                    "relevance": "test",
                }
            ],
            "key_findings": ["finding"],
            "overall_confidence": 0.5,
            "agent_results_used": ["sec_filings"],
        })

        with patch.object(agent, "_call_claude", return_value=(bad_response, 100)):
            result = agent.run(query, memory)

        # Should succeed but with 0 citations (out-of-range silently skipped)
        assert result.error is None
        synthesis = memory.scratchpad.get("synthesis", {})
        assert synthesis.get("citations") == []


# ---------------------------------------------------------------------------
# CriticAgent
# ---------------------------------------------------------------------------

class TestCriticAgent:
    def _populate_synthesis_scratchpad(self, memory: SharedMemory, chunk: RetrievedChunk) -> None:
        """Write a minimal synthesis result into scratchpad."""
        from core.models import Citation, SynthesisResult

        synthesis = SynthesisResult(
            answer="Apple faces competition [1].",
            citations=[
                Citation(
                    citation_id="[1]",
                    chunk=chunk,
                    quote="Apple faces significant competition",
                    relevance="Directly relevant",
                )
            ],
            key_findings=["competition risk"],
            overall_confidence=0.75,
            agent_results_used=[AgentName.SEC_FILINGS],
        )
        memory.scratchpad["synthesis"] = synthesis.model_dump(mode="json")

    def _make_critic_response(self, passed: bool = True, n_issues: int = 0) -> str:
        issues = []
        if n_issues > 0:
            issues.append({
                "claim": "Some claim",
                "issue_type": "unsupported",
                "explanation": "No evidence in cited sources.",
                "severity": "medium",
            })
        return json.dumps({
            "issues": issues,
            "revised_confidence": 0.70 if n_issues == 0 else 0.60,
            "final_answer": "Apple faces competition [1].",
            "critique_summary": "Answer is well-grounded." if passed else "Minor issues found.",
            "passed": passed,
        })

    def test_critic_passes_clean_answer(self):
        from agents.critic_agent import CriticAgent

        agent = CriticAgent()
        query = make_query()
        memory = make_memory(query)
        self._populate_synthesis_scratchpad(memory, make_chunk())

        with patch.object(agent, "_call_claude", return_value=(self._make_critic_response(passed=True), 150)):
            result = agent.run(query, memory)

        assert result.error is None
        assert result.confidence == pytest.approx(0.70)
        assert memory.scratchpad["critic"]["passed"] is True
        assert memory.scratchpad["critic"]["issues"] == []

    def test_critic_guard_missing_synthesis(self):
        from agents.critic_agent import CriticAgent

        agent = CriticAgent()
        query = make_query()
        memory = make_memory(query)  # no synthesis in scratchpad

        result = agent.run(query, memory)

        assert result.error is not None
        assert "synthesis" in result.error.lower()

    def test_critic_normalizes_unknown_severity(self):
        from agents.critic_agent import CriticAgent

        agent = CriticAgent()
        query = make_query()
        memory = make_memory(query)
        self._populate_synthesis_scratchpad(memory, make_chunk())

        bad_issues = json.dumps({
            "issues": [
                {
                    "claim": "Some claim",
                    "issue_type": "hallucination",
                    "explanation": "test",
                    "severity": "extreme",  # invalid — should normalize to "low"
                }
            ],
            "revised_confidence": 0.5,
            "final_answer": "answer",
            "critique_summary": "test",
            "passed": True,
        })

        with patch.object(agent, "_call_claude", return_value=(bad_issues, 100)):
            result = agent.run(query, memory)

        assert result.error is None
        issues = memory.scratchpad["critic"]["issues"]
        assert len(issues) == 1
        assert issues[0]["severity"] == "low"  # normalized


# ---------------------------------------------------------------------------
# SharedMemory
# ---------------------------------------------------------------------------

class TestSharedMemory:
    def test_add_and_retrieve_chunks(self):
        query = make_query()
        memory = make_memory(query)
        chunks = [make_chunk(text=f"chunk {i}") for i in range(3)]

        memory.add_chunks(AgentName.SEC_FILINGS, chunks)
        all_chunks = memory.all_chunks()

        assert len(all_chunks) == 3

    def test_all_chunks_flattens_multiple_agents(self):
        query = make_query()
        memory = make_memory(query)

        memory.add_chunks(AgentName.SEC_FILINGS, [make_chunk("sec chunk")])
        memory.add_chunks(AgentName.NEWS, [make_chunk("news chunk")])

        all_chunks = memory.all_chunks()
        assert len(all_chunks) == 2

    def test_add_chunks_accumulates(self):
        query = make_query()
        memory = make_memory(query)

        memory.add_chunks(AgentName.SEC_FILINGS, [make_chunk("a")])
        memory.add_chunks(AgentName.SEC_FILINGS, [make_chunk("b")])

        chunks = memory.retrieved_chunks_by_agent[AgentName.SEC_FILINGS]
        assert len(chunks) == 2
