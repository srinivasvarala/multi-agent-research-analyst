"""
agents/synthesis_agent.py

SynthesisAgent — merges multi-source evidence from all specialist agents into
a single grounded answer with inline citations.

Does NOT retrieve — reads memory.all_chunks() and memory.agent_summaries.
"""

from __future__ import annotations

import json

import structlog

from agents.base_agent import BaseAgent
from core.models import (
    AgentName,
    AgentResult,
    Citation,
    ResearchQuery,
    RetrievedChunk,
    SharedMemory,
    SynthesisResult,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial research analyst synthesizing evidence from multiple sources
into a single, well-grounded answer.

Your inputs:
- A research question about a company
- Summaries from specialist agents (SEC filings, earnings calls, news)
- Individual source chunks with full text

Your responsibilities:
1. **Integrate across sources** — draw on SEC filings for facts, earnings calls for management tone,
   and news for market context. Surface convergence and disagreement.
2. **Cite every claim** — every factual assertion must have an inline citation [CHUNK_N] where N is
   the zero-based index of the supporting chunk.
3. **Surface disagreements** — if management guidance conflicts with analyst sentiment or news, flag
   it explicitly.
4. **Calibrate confidence**:
   - 0.8–1.0: Strong multi-source corroboration, relevant and recent data
   - 0.5–0.79: Partial coverage, data gaps, or single-source claims
   - 0.2–0.49: Thin evidence, heavily caveated, or low-relevance chunks
   - 0.0–0.19: Near-empty corpus, severe data gaps

Always use the synthesize_answer tool — never respond in free text.
"""

# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

SYNTHESIZE_ANSWER_TOOL = {
    "name": "synthesize_answer",
    "description": "Synthesize all retrieved evidence into a final grounded answer with citations",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "Full markdown answer with inline [CHUNK_N] citations. "
                    "Use headers, bullets, and bold for readability."
                ),
            },
            "citations": {
                "type": "array",
                "description": "List of all citations used in the answer",
                "items": {
                    "type": "object",
                    "properties": {
                        "citation_id": {
                            "type": "string",
                            "description": "e.g. [1], [2], [3] — matches inline markers in the answer",
                        },
                        "chunk_index": {
                            "type": "integer",
                            "description": "Zero-based index of the chunk in the provided chunk list",
                        },
                        "quote": {
                            "type": "string",
                            "description": "Verbatim excerpt from the chunk (≤150 characters)",
                        },
                        "relevance": {
                            "type": "string",
                            "description": "One sentence explaining why this chunk supports the cited claim",
                        },
                    },
                    "required": ["citation_id", "chunk_index", "quote", "relevance"],
                },
            },
            "key_findings": {
                "type": "array",
                "description": "4-6 executive bullet points summarising the most important findings",
                "items": {"type": "string"},
            },
            "overall_confidence": {
                "type": "number",
                "description": "Confidence score 0.0–1.0 based on source coverage and agreement",
            },
            "agent_results_used": {
                "type": "array",
                "description": "Which agent corpora contributed meaningful evidence",
                "items": {
                    "type": "string",
                    "enum": ["sec_filings", "earnings_call", "news"],
                },
            },
        },
        "required": [
            "answer",
            "citations",
            "key_findings",
            "overall_confidence",
            "agent_results_used",
        ],
    },
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SynthesisAgent(BaseAgent):
    """
    Reads all retrieved chunks from SharedMemory, calls Claude to synthesize
    a grounded multi-source answer, and writes the result back to scratchpad.
    """

    name = AgentName.SYNTHESIS

    def _run(self, query: ResearchQuery, memory: SharedMemory) -> AgentResult:
        all_chunks = memory.all_chunks()

        if not all_chunks:
            self.log.warning("synthesis_no_chunks")
            return AgentResult(
                agent_name=self.name,
                query=query,
                error="No chunks available in shared memory — specialist agents may have failed.",
                confidence=0.0,
            )

        self.log.info("synthesis_start", n_chunks=len(all_chunks), n_agents=len(memory.agent_summaries))

        user_message = self._build_prompt(query, all_chunks, memory)

        raw_response, tokens = self._call_claude(
            system=SYSTEM_PROMPT,
            user_message=user_message,
            tools=[SYNTHESIZE_ANSWER_TOOL],
            max_tokens=self.settings.claude_max_tokens_synthesis,
        )

        tool_output = json.loads(raw_response)

        citations = self._build_citation_objects(tool_output, all_chunks)

        # Reconstruct agent_results_used as AgentName enums (skip unknowns)
        agent_results_used: list[AgentName] = []
        for raw_name in tool_output.get("agent_results_used", []):
            try:
                agent_results_used.append(AgentName(raw_name))
            except ValueError:
                self.log.warning("unknown_agent_name_in_synthesis", raw_name=raw_name)

        synthesis = SynthesisResult(
            answer=tool_output.get("answer", ""),
            citations=citations,
            key_findings=tool_output.get("key_findings", []),
            overall_confidence=float(tool_output.get("overall_confidence", 0.0)),
            agent_results_used=agent_results_used,
        )

        # Write to scratchpad so CriticAgent can read it
        memory.scratchpad["synthesis"] = synthesis.model_dump(mode="json")

        # Deduplicate cited chunks: same chunk cited multiple times → stored once
        cited_chunks: dict[int, RetrievedChunk] = {}
        for citation in citations:
            # Use the chunk's position in all_chunks as key
            for i, chunk in enumerate(all_chunks):
                if chunk is citation.chunk:
                    cited_chunks[i] = chunk
                    break

        return AgentResult(
            agent_name=self.name,
            query=query,
            chunks=list(cited_chunks.values()),
            summary=synthesis.answer,
            confidence=synthesis.overall_confidence,
            tokens_used=tokens,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        query: ResearchQuery,
        all_chunks: list[RetrievedChunk],
        memory: SharedMemory,
    ) -> str:
        sections: list[str] = []

        # Section 1: Research question
        sections.append(f"## Research Question\n{query.query}")
        sections.append(f"**Company:** {query.company_name or query.ticker} ({query.ticker})")

        # Section 2: Agent summaries
        if memory.agent_summaries:
            sections.append("## Agent Summaries")
            for agent_name, summary in memory.agent_summaries.items():
                label = agent_name.value if hasattr(agent_name, "value") else str(agent_name)
                sections.append(f"### {label}\n{summary}")

        # Section 3: Scratchpad signals from specialist agents
        scratchpad_signals: list[str] = []
        for key in ["sec_filings", "earnings_call", "news"]:
            sp = memory.scratchpad.get(key)
            if not sp:
                continue
            signal_lines = [f"### {key}"]
            if isinstance(sp, dict):
                for field in ("key_findings", "management_tone", "market_sentiment"):
                    val = sp.get(field)
                    if val:
                        signal_lines.append(f"**{field}:** {val}")
            scratchpad_signals.append("\n".join(signal_lines))

        if scratchpad_signals:
            sections.append("## Specialist Agent Signals\n" + "\n\n".join(scratchpad_signals))

        # Section 4: All source chunks (globally indexed)
        sections.append("## Source Chunks")
        for i, chunk in enumerate(all_chunks):
            agent_label = _chunk_agent_label(chunk, memory)
            date_str = chunk.date.strftime("%Y-%m-%d") if chunk.date else "unknown date"
            title = chunk.source_title or chunk.source_url or "untitled"
            sections.append(
                f"### [CHUNK_{i}] {agent_label} | {chunk.doc_type.value} | {date_str}\n"
                f"**Source:** {title}\n\n"
                f"{chunk.text}"
            )

        return "\n\n".join(sections)

    def _build_citation_objects(
        self,
        tool_output: dict,
        all_chunks: list[RetrievedChunk],
    ) -> list[Citation]:
        citations: list[Citation] = []
        for raw in tool_output.get("citations", []):
            idx = raw.get("chunk_index")
            if idx is None or not isinstance(idx, int):
                self.log.warning("citation_missing_chunk_index", citation_id=raw.get("citation_id"))
                continue
            if idx < 0 or idx >= len(all_chunks):
                self.log.warning(
                    "citation_chunk_index_out_of_range",
                    chunk_index=idx,
                    total_chunks=len(all_chunks),
                )
                continue
            citations.append(Citation(
                citation_id=raw.get("citation_id", f"[{idx + 1}]"),
                chunk=all_chunks[idx],
                quote=raw.get("quote", ""),
                relevance=raw.get("relevance", ""),
            ))
        return citations


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _chunk_agent_label(chunk: RetrievedChunk, memory: SharedMemory) -> str:
    """Determine which agent contributed this chunk by scanning retrieved_chunks_by_agent."""
    for agent_name, chunks in memory.retrieved_chunks_by_agent.items():
        if chunk in chunks:
            return agent_name.value if hasattr(agent_name, "value") else str(agent_name)
    return "unknown"
