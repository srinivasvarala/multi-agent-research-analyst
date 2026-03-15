"""
agents/critic_agent.py

CriticAgent — verifies claims against cited sources, flags hallucinations or
unsupported assertions, adjusts confidence, and optionally patches the answer.

Does NOT retrieve — reads memory.scratchpad["synthesis"].
"""

from __future__ import annotations

import json

import structlog

from agents.base_agent import BaseAgent
from core.models import (
    AgentName,
    AgentResult,
    CriticIssue,
    CriticReview,
    ResearchQuery,
    SharedMemory,
    SynthesisResult,
)

logger = structlog.get_logger()

# Valid enum values for validation
_VALID_ISSUE_TYPES = {"unsupported", "contradicted", "ambiguous", "hallucination"}
_VALID_SEVERITIES = {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a rigorous fact-checker and quality-control reviewer for financial research.

You receive:
- The original research question
- A synthesized answer with inline [CHUNK_N] citations
- The full text of every cited passage

Your job:
1. **Verify each cited claim** — check that the cited passage actually supports what is claimed.
2. **Flag issues** using these categories:
   - `unsupported`: The citation does not back up the claim
   - `contradicted`: The cited evidence directly contradicts the claim
   - `ambiguous`: The claim or citation is unclear enough to mislead
   - `hallucination`: The claim has no citation at all or the cited chunk doesn't contain the claimed fact
3. **Severity guide**:
   - `high`: Material factual error or complete fabrication — readers would be misled
   - `medium`: Overstatement, missing nuance, or weakly supported claim
   - `low`: Minor imprecision, stylistic exaggeration, or dated citation
4. **Revise confidence** from the original:
   - Each high-severity issue: −0.15
   - Each medium-severity issue: −0.05
   - Each low-severity issue: −0.01
   - No issues at all: +0.05 (capped at 1.0, floored at 0.0)
5. **Patch the final answer** only for high-severity issues — copy original text for low/medium,
   but fix or caveat sentences that contain high-severity problems.
6. **passed** is True if and only if there are zero high-severity issues.

Always use the critique_answer tool — never respond in free text.
"""

# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

CRITIQUE_ANSWER_TOOL = {
    "name": "critique_answer",
    "description": "Review a synthesized answer for hallucinations and unsupported claims",
    "input_schema": {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "description": "List of issues found (empty list if none)",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "The exact claim being critiqued (quote from the answer)",
                        },
                        "issue_type": {
                            "type": "string",
                            "enum": ["unsupported", "contradicted", "ambiguous", "hallucination"],
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Why this is an issue and what the evidence actually shows",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": ["claim", "issue_type", "explanation", "severity"],
                },
            },
            "revised_confidence": {
                "type": "number",
                "description": "Adjusted confidence score 0.0–1.0 after applying issue penalties",
            },
            "final_answer": {
                "type": "string",
                "description": (
                    "The final answer text. Copy the original if no high-severity issues. "
                    "Patch or add caveats to sentences with high-severity issues."
                ),
            },
            "critique_summary": {
                "type": "string",
                "description": "1-3 sentence quality assessment of the synthesized answer",
            },
            "passed": {
                "type": "boolean",
                "description": "True if and only if there are zero high-severity issues",
            },
        },
        "required": [
            "issues",
            "revised_confidence",
            "final_answer",
            "critique_summary",
            "passed",
        ],
    },
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CriticAgent(BaseAgent):
    """
    Reads SynthesisResult from scratchpad, verifies claims against cited passages,
    and returns a CriticReview with revised confidence and optional answer patches.
    """

    name = AgentName.CRITIC

    def _run(self, query: ResearchQuery, memory: SharedMemory) -> AgentResult:
        raw_synthesis = memory.scratchpad.get("synthesis")
        if not raw_synthesis:
            self.log.warning("critic_no_synthesis_in_scratchpad")
            return AgentResult(
                agent_name=self.name,
                query=query,
                error="No synthesis result found in scratchpad — SynthesisAgent must run first.",
                confidence=0.0,
            )

        try:
            synthesis = SynthesisResult.model_validate(raw_synthesis)
        except Exception as exc:
            self.log.error("critic_synthesis_parse_error", error=str(exc))
            return AgentResult(
                agent_name=self.name,
                query=query,
                error=f"Failed to parse synthesis from scratchpad: {exc}",
                confidence=0.0,
            )

        self.log.info(
            "critic_start",
            n_citations=len(synthesis.citations),
            original_confidence=synthesis.overall_confidence,
        )

        user_message = self._build_prompt(query, synthesis)

        raw_response, tokens = self._call_claude(
            system=SYSTEM_PROMPT,
            user_message=user_message,
            tools=[CRITIQUE_ANSWER_TOOL],
            max_tokens=self.settings.claude_max_tokens_critic,
        )

        tool_output = json.loads(raw_response)

        issues = self._build_critic_issues(tool_output)

        critic_review = CriticReview(
            issues=issues,
            revised_confidence=float(tool_output.get("revised_confidence", synthesis.overall_confidence)),
            final_answer=tool_output.get("final_answer", synthesis.answer),
            critique_summary=tool_output.get("critique_summary", ""),
            passed=bool(tool_output.get("passed", True)),
        )

        # Write to scratchpad
        memory.scratchpad["critic"] = critic_review.model_dump(mode="json")

        self.log.info(
            "critic_done",
            passed=critic_review.passed,
            n_issues=len(issues),
            revised_confidence=critic_review.revised_confidence,
        )

        return AgentResult(
            agent_name=self.name,
            query=query,
            chunks=[],
            summary=critic_review.final_answer,
            confidence=critic_review.revised_confidence,
            tokens_used=tokens,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, query: ResearchQuery, synthesis: SynthesisResult) -> str:
        sections: list[str] = []

        sections.append(f"## Original Research Question\n{query.query}")
        sections.append(
            f"**Company:** {query.company_name or query.ticker} ({query.ticker})\n"
            f"**Original confidence:** {synthesis.overall_confidence:.2f}"
        )

        sections.append(f"## Synthesized Answer to Review\n{synthesis.answer}")

        # Include only the cited passages to keep the prompt bounded
        if synthesis.citations:
            sections.append("## Cited Passages")
            for citation in synthesis.citations:
                chunk = citation.chunk
                date_str = chunk.date.strftime("%Y-%m-%d") if chunk.date else "unknown date"
                title = chunk.source_title or chunk.source_url or "untitled"
                sections.append(
                    f"### {citation.citation_id} — {chunk.doc_type.value} | {date_str}\n"
                    f"**Source:** {title}\n"
                    f"**Quoted as:** \"{citation.quote}\"\n\n"
                    f"**Full passage:**\n{chunk.text}"
                )
        else:
            sections.append("## Cited Passages\n*(No citations provided — all claims are uncited.)*")

        return "\n\n".join(sections)

    def _build_critic_issues(self, tool_output: dict) -> list[CriticIssue]:
        issues: list[CriticIssue] = []
        for raw in tool_output.get("issues", []):
            issue_type = raw.get("issue_type", "ambiguous")
            severity = raw.get("severity", "low")

            if issue_type not in _VALID_ISSUE_TYPES:
                self.log.warning(
                    "critic_unknown_issue_type",
                    raw_type=issue_type,
                    fallback="ambiguous",
                )
                issue_type = "ambiguous"

            if severity not in _VALID_SEVERITIES:
                self.log.warning(
                    "critic_unknown_severity",
                    raw_severity=severity,
                    fallback="low",
                )
                severity = "low"

            issues.append(CriticIssue(
                claim=raw.get("claim", ""),
                issue_type=issue_type,
                explanation=raw.get("explanation", ""),
                severity=severity,
            ))
        return issues
