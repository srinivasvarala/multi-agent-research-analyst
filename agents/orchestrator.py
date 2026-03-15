"""
agents/orchestrator.py

OrchestratorAgent — decomposes a research query into sub-queries and routes
each to the appropriate specialist agent (SecFilings, EarningsCall, News).
Does NOT do retrieval itself.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import structlog

from agents.base_agent import BaseAgent
from core.config import get_settings
from core.models import (
    AgentName,
    AgentResult,
    CriticReview,
    OrchestratorPlan,
    ResearchQuery,
    ResearchReport,
    SharedMemory,
    SubQuery,
    SynthesisResult,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a research orchestrator that decomposes complex financial research questions
into targeted sub-queries, each routed to the most appropriate specialist agent.

Available specialist agents:
- **sec_filings**: Retrieves information from SEC 10-K and 10-Q filings. Best for: risk factors,
  financial statements, business description, management discussion and analysis, regulatory filings.
- **earnings_call**: Retrieves information from quarterly earnings call transcripts. Best for:
  management commentary, forward guidance, analyst Q&A, tone and sentiment from executives.
- **news**: Retrieves information from recent news articles. Best for: recent events, market reaction,
  analyst coverage, competitive developments, macroeconomic context.

Rules:
1. Decompose the user query into 1-3 focused sub-queries
2. Each sub-query should target exactly ONE agent
3. Sub-queries should be specific enough that the target agent can retrieve relevant passages
4. Do not create redundant sub-queries — each should address a distinct aspect of the question
5. Always use the decompose_query tool — never respond in free text
"""

# ---------------------------------------------------------------------------
# Tool Definition
# ---------------------------------------------------------------------------

DECOMPOSE_QUERY_TOOL = {
    "name": "decompose_query",
    "description": "Decompose a research question into targeted sub-queries for specialist agents",
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sub_query": {
                            "type": "string",
                            "description": "The focused sub-question for this agent"
                        },
                        "target_agent": {
                            "type": "string",
                            "enum": ["sec_filings", "earnings_call", "news"],
                            "description": "Which specialist agent should handle this sub-query"
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this sub-query belongs to this agent"
                        }
                    },
                    "required": ["sub_query", "target_agent", "rationale"]
                },
                "description": "List of 1-3 targeted sub-queries"
            },
            "reasoning": {
                "type": "string",
                "description": "Overall reasoning for how the original query was decomposed"
            }
        },
        "required": ["sub_queries", "reasoning"]
    }
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OrchestratorAgent(BaseAgent):
    """
    Decomposes a research query into sub-queries and routes each to the
    appropriate specialist agent. Collects and summarizes all results.
    """

    name = AgentName.ORCHESTRATOR

    # Lazy-loaded class-level registry to avoid circular imports
    _AGENT_CLASS_MAP: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()

    @classmethod
    def _get_agent_class_map(cls) -> dict:
        """Lazily import specialist agents to avoid circular imports."""
        if not cls._AGENT_CLASS_MAP:
            from agents.sec_filings_agent import SecFilingsAgent
            from agents.earnings_call_agent import EarningsCallAgent
            from agents.news_agent import NewsAgent
            cls._AGENT_CLASS_MAP = {
                AgentName.SEC_FILINGS: SecFilingsAgent,
                AgentName.EARNINGS_CALL: EarningsCallAgent,
                AgentName.NEWS: NewsAgent,
            }
        return cls._AGENT_CLASS_MAP

    def _decompose_query(self, query: ResearchQuery) -> OrchestratorPlan:
        """Call Claude to decompose the query into sub-queries."""
        user_message = f"""Research Question: {query.query}
Company: {query.company_name or query.ticker} ({query.ticker})

Please decompose this question into targeted sub-queries using the decompose_query tool.
"""
        raw_response, _ = self._call_claude(
            system=SYSTEM_PROMPT,
            user_message=user_message,
            tools=[DECOMPOSE_QUERY_TOOL],
            max_tokens=512,
        )

        plan_data = json.loads(raw_response)

        sub_queries = []
        for sq in plan_data.get("sub_queries", []):
            raw_agent = sq.get("target_agent", "sec_filings")
            try:
                target_agent = AgentName(raw_agent)
            except ValueError:
                self.log.warning(
                    "unknown_target_agent",
                    raw_agent=raw_agent,
                    fallback=AgentName.SEC_FILINGS.value,
                )
                target_agent = AgentName.SEC_FILINGS

            sub_queries.append(SubQuery(
                sub_query=sq["sub_query"],
                target_agent=target_agent,
                rationale=sq.get("rationale", ""),
            ))

        return OrchestratorPlan(
            original_query=query,
            sub_queries=sub_queries,
            reasoning=plan_data.get("reasoning", ""),
        )

    def _run(self, query: ResearchQuery, memory: SharedMemory) -> AgentResult:
        # 1. Decompose query into sub-queries
        plan = self._decompose_query(query)
        self.log.info(
            "query_decomposed",
            n_sub_queries=len(plan.sub_queries),
            reasoning=plan.reasoning[:120],
        )

        # 2. Write orchestrator plan to scratchpad (use .value strings for serializability)
        memory.scratchpad["orchestrator"] = {
            "plan_reasoning": plan.reasoning,
            "sub_queries": [
                {"sub_query": sq.sub_query, "target": sq.target_agent.value}
                for sq in plan.sub_queries
            ],
        }

        # 3. Run each sub-query through the target specialist agent (in parallel)
        agent_class_map = self._get_agent_class_map()

        def _run_sub_query(sq: SubQuery) -> AgentResult | None:
            agent_cls = agent_class_map.get(sq.target_agent)
            if agent_cls is None:
                self.log.warning("no_agent_class", target=sq.target_agent.value)
                return None
            sub_rq = ResearchQuery(
                query=sq.sub_query,
                ticker=query.ticker,
                company_name=query.company_name,
                date_from=query.date_from,
                date_to=query.date_to,
                session_id=query.session_id,
            )
            self.log.info(
                "routing_sub_query",
                target=sq.target_agent.value,
                sub_query=sq.sub_query[:80],
            )
            return agent_cls().run(sub_rq, memory)

        # Use ThreadPoolExecutor so all 3 specialist agents run concurrently.
        # Thread safety: each agent writes only to its own key in SharedMemory.
        result_map: dict[SubQuery, AgentResult] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_run_sub_query, sq): sq for sq in plan.sub_queries}
            for future in as_completed(futures):
                sq = futures[future]
                result = future.result()
                if result is not None:
                    result_map[sq] = result

        # Preserve original ordering
        sub_results: list[AgentResult] = [
            result_map[sq] for sq in plan.sub_queries if sq in result_map
        ]

        # 4. Compute summary and average confidence
        successful = [r for r in sub_results if not r.error and r.confidence > 0]
        avg_confidence = (
            sum(r.confidence for r in successful) / len(successful)
            if successful else 0.0
        )

        routing_summary = self._build_routing_summary(plan, sub_results)

        # Store for run_full_pipeline to access after _run() completes
        self._last_plan = plan
        self._sub_results = sub_results

        return AgentResult(
            agent_name=self.name,
            query=query,
            chunks=[],  # Orchestrator doesn't retrieve directly
            summary=routing_summary,
            confidence=avg_confidence,
        )

    def _build_routing_summary(
        self,
        plan: OrchestratorPlan,
        sub_results: list[AgentResult],
    ) -> str:
        lines = [f"**Orchestration Plan:** {plan.reasoning}\n"]
        for sq, result in zip(plan.sub_queries, sub_results):
            status = "error" if result.error else f"confidence {result.confidence:.0%}"
            lines.append(
                f"- **{sq.target_agent.value}**: {sq.sub_query[:80]} → {status}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public pipeline entry point (used by FastAPI in Phase 4)
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        query: ResearchQuery,
        memory: SharedMemory | None = None,
        event_callback: Callable[[str, dict], None] | None = None,
    ) -> tuple[ResearchReport, SharedMemory]:
        """
        Full pipeline: Orchestrator → 3 Specialists → SynthesisAgent → CriticAgent.
        Returns a ResearchReport and the SharedMemory for inspection.

        event_callback(event_name, data_dict) is called after each major stage,
        allowing the streaming route to enqueue SSE events without duplicating logic.
        """
        import time
        pipeline_start = time.monotonic()

        if memory is None:
            memory = SharedMemory(session_id=query.session_id, query=query)

        # Step 1: Orchestrator + specialist agents
        orch_result = self.run(query, memory)
        if event_callback:
            event_callback("orchestrator_done", {
                "summary": (orch_result.summary or "")[:300],
                "confidence": orch_result.confidence,
                "latency_ms": orch_result.latency_ms,
            })
            for agent_name, summary in memory.agent_summaries.items():
                chunks = memory.retrieved_chunks_by_agent.get(agent_name, [])
                event_callback("agent_done", {
                    "agent": agent_name.value,
                    "n_chunks": len(chunks),
                    "summary_preview": summary[:200],
                })

        # Step 2: Synthesis
        from agents.synthesis_agent import SynthesisAgent
        synth_result = SynthesisAgent().run(query, memory)
        if event_callback:
            event_callback("synthesis_done", {
                "confidence": synth_result.confidence,
                "latency_ms": synth_result.latency_ms,
                "error": synth_result.error,
            })

        # Step 3: Critic
        from agents.critic_agent import CriticAgent
        critic_result = CriticAgent().run(query, memory)
        if event_callback:
            raw_critic = memory.scratchpad.get("critic", {})
            event_callback("critic_done", {
                "passed": raw_critic.get("passed", True),
                "n_issues": len(raw_critic.get("issues", [])),
                "revised_confidence": raw_critic.get("revised_confidence"),
                "latency_ms": critic_result.latency_ms,
            })

        # Reconstruct typed results from scratchpad
        synthesis: SynthesisResult | None = None
        raw_synthesis = memory.scratchpad.get("synthesis")
        if raw_synthesis:
            try:
                synthesis = SynthesisResult.model_validate(raw_synthesis)
            except Exception as exc:
                self.log.error("pipeline_synthesis_parse_error", error=str(exc))

        critic_review: CriticReview | None = None
        raw_critic = memory.scratchpad.get("critic")
        if raw_critic:
            try:
                critic_review = CriticReview.model_validate(raw_critic)
            except Exception as exc:
                self.log.error("pipeline_critic_parse_error", error=str(exc))

        # Build agent_results dict keyed by AgentName
        last_plan = getattr(self, "_last_plan", None)
        sub_results = getattr(self, "_sub_results", [])
        agent_results: dict[AgentName, AgentResult] = {}
        if last_plan is not None:
            for sq, result in zip(last_plan.sub_queries, sub_results):
                agent_results[sq.target_agent] = result

        total_latency_ms = int((time.monotonic() - pipeline_start) * 1000)

        report = ResearchReport(
            query=query,
            orchestrator_plan=last_plan,
            agent_results=agent_results,
            synthesis=synthesis,
            critic_review=critic_review,
            total_latency_ms=total_latency_ms,
        )

        return report, memory


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full Phase 3 research pipeline")
    parser.add_argument("--query", required=True, help="Research question")
    parser.add_argument("--ticker", default="AAPL", help="Company ticker")
    parser.add_argument("--company", default=None, help="Full company name")
    args = parser.parse_args()

    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    from rich import box

    console = Console()

    rq = ResearchQuery(
        query=args.query,
        ticker=args.ticker.upper(),
        company_name=args.company,
    )

    orchestrator = OrchestratorAgent()
    report, mem = orchestrator.run_full_pipeline(rq)

    # Panel 1: Final answer + top-level metrics
    confidence_pct = f"{report.final_confidence:.0%}"
    latency = f"{report.total_latency_ms}ms" if report.total_latency_ms else "n/a"
    console.print(Panel(
        Markdown(report.final_answer),
        title=f"[bold green]Final Answer — {args.ticker.upper()}[/bold green]",
        subtitle=f"Confidence: {confidence_pct} | Latency: {latency}",
    ))

    # Panel 2: Critic review
    if report.critic_review:
        cr = report.critic_review
        passed_label = "[bold green]PASSED[/bold green]" if cr.passed else "[bold red]FAILED[/bold red]"
        header = f"Critic Review — {passed_label}\n{cr.critique_summary}"

        issue_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        issue_table.add_column("Severity", style="bold")
        issue_table.add_column("Type")
        issue_table.add_column("Claim", max_width=40)
        issue_table.add_column("Explanation", max_width=60)

        severity_colors = {"high": "red", "medium": "yellow", "low": "cyan"}
        for issue in cr.issues:
            color = severity_colors.get(issue.severity, "white")
            issue_table.add_row(
                f"[{color}]{issue.severity}[/{color}]",
                issue.issue_type,
                issue.claim[:80],
                issue.explanation[:120],
            )

        from rich.console import Group
        from rich.text import Text
        content = Group(Text(header), issue_table) if cr.issues else Text(header)
        console.print(Panel(
            content,
            title="[bold yellow]Critic Review[/bold yellow]",
            subtitle=f"Revised confidence: {cr.revised_confidence:.0%}",
        ))

    # Panel 3: Key findings
    if report.synthesis and report.synthesis.key_findings:
        findings_md = "\n".join(f"- {f}" for f in report.synthesis.key_findings)
        console.print(Panel(
            Markdown(findings_md),
            title="[bold blue]Key Findings[/bold blue]",
        ))

    # Panel 4+: Per-agent summaries
    for agent_name, agent_result in report.agent_results.items():
        chunks = mem.retrieved_chunks_by_agent.get(agent_name, [])
        status = "error" if agent_result.error else f"confidence {agent_result.confidence:.0%}"
        console.print(Panel(
            Markdown(agent_result.summary or agent_result.error or "(no output)"),
            title=f"[bold]{agent_name.value}[/bold]",
            subtitle=f"{status} | Chunks: {len(chunks)}",
        ))
