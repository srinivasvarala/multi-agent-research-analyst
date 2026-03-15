"""
api/routes/research.py

POST /research — runs the full multi-agent pipeline and returns a ResearchResponse.

The pipeline (OrchestratorAgent → SynthesisAgent → CriticAgent) is synchronous.
It's wrapped in run_in_executor so it doesn't block the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import functools

import structlog
from fastapi import APIRouter, HTTPException

from agents.orchestrator import OrchestratorAgent
from api.schemas import ResearchRequest, ResearchResponse, report_to_response
from core.models import ResearchQuery

logger = structlog.get_logger()
router = APIRouter()


@router.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """
    Run the full research pipeline for a given query and ticker.

    Wraps the synchronous OrchestratorAgent.run_full_pipeline() in a threadpool
    so it doesn't block the event loop during Anthropic API calls.
    """
    log = logger.bind(ticker=request.ticker, query=request.query[:80])
    log.info("research_request_received")

    rq = ResearchQuery(
        query=request.query,
        ticker=request.ticker.upper(),
        company_name=request.company_name,
    )

    try:
        loop = asyncio.get_running_loop()
        orchestrator = OrchestratorAgent()
        report, _memory = await loop.run_in_executor(
            None,
            functools.partial(orchestrator.run_full_pipeline, rq),
        )
    except Exception as exc:
        log.error("research_pipeline_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    log.info(
        "research_request_done",
        final_confidence=report.final_confidence,
        latency_ms=report.total_latency_ms,
    )

    return report_to_response(report)
