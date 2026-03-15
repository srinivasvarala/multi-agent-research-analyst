"""
api/routes/stream.py

POST /research/stream — Server-Sent Events endpoint that streams pipeline
progress as each agent stage completes.

SSE event format (each line):
    data: {"event": "<name>", "data": {...}}\n\n

Events emitted (in order):
    started          — pipeline kicked off
    orchestrator_done — query decomposed, sub-agents routed
    agent_done        — one specialist agent finished (emitted per agent)
    synthesis_done    — SynthesisAgent finished
    critic_done       — CriticAgent finished
    complete          — full ResearchResponse ready
    error             — unhandled exception
"""

from __future__ import annotations

import asyncio
import json
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

import structlog
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agents.orchestrator import OrchestratorAgent
from api.schemas import ResearchRequest, report_to_response
from core.models import ResearchQuery, SharedMemory

logger = structlog.get_logger()
router = APIRouter()

_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Format a single SSE message."""
    payload = json.dumps({"event": event, "data": data})
    return f"data: {payload}\n\n"


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------

async def _pipeline_event_stream(request: ResearchRequest) -> AsyncIterator[str]:
    """
    Async generator that yields SSE strings as each pipeline stage completes.
    Delegates to run_full_pipeline via an event_callback, eliminating duplicate
    pipeline logic.
    """
    loop = asyncio.get_running_loop()
    log = logger.bind(ticker=request.ticker, query=request.query[:80])

    rq = ResearchQuery(
        query=request.query,
        ticker=request.ticker.upper(),
        company_name=request.company_name,
    )

    q: queue.SimpleQueue = queue.SimpleQueue()

    def callback(event: str, data: dict) -> None:
        q.put((event, data))

    def run() -> None:
        try:
            report, _ = OrchestratorAgent().run_full_pipeline(rq, event_callback=callback)
            q.put(("complete", report_to_response(report).model_dump()))
        except Exception as exc:
            log.error("stream_pipeline_error", error=str(exc), exc_info=True)
            q.put(("error", {"message": str(exc)}))
        finally:
            q.put(None)  # sentinel

    loop.run_in_executor(_executor, run)

    yield _sse("started", {"message": "Pipeline started", "ticker": rq.ticker})

    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is None:
            break
        event, data = item
        yield _sse(event, data)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/research/stream")
async def stream_research(request: ResearchRequest) -> StreamingResponse:
    """
    Stream pipeline progress via Server-Sent Events.
    The client receives one SSE message per pipeline stage, ending with 'complete'.
    """
    logger.info("stream_request_received", ticker=request.ticker, query=request.query[:80])

    return StreamingResponse(
        _pipeline_event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx buffering if behind a proxy
        },
    )
