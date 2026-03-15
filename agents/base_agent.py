"""
agents/base_agent.py

Abstract base class that ALL agents must inherit from.
Enforces the agent contract: ResearchQuery + SharedMemory → AgentResult.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import anthropic
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import get_settings
from core.models import AgentName, AgentResult, ResearchQuery, SharedMemory

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Contract:
    - Subclasses implement `_run()` which does the actual work
    - `run()` wraps it with timing, error handling, and memory writes
    - All agents share one Anthropic client (set at class level)
    """

    name: AgentName  # Must be set by subclass

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        self.log = logger.bind(agent=self.name)

    def run(self, query: ResearchQuery, memory: SharedMemory) -> AgentResult:
        """
        Public entry point. Wraps _run() with timing and error handling.
        Always returns an AgentResult — never raises.
        """
        start = time.monotonic()
        self.log.info("agent_start", query=query.query[:80])

        try:
            result = self._run(query, memory)
        except Exception as exc:
            self.log.error("agent_error", error=str(exc), exc_info=True)
            result = AgentResult(
                agent_name=self.name,
                query=query,
                error=str(exc),
                confidence=0.0,
            )

        result.latency_ms = int((time.monotonic() - start) * 1000)
        self.log.info(
            "agent_done",
            confidence=result.confidence,
            chunks=len(result.chunks),
            latency_ms=result.latency_ms,
            error=result.error,
        )

        # Write summary to shared memory for downstream agents
        if result.summary:
            memory.agent_summaries[self.name] = result.summary
        if result.chunks:
            memory.add_chunks(self.name, result.chunks)

        return result

    @abstractmethod
    def _run(self, query: ResearchQuery, memory: SharedMemory) -> AgentResult:
        """Subclasses implement the actual retrieval + LLM call here."""
        ...

    # ------------------------------------------------------------------
    # Shared helper: call Claude with tool use and return parsed content
    # ------------------------------------------------------------------

    def _call_claude(
        self,
        system: str,
        user_message: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, int]:
        """
        Call Claude. Returns (text_response, tokens_used).
        If tools provided, returns the first tool_use block's input as JSON string.
        Automatically retries on rate-limit and timeout errors with exponential backoff.
        """
        max_tokens = max_tokens or self.settings.claude_max_tokens_retrieval

        kwargs: dict = dict(
            model=self.settings.claude_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        if tools:
            kwargs["tools"] = tools

        response = self._call_claude_with_retry(**kwargs)
        tokens = response.usage.input_tokens + response.usage.output_tokens

        # If tool use, return the tool input as JSON string
        for block in response.content:
            if block.type == "tool_use":
                import json
                return json.dumps(block.input), tokens

        # Otherwise return plain text
        text = " ".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        return text, tokens

    @retry(
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        )),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _call_claude_with_retry(self, **kwargs):
        """Inner call with tenacity retry — separated so the decorator applies cleanly."""
        return self.client.messages.create(**kwargs)
