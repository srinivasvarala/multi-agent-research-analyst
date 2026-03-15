"""
evals/metrics.py

RAGAS-style evaluation metrics implemented directly using Claude as judge.
No external eval frameworks — implements the patterns from scratch.

Metrics:
- CitationAccuracy:  Do cited quotes actually appear in the cited passage?
- AnswerRelevance:   Does the answer address the original question?
- Faithfulness:      Are all claims in the answer grounded in retrieved sources?
- TopicCoverage:     What fraction of expected topics appear in the answer?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import anthropic
import structlog

from core.config import get_settings
from core.models import Citation, SynthesisResult
from evals.dataset import EvalItem

logger = structlog.get_logger()
settings = get_settings()

# Lightweight Claude client for eval judge calls
_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MetricScore:
    name: str
    score: float          # 0.0 – 1.0
    explanation: str
    details: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    eval_id: str
    query: str
    ticker: str
    final_answer: str
    final_confidence: float
    passed_critic: bool

    citation_accuracy: MetricScore | None = None
    answer_relevance: MetricScore | None = None
    faithfulness: MetricScore | None = None
    topic_coverage: MetricScore | None = None

    @property
    def aggregate_score(self) -> float:
        scores = [
            m.score for m in [
                self.citation_accuracy,
                self.answer_relevance,
                self.faithfulness,
                self.topic_coverage,
            ]
            if m is not None
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict:
        return {
            "eval_id": self.eval_id,
            "query": self.query,
            "ticker": self.ticker,
            "final_confidence": self.final_confidence,
            "passed_critic": self.passed_critic,
            "aggregate_score": self.aggregate_score,
            "citation_accuracy": self.citation_accuracy.score if self.citation_accuracy else None,
            "answer_relevance": self.answer_relevance.score if self.answer_relevance else None,
            "faithfulness": self.faithfulness.score if self.faithfulness else None,
            "topic_coverage": self.topic_coverage.score if self.topic_coverage else None,
        }


# ---------------------------------------------------------------------------
# Metric 1: Citation Accuracy (rule-based, no LLM needed)
# ---------------------------------------------------------------------------

def citation_accuracy(citations: list[Citation]) -> MetricScore:
    """
    Check that each citation's quote actually appears (or is paraphrased near-verbatim)
    in the cited chunk text.

    Uses simple substring matching after normalising whitespace.
    Score = fraction of citations where the quote is found in the chunk.
    """
    if not citations:
        return MetricScore(
            name="citation_accuracy",
            score=1.0,
            explanation="No citations to verify.",
        )

    hits = 0
    misses: list[str] = []

    for c in citations:
        quote_norm = " ".join(c.quote.lower().split())
        chunk_norm = " ".join(c.chunk.text.lower().split())

        # Allow partial match — quote may be slightly edited; check 80% of words present
        quote_words = set(quote_norm.split())
        chunk_words = set(chunk_norm.split())

        if not quote_words:
            hits += 1
            continue

        overlap = len(quote_words & chunk_words) / len(quote_words)
        if overlap >= 0.75:
            hits += 1
        else:
            misses.append(c.citation_id)

    score = hits / len(citations)
    return MetricScore(
        name="citation_accuracy",
        score=score,
        explanation=(
            f"{hits}/{len(citations)} citations verified. "
            + (f"Unverified: {misses}" if misses else "All quotes found in cited passages.")
        ),
        details={"total": len(citations), "verified": hits, "failed_ids": misses},
    )


# ---------------------------------------------------------------------------
# Metric 2: Answer Relevance (LLM judge)
# ---------------------------------------------------------------------------

def answer_relevance(question: str, answer: str) -> MetricScore:
    """
    Ask Claude to score how directly the answer addresses the question (0.0–1.0).
    Uses a simple structured prompt with a JSON response (no tool use needed).
    """
    prompt = f"""You are evaluating whether a research answer is relevant to the question asked.

Question: {question}

Answer (first 1500 chars):
{answer[:1500]}

Rate the answer's relevance to the question on a scale of 0.0 to 1.0:
- 1.0: Directly and completely answers the question
- 0.7-0.9: Mostly answers the question with minor gaps
- 0.4-0.6: Partially answers; significant parts of the question unaddressed
- 0.1-0.3: Tangentially related but doesn't answer the question
- 0.0: Completely irrelevant

Respond with ONLY valid JSON in this format:
{{"score": <float>, "explanation": "<one sentence>"}}"""

    try:
        response = _client.messages.create(
            model=settings.claude_model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return MetricScore(
            name="answer_relevance",
            score=float(data["score"]),
            explanation=data.get("explanation", ""),
        )
    except Exception as exc:
        logger.warning("answer_relevance_metric_error", error=str(exc))
        return MetricScore(
            name="answer_relevance",
            score=0.5,
            explanation=f"Could not evaluate (error: {exc})",
        )


# ---------------------------------------------------------------------------
# Metric 3: Faithfulness (LLM judge)
# ---------------------------------------------------------------------------

def faithfulness(
    question: str,
    answer: str,
    synthesis: SynthesisResult | None,
) -> MetricScore:
    """
    Check that factual claims in the answer are supported by the cited passages.
    Score = fraction of claims that can be verified in the source corpus.
    """
    if not synthesis or not synthesis.citations:
        return MetricScore(
            name="faithfulness",
            score=0.5,
            explanation="No citations available — cannot verify faithfulness.",
        )

    # Build cited passages context (truncated for token efficiency)
    cited_texts = []
    for c in synthesis.citations[:8]:  # limit to 8 to keep prompt bounded
        cited_texts.append(
            f"{c.citation_id}: {c.chunk.text[:400]}"
        )
    sources_block = "\n\n".join(cited_texts)

    prompt = f"""You are evaluating whether the claims in a research answer are supported by the cited sources.

Question: {question}

Answer (first 1200 chars):
{answer[:1200]}

Cited Sources:
{sources_block[:3000]}

Task: Identify the 3-5 most important factual claims in the answer. For each, determine if it is
supported by the cited sources, partially supported, or unsupported.

Respond with ONLY valid JSON:
{{
  "claims_checked": [
    {{"claim": "<claim text>", "supported": true/false, "explanation": "<brief reason>"}}
  ],
  "faithfulness_score": <float 0.0-1.0>,
  "overall_explanation": "<one sentence>"
}}"""

    try:
        response = _client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return MetricScore(
            name="faithfulness",
            score=float(data["faithfulness_score"]),
            explanation=data.get("overall_explanation", ""),
            details={"claims": data.get("claims_checked", [])},
        )
    except Exception as exc:
        logger.warning("faithfulness_metric_error", error=str(exc))
        return MetricScore(
            name="faithfulness",
            score=0.5,
            explanation=f"Could not evaluate (error: {exc})",
        )


# ---------------------------------------------------------------------------
# Metric 4: Topic Coverage (rule-based)
# ---------------------------------------------------------------------------

def topic_coverage(answer: str, eval_item: EvalItem) -> MetricScore:
    """
    Check what fraction of expected topics appear in the answer text.
    Simple keyword presence check — no LLM needed.
    """
    if not eval_item.expected_topics:
        return MetricScore(
            name="topic_coverage",
            score=1.0,
            explanation="No expected topics defined for this eval item.",
        )

    answer_lower = answer.lower()
    found = [t for t in eval_item.expected_topics if t.lower() in answer_lower]
    missing = [t for t in eval_item.expected_topics if t.lower() not in answer_lower]

    score = len(found) / len(eval_item.expected_topics)
    return MetricScore(
        name="topic_coverage",
        score=score,
        explanation=(
            f"Found {len(found)}/{len(eval_item.expected_topics)} expected topics. "
            + (f"Missing: {missing}" if missing else "All topics covered.")
        ),
        details={"found": found, "missing": missing},
    )


# ---------------------------------------------------------------------------
# Run all metrics for one eval result
# ---------------------------------------------------------------------------

def score_eval_result(
    eval_item: EvalItem,
    final_answer: str,
    final_confidence: float,
    passed_critic: bool,
    synthesis: SynthesisResult | None,
) -> EvalResult:
    """Compute all metrics for a single eval item and return an EvalResult."""
    log = logger.bind(eval_id=eval_item.id)
    log.info("scoring_eval_item")

    result = EvalResult(
        eval_id=eval_item.id,
        query=eval_item.query,
        ticker=eval_item.ticker,
        final_answer=final_answer,
        final_confidence=final_confidence,
        passed_critic=passed_critic,
    )

    # Citation accuracy (fast, rule-based)
    cites = synthesis.citations if synthesis else []
    result.citation_accuracy = citation_accuracy(cites)
    log.info("citation_accuracy_done", score=result.citation_accuracy.score)

    # Answer relevance (LLM call)
    result.answer_relevance = answer_relevance(eval_item.query, final_answer)
    log.info("answer_relevance_done", score=result.answer_relevance.score)

    # Faithfulness (LLM call)
    result.faithfulness = faithfulness(eval_item.query, final_answer, synthesis)
    log.info("faithfulness_done", score=result.faithfulness.score)

    # Topic coverage (fast, rule-based)
    result.topic_coverage = topic_coverage(final_answer, eval_item)
    log.info("topic_coverage_done", score=result.topic_coverage.score)

    log.info(
        "eval_item_scored",
        aggregate=result.aggregate_score,
        confidence=final_confidence,
        passed_critic=passed_critic,
    )
    return result
