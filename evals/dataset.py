"""
evals/dataset.py

Curated evaluation dataset — Q&A pairs with ground-truth expectations.
Each EvalItem captures what a correct answer should cover, not exact text,
so the eval metrics can assess coverage rather than string matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalItem:
    """A single evaluation question with expectations for scoring."""
    id: str
    query: str
    ticker: str
    company_name: str

    # Topics/keywords that should appear in a good answer
    expected_topics: list[str] = field(default_factory=list)

    # Claims that should be supported by sources (for faithfulness check)
    expected_claims: list[str] = field(default_factory=list)

    # Minimum acceptable confidence from the pipeline
    min_confidence: float = 0.4

    # Should the critic pass with no high-severity issues?
    should_pass_critic: bool = True

    notes: str = ""


# ---------------------------------------------------------------------------
# AAPL Eval Dataset
# ---------------------------------------------------------------------------

AAPL_EVALS: list[EvalItem] = [
    EvalItem(
        id="aapl_001",
        query="What are Apple's primary risk factors as disclosed in their 10-K filing?",
        ticker="AAPL",
        company_name="Apple Inc.",
        expected_topics=[
            "competition", "supply chain", "macroeconomic", "regulation",
            "intellectual property", "international operations",
        ],
        expected_claims=[
            "Apple faces competition in all of its markets",
            "Apple relies on third-party manufacturers",
        ],
        min_confidence=0.6,
        should_pass_critic=True,
        notes="Core risk factor question — should have strong SEC 10-K coverage.",
    ),
    EvalItem(
        id="aapl_002",
        query="What did Apple's management say about iPhone revenue trends in recent earnings calls?",
        ticker="AAPL",
        company_name="Apple Inc.",
        expected_topics=[
            "iPhone", "revenue", "growth", "services", "guidance",
        ],
        expected_claims=[
            "iPhone is Apple's largest revenue segment",
        ],
        min_confidence=0.5,
        should_pass_critic=True,
        notes="Management commentary — should lean on earnings call transcripts.",
    ),
    EvalItem(
        id="aapl_003",
        query="How has Apple's Services segment performed and what are its growth drivers?",
        ticker="AAPL",
        company_name="Apple Inc.",
        expected_topics=[
            "App Store", "Apple Music", "iCloud", "Apple TV+",
            "recurring revenue", "subscription", "margin",
        ],
        expected_claims=[
            "Services is Apple's fastest growing segment",
            "Services revenue carries higher margins than hardware",
        ],
        min_confidence=0.55,
        should_pass_critic=True,
        notes="Multi-source question: 10-K for segment reporting, earnings for commentary.",
    ),
    EvalItem(
        id="aapl_004",
        query="What is Apple's cash position and capital return program?",
        ticker="AAPL",
        company_name="Apple Inc.",
        expected_topics=[
            "buyback", "dividend", "cash", "balance sheet",
            "shareholder return", "debt",
        ],
        expected_claims=[
            "Apple returns capital to shareholders through buybacks and dividends",
        ],
        min_confidence=0.5,
        should_pass_critic=True,
        notes="Financial structure question — SEC filings primary source.",
    ),
    EvalItem(
        id="aapl_005",
        query="What are Apple's quantum computing revenue projections for 2030?",
        ticker="AAPL",
        company_name="Apple Inc.",
        expected_topics=[],  # No meaningful content expected
        expected_claims=[],
        min_confidence=0.0,   # Should be very low confidence
        should_pass_critic=False,  # Critic should flag unsupported claims
        notes="Thin-evidence adversarial case — pipeline should return low confidence.",
    ),
    EvalItem(
        id="aapl_006",
        query="What are the main legal and regulatory challenges Apple faces?",
        ticker="AAPL",
        company_name="Apple Inc.",
        expected_topics=[
            "antitrust", "App Store", "EU", "DOJ", "privacy",
            "litigation", "regulatory",
        ],
        expected_claims=[
            "Apple faces regulatory scrutiny over its App Store practices",
        ],
        min_confidence=0.5,
        should_pass_critic=True,
        notes="Regulatory landscape — should combine SEC risk factors with news.",
    ),
]


# ---------------------------------------------------------------------------
# Multi-ticker dataset (extend as more tickers are ingested)
# ---------------------------------------------------------------------------

ALL_EVALS: list[EvalItem] = AAPL_EVALS


def get_evals_for_ticker(ticker: str) -> list[EvalItem]:
    return [e for e in ALL_EVALS if e.ticker.upper() == ticker.upper()]


def get_eval_by_id(eval_id: str) -> EvalItem | None:
    return next((e for e in ALL_EVALS if e.id == eval_id), None)
