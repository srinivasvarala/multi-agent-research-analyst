# Multi-Agent Research Analyst

> Ask any research question about a public company and get a cited, hallucination-checked answer synthesized from SEC filings, earnings call transcripts, and news — produced by a team of specialized AI agents.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Anthropic Claude](https://img.shields.io/badge/powered%20by-Claude%20Sonnet-orange)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## What It Does

```
User: "What are Apple's biggest competitive risks based on their 2023 10-K and recent earnings calls?"

→ OrchestratorAgent   decomposes query into sub-tasks, routes to specialist agents
→ SecFilingsAgent      retrieves from 10-K/10-Q using hybrid dense + sparse search
→ EarningsCallAgent   retrieves from earnings call transcript corpus
→ NewsAgent            retrieves from recent news articles
→ SynthesisAgent       merges multi-source context, writes a cited answer
→ CriticAgent          checks for unsupported claims, assigns a confidence score

← Final report: structured JSON + markdown with page-level citations
```

## Architecture

```
┌─────────────────────┐
│  OrchestratorAgent  │  Query decomposition + dynamic routing (Claude tool use)
└──────────┬──────────┘
           │
    ┌──────┼──────────────┐
    ▼      ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ SecFilings   │  │ EarningsCall │  │    News      │
│   Agent      │  │   Agent      │  │   Agent      │
│  (RAG: 10-K) │  │  (RAG: XLST) │  │  (RAG: API)  │
└──────────────┘  └──────────────┘  └──────────────┘
       │                 │                  │
       └────────┬────────┘                  │
                ▼                           │
        ┌─────────────┐◄───────────────────┘
        │  Synthesis  │  Multi-source grounding, citation assembly
        │   Agent     │
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │   Critic    │  Hallucination detection, confidence scoring
        │   Agent     │
        └─────────────┘
               │
               ▼
      ResearchReport (JSON + Markdown)
```

## Technical Highlights

**Multi-agent orchestration**
- Agents coordinate via a shared scratchpad (`SharedMemory`) and communicate through typed Pydantic contracts — no raw dicts passed between agents
- Orchestrator uses Claude's native tool use to dynamically route queries and merge results

**Hybrid RAG retrieval pipeline (implemented from scratch — no LangChain)**
- Dense retrieval: Voyage AI `voyage-3` embeddings → ChromaDB cosine similarity (top-20)
- Sparse retrieval: BM25 (`rank-bm25`) keyword search (top-20)
- Fusion: Reciprocal Rank Fusion (RRF) → cross-encoder reranking → final top-5 chunks
- Chunk strategy: 512-token semantic chunks, 64-token overlap, per-chunk metadata (ticker, doc_type, date, page_number, source_url)

**Structured outputs via tool use**
- Every agent returns structured data via Claude's tool use API — no brittle regex or JSON parsing of free-form text
- Full output model: `ResearchQuery → AgentResult[] → SynthesisResult → CriticReview → ResearchReport`

**Self-critique layer**
- A dedicated `CriticAgent` reviews the synthesis, flags unsupported claims, and outputs a revised confidence score — reducing hallucination risk in the final answer

**Evaluation harness**
- RAGAS-style metrics: faithfulness, answer relevance, context precision, citation accuracy
- Designed to catch retrieval and synthesis regressions across agent updates

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| LLM | Anthropic Claude Sonnet | Tool use for structured outputs |
| Embeddings | Voyage AI `voyage-3` | 1024-dim, outperforms OpenAI on retrieval |
| Vector DB | ChromaDB | Persistent, one collection per corpus |
| Sparse Search | BM25 (`rank-bm25`) | Complements dense for keyword-heavy queries |
| Reranker | Cross-encoder | Final precision boost before agent sees chunks |
| Document Parsing | PyMuPDF | PDF → text with page metadata |
| API | FastAPI + uvicorn | Async, streaming-ready |
| UI | Streamlit | Interactive query interface |
| Config | pydantic-settings | Typed env var management |
| Logging | structlog | Structured JSON logs for observability |

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/multi-agent-research-analyst
cd multi-agent-research-analyst
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your keys:
#   ANTHROPIC_API_KEY  — required
#   VOYAGE_API_KEY     — required (free tier at voyageai.com)
#   NEWS_API_KEY       — optional (newsapi.org free tier)
```

### 3. Ingest a company's filings

```bash
python scripts/ingest_sec.py --ticker AAPL --year 2023
```

### 4. Run a query

```bash
# Phase 1 (SEC filings only)
python -m agents.sec_filings_agent \
  --ticker AAPL \
  --query "What are Apple's main competitive risks?"
```

### 5. Start the full API + UI

```bash
uvicorn api.main:app --reload   # Terminal 1
streamlit run ui/app.py         # Terminal 2
```

## Project Structure

```
multi-agent-research-analyst/
├── agents/                ← One file per agent (OrchestratorAgent, SecFilingsAgent, ...)
├── core/
│   ├── models.py          ← Typed Pydantic contracts (ResearchQuery → ResearchReport)
│   ├── config.py          ← All config via pydantic-settings
│   ├── memory.py          ← SharedMemory scratchpad
│   └── retrieval/         ← ChromaDB, hybrid search (BM25 + dense), cross-encoder reranker
├── ingestion/             ← PDF loading, semantic chunking, Voyage AI embedding
├── api/                   ← FastAPI REST layer (async)
├── ui/                    ← Streamlit frontend
├── evals/                 ← RAGAS-style evaluation harness
└── scripts/               ← Ingestion CLI scripts for SEC, transcripts, news
```

## Data Sources

All sources are free and publicly available:
- **SEC EDGAR** — 10-K and 10-Q filings
- **Earnings transcripts** — public investor relations pages
- **News** — NewsAPI free tier

## Why This Architecture

This project deliberately avoids LangChain and LlamaIndex to implement RAG patterns from first principles. The goal is to demonstrate a clear understanding of what these frameworks abstract away — embedding pipelines, vector stores, hybrid search fusion, and agent coordination — rather than relying on framework magic.

The multi-agent design reflects production patterns: agents are decoupled, have typed contracts, write intermediate reasoning to a shared scratchpad, and never raise uncaught exceptions. The Critic agent reflects real system design thinking — grounding and self-verification are as important as retrieval quality.

## License

MIT
