"""
ui/app.py

Streamlit frontend for the Multi-Agent Research Analyst.
Run with: streamlit run ui/app.py
"""

from __future__ import annotations

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Research Analyst",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, str, str | None]:
    """Renders sidebar inputs. Returns (api_url, ticker, company_name)."""
    st.sidebar.title("Configuration")
    api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
    ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()
    company_raw = st.sidebar.text_input("Company Name (optional)", value="")
    company_name = company_raw.strip() or None

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About**")
    st.sidebar.markdown(
        "Multi-agent RAG system that answers financial research questions "
        "by reasoning across SEC 10-K filings, earnings call transcripts, "
        "and news articles."
    )
    return api_url, ticker, company_name


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(api_url: str, payload: dict) -> dict:
    """POST /research and return parsed JSON. Raises on HTTP error."""
    url = f"{api_url.rstrip('/')}/research"
    with httpx.Client(timeout=300.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

def render_metric_row(confidence: float, latency_ms: int | None, passed: bool) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{confidence:.0%}")
    col2.metric("Latency", f"{latency_ms:,}ms" if latency_ms else "n/a")
    col3.metric("Critic", "✅ PASSED" if passed else "❌ FAILED")


def render_results(result: dict) -> None:
    render_metric_row(
        result["final_confidence"],
        result.get("total_latency_ms"),
        result.get("passed", True),
    )

    st.markdown("---")

    tab_answer, tab_findings, tab_citations, tab_issues, tab_agents = st.tabs(
        ["Answer", "Key Findings", "Citations", "Issues", "Agent Summaries"]
    )

    with tab_answer:
        st.markdown(result["final_answer"])

    with tab_findings:
        findings = result.get("key_findings", [])
        if findings:
            for f in findings:
                st.markdown(f"- {f}")
        else:
            st.info("No key findings available.")

    with tab_citations:
        citations = result.get("citations", [])
        if citations:
            rows = [
                {
                    "ID": c["citation_id"],
                    "Quote": c["quote"][:100] + ("…" if len(c["quote"]) > 100 else ""),
                    "Source": c.get("source_title") or "—",
                    "Type": c.get("doc_type", ""),
                    "Date": c.get("date", "")[:10] if c.get("date") else "—",
                    "Ticker": c.get("ticker", ""),
                }
                for c in citations
            ]
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No citations available.")

    with tab_issues:
        issues = result.get("issues", [])
        if not issues:
            st.success("No issues found — answer passed quality review.")
        else:
            severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            for issue in issues:
                icon = severity_color.get(issue["severity"], "⚪")
                with st.expander(
                    f"{icon} [{issue['severity'].upper()}] {issue['issue_type']} — "
                    f"{issue['claim'][:80]}{'…' if len(issue['claim']) > 80 else ''}"
                ):
                    st.markdown(f"**Claim:** {issue['claim']}")
                    st.markdown(f"**Type:** `{issue['issue_type']}`")
                    st.markdown(f"**Severity:** `{issue['severity']}`")
                    st.markdown(f"**Explanation:** {issue['explanation']}")

        if result.get("critique_summary"):
            st.markdown("---")
            st.markdown(f"**Critique Summary:** {result['critique_summary']}")

    with tab_agents:
        summaries = result.get("agent_summaries", {})
        if summaries:
            for agent_name, summary in summaries.items():
                with st.expander(agent_name.replace("_", " ").title(), expanded=False):
                    st.markdown(summary)
        else:
            st.info("No agent summaries available.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🔍 Multi-Agent Research Analyst")
    st.caption("Answers financial research questions across SEC filings, earnings calls, and news.")

    api_url, ticker, company_name = render_sidebar()

    query = st.text_area(
        "Research Question",
        placeholder="e.g. What are Apple's key growth drivers and risks for 2024?",
        height=120,
    )

    submitted = st.button("Submit", type="primary")

    if submitted and query.strip():
        payload = {
            "query": query.strip(),
            "ticker": ticker,
            "company_name": company_name,
        }

        with st.spinner("Running research pipeline… (this may take 1–2 minutes)"):
            try:
                result = call_api(api_url, payload)
                st.session_state["last_result"] = result
            except httpx.HTTPStatusError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text[:500]}")
                st.stop()
            except httpx.RequestError as e:
                st.error(f"Could not reach API at {api_url}: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                st.stop()

    if "last_result" in st.session_state:
        render_results(st.session_state["last_result"])


if __name__ == "__main__":
    main()
