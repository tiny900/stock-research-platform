"""
AI-Powered Stock Research Platform — Streamlit Web UI
======================================================
Interactive web interface with 5 tabs:
1. Analysis Dashboard — Charts + indicator cards
2. RAG Knowledge Context — Retrieved documents
3. Generated Report — Full markdown report
4. Prompt Engineering Lab — Compare prompt versions
5. System Metrics — Performance and evaluation data

Usage:
    streamlit run app.py
"""

import json
import os
import time
from datetime import datetime

import streamlit as st

# Page config
st.set_page_config(
    page_title="AI Stock Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Global CSS
# ============================================================

st.markdown("""
<style>
/* ---- colour tokens ---- */
:root {
    --accent: #4f8ff7;
    --accent-light: #e8f0fe;
    --green: #22c55e;
    --red: #ef4444;
    --amber: #f59e0b;
    --surface: #f8fafc;
    --border: #e2e8f0;
}

/* ---- sidebar polish ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: .55rem 0 !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #3b7be8 !important;
}

/* ---- metric cards ---- */
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
div[data-testid="stMetric"] label {
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: .03em;
    font-size: .72rem !important;
    color: #64748b !important;
}

/* ---- tabs ---- */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: .88rem !important;
    padding: 10px 20px !important;
}

/* ---- expander headers ---- */
details summary span {
    font-weight: 600 !important;
}

/* ---- overall signal banner ---- */
.signal-banner {
    border-radius: 12px;
    padding: 16px 24px;
    margin: 12px 0 4px;
    font-size: 1.15rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}
.signal-banner.bullish  { background: #dcfce7; color: #166534; border-left: 5px solid var(--green); }
.signal-banner.bearish  { background: #fee2e2; color: #991b1b; border-left: 5px solid var(--red); }
.signal-banner.neutral  { background: #fef9c3; color: #854d0e; border-left: 5px solid var(--amber); }

/* ---- RAG source card ---- */
.rag-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.rag-card .score-badge {
    display: inline-block;
    background: var(--accent-light);
    color: var(--accent);
    font-weight: 700;
    font-size: .78rem;
    padding: 2px 10px;
    border-radius: 20px;
}

/* ---- prompt version card ---- */
.prompt-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    height: 100%;
}
.prompt-card h4 { margin: 0 0 4px; }
.prompt-card .tag {
    display: inline-block;
    background: var(--accent-light);
    color: var(--accent);
    font-size: .72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    margin-right: 4px;
}

/* ---- report container ---- */
.report-container {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px 40px;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
}

/* ---- hide default header anchor links ---- */
h1 a, h2 a, h3 a { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Helper Functions
# ============================================================

@st.cache_data(ttl=300)
def get_stock_data(symbol: str, days: int) -> dict:
    from tools.builtin_tools import StockDataRetrieverTool
    return json.loads(StockDataRetrieverTool()._run(symbol, days))

@st.cache_data(ttl=300)
def get_data_stats(raw_json: str) -> dict:
    from tools.builtin_tools import DataProcessorTool
    t = DataProcessorTool()
    return {
        "summary": json.loads(t._run(raw_json, "summary_stats")),
        "returns": json.loads(t._run(raw_json, "returns")),
        "volatility": json.loads(t._run(raw_json, "volatility")),
    }

@st.cache_data(ttl=300)
def get_indicators(closes_json: str) -> dict:
    from tools.technical_indicators import TechnicalIndicatorCalculator
    t = TechnicalIndicatorCalculator()
    return {ind: json.loads(t._run(closes_json, ind)) for ind in ["RSI", "MACD", "SMA", "EMA", "BOLLINGER"]}

@st.cache_data(ttl=600)
def get_rag_results(query: str, ticker: str, top_k: int = 5) -> list[dict]:
    from rag.retriever import get_retriever
    return get_retriever().retrieve(query=query, ticker=ticker, top_k=top_k)

@st.cache_data(ttl=600)
def get_rag_context_string(query: str, ticker: str) -> str:
    from rag.retriever import get_retriever
    r = get_retriever()
    return r.format_context(r.retrieve(query=query, ticker=ticker, top_k=3))

def get_prompt_info(role, version, symbol, rag_results):
    from prompts.strategies import build_agent_prompt, get_strategy_description
    p = build_agent_prompt(role, version, symbol=symbol, rag_results=rag_results)
    p["description"] = get_strategy_description(version)
    return p

def signal_delta(signal: str) -> str:
    s = signal.upper()
    if "BULLISH" in s: return "Bullish"
    if "BEARISH" in s: return "Bearish"
    if "OVERBOUGHT" in s: return "Overbought"
    if "OVERSOLD" in s: return "Oversold"
    return "Neutral"

def esc(val):
    """Dollar-safe formatting for Streamlit markdown."""
    if isinstance(val, (int, float)):
        return f"\\${val:.2f}"
    return str(val).replace("$", "\\$")


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 📊 Stock Research")
    st.caption("AI-powered multi-agent analysis")
    st.markdown("---")

    symbol = st.text_input("Stock Symbol", value="AAPL", max_chars=10,
                           placeholder="e.g. AAPL, TSLA").upper().strip()
    days = st.slider("Analysis Period (days)", 30, 365, 90)
    prompt_version = st.selectbox(
        "Prompt Version",
        ["v3_cot_rag", "v2_structured", "v1_basic"],
        format_func=lambda x: {
            "v1_basic": "v1 — Basic",
            "v2_structured": "v2 — Structured + Few-Shot",
            "v3_cot_rag": "v3 — CoT + RAG + Few-Shot",
        }[x]
    )
    st.markdown("")
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    st.markdown("---")
    has_api_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    if has_api_key:
        st.success("API Key connected", icon="✅")
    else:
        st.info("Demo mode — synthetic data", icon="ℹ️")

    st.markdown("---")
    st.caption("INFO 7375 Final Project\nNortheastern University")

# ============================================================
# Main content
# ============================================================

if not symbol:
    st.warning("Enter a stock symbol in the sidebar.")
    st.stop()

# ---- run analysis ----
with st.spinner(f"Analyzing **{symbol}**..."):
    t0 = time.time()
    data = get_stock_data(symbol, days)
    raw_json = json.dumps(data)
    closes = [d["close"] for d in data["data"]]
    dates  = [d["date"]  for d in data["data"]]
    stats  = get_data_stats(raw_json)
    indicators = get_indicators(json.dumps(closes))
    rag_query   = f"{symbol} financial analysis earnings revenue outlook risk"
    rag_results = get_rag_results(rag_query, symbol, top_k=5)
    analysis_time = time.time() - t0

# ---- overall signal ----
INDICATOR_NAMES = ["RSI", "MACD", "SMA", "EMA", "BOLLINGER"]
signals  = [indicators[i].get("signal", "") for i in INDICATOR_NAMES]
bullish  = sum(1 for s in signals if "BULLISH" in s.upper())
bearish  = sum(1 for s in signals if "BEARISH" in s.upper() or "OVERBOUGHT" in s.upper())
if   bullish >= 4: overall = "STRONG BULLISH"
elif bullish >= 3: overall = "BULLISH"
elif bearish >= 3: overall = "BEARISH"
else:              overall = "NEUTRAL"

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Dashboard",
    "🔍 RAG Context",
    "📄 Report",
    "🧪 Prompt Lab",
    "📊 Metrics",
])

# ─────────────────────── Tab 1 ───────────────────────
with tab1:
    # header row
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.markdown(f"## {symbol} — Technical Analysis")
    with hcol2:
        st.caption(f"{data.get('source', 'unknown')}  ·  {days}d  ·  {data.get('data_points', 0)} pts")

    # signal banner
    css_class = "bullish" if "BULLISH" in overall else ("bearish" if "BEARISH" in overall else "neutral")
    icon = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}[css_class]
    st.markdown(
        f'<div class="signal-banner {css_class}">{icon} {overall} &nbsp;—&nbsp; '
        f'{bullish} bullish / {bearish} bearish out of 5 indicators</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # chart
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=dates,
            open=[d["open"] for d in data["data"]],
            high=[d["high"] for d in data["data"]],
            low=[d["low"]  for d in data["data"]],
            close=closes,
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ))
        sma_v = indicators["SMA"].get("current_value")
        if sma_v:
            fig.add_hline(y=sma_v, line_dash="dot", line_color="#4f8ff7",
                          annotation_text=f"SMA-20: ${sma_v:.2f}",
                          annotation_font_color="#4f8ff7")
        fig.update_layout(
            yaxis_title="Price ($)", xaxis_title="",
            height=420,
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.line_chart({"Close": closes})

    # indicator cards
    st.markdown("#### Technical Indicators")
    cols = st.columns(5)
    for col, ind_name in zip(cols, INDICATOR_NAMES):
        ind = indicators.get(ind_name, {})
        sig = ind.get("signal", "N/A")
        val = ind.get("current_value", ind.get("macd_line", "—"))
        if isinstance(val, (int, float)):
            val = f"{val:.2f}"
        with col:
            st.metric(
                label=ind_name,
                value=val,
                delta=signal_delta(sig),
                delta_color="normal" if "BULLISH" in sig.upper()
                    else "inverse" if "BEARISH" in sig.upper() else "off",
            )

    # stats row
    st.markdown("#### Key Statistics")
    ret = stats["returns"]
    vol = stats["volatility"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price",       f"${closes[-1]:.2f}")
    c2.metric("Return",      f"{ret['cumulative_return_pct']}%",
              delta=f"{ret['cumulative_return_pct']}%")
    c3.metric("Volatility",  f"{vol['annualized_volatility_pct']}%")
    c4.metric("Risk",        vol["risk_level"])
    c5.metric("Max DD",      f"{vol['max_drawdown_pct']}%")
    c6.metric("Win Rate",    f"{ret['win_rate_pct']}%")


# ─────────────────────── Tab 2 ───────────────────────
with tab2:
    st.markdown("## RAG Knowledge Context")

    from rag.retriever import get_retriever
    retriever = get_retriever()

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Backend",   retriever.backend_name)
    mc2.metric("Indexed",   retriever.document_count)
    mc3.metric("Returned",  len(rag_results))

    st.markdown("")

    if not rag_results:
        st.warning("No relevant documents found.")
    else:
        for r in rag_results:
            pct = r["relevance_score"] * 100
            st.markdown(
                f'<div class="rag-card">'
                f'<span class="score-badge">{pct:.0f}%</span> &nbsp; '
                f'<strong>#{r["rank"]}</strong> &nbsp; <code>{r["source"]}</code> &nbsp; '
                f'<span style="color:#94a3b8;font-size:.82rem">{r["doc_type"]} · {r["section"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("View content", expanded=(r["rank"] <= 2)):
                st.markdown(r["content"].replace("$", "\\$"))
                st.caption(r["citation"])


# ─────────────────────── Tab 3 ───────────────────────
with tab3:
    st.markdown("## Generated Research Report")

    rag_context = get_rag_context_string(rag_query, symbol)
    company_context = rag_results[0]["content"][:300] if rag_results else "No fundamental data available."

    # support / resistance
    sma_val  = indicators['SMA'].get('current_value')
    bb_lower = indicators['BOLLINGER'].get('lower_band')
    bb_upper = indicators['BOLLINGER'].get('upper_band')
    rsi_val  = indicators['RSI'].get('current_value')
    support_level = bb_lower if bb_lower else (min(closes[-20:]) if len(closes) >= 20 else min(closes))
    resist_level  = bb_upper if bb_upper else (max(closes[-20:]) if len(closes) >= 20 else max(closes))
    support_str = esc(support_level) if isinstance(support_level, (int, float)) else esc(min(closes))
    resist_str  = esc(resist_level)  if isinstance(resist_level, (int, float))  else esc(max(closes))

    # risk factors
    risk_factors = []
    ann_vol = stats['volatility']['annualized_volatility_pct']
    if ann_vol > 30:
        risk_factors.append(f"High annualized volatility ({ann_vol}%) indicates significant price swings")
    if rsi_val and rsi_val > 70:
        risk_factors.append(f"RSI at {rsi_val:.1f} signals overbought conditions — pullback risk elevated")
    elif rsi_val and rsi_val < 30:
        risk_factors.append(f"RSI at {rsi_val:.1f} signals oversold conditions — potential capitulation risk")
    max_dd = stats['volatility']['max_drawdown_pct']
    if max_dd > 15:
        risk_factors.append(f"Max drawdown of {max_dd}% suggests vulnerability to sharp corrections")
    if not risk_factors:
        risk_factors.append(f"Moderate risk profile with {ann_vol}% annualized volatility")

    # catalysts
    def extract_catalysts(results, sym):
        cats = []
        for r in results[:3]:
            for ln in [l.strip() for l in r["content"].replace("\n\n","\n").split("\n") if l.strip()]:
                if ln.startswith("#") or ln.startswith("[Source") or len(ln) < 20:
                    continue
                if any(k in ln.lower() for k in ["expected","guidance","outlook","forecast","growth",
                        "plan","launch","expand","target","initiative","accelerat","ramp","invest","announced"]):
                    c = ln.lstrip("-•* ").replace("**","").replace("$","\\$")
                    if ". " in c and len(c) > 120: c = c[:c.index(". ")+1]
                    if c and c not in cats: cats.append(c)
                    if len(cats) >= 4: break
            if len(cats) >= 4: break
        return cats or [f"Monitor upcoming earnings releases and guidance for {sym}",
                        "Watch for sector-wide catalysts including monetary policy decisions"]

    catalysts = extract_catalysts(rag_results, symbol)
    company_esc = company_context.replace("$", "\\$")
    cite_esc = rag_results[0]["citation"].replace("$","\\$") if rag_results else ""

    report_md = f"""# {symbol} Stock Research Report
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Prompt: {prompt_version}*

---

## Executive Summary
{symbol} shows a **{overall}** technical profile over the past {days} days with a cumulative return of {ret['cumulative_return_pct']}%. {bullish} of 5 technical indicators are bullish.

## Company & Market Context
{company_esc}

{f'*{cite_esc}*' if rag_results else ''}

## Price Overview
| Metric | Value |
|--------|-------|
| Current Price | {esc(closes[-1])} |
| Period High | {esc(max(closes))} |
| Period Low | {esc(min(closes))} |
| Mean Price | {esc(stats['summary']['price_stats']['mean'])} |
| Cumulative Return | {ret['cumulative_return_pct']}% |
| Win Rate | {ret['win_rate_pct']}% |

## Technical Analysis
| Indicator | Value | Signal |
|-----------|-------|--------|
| RSI-14 | {indicators['RSI'].get('current_value','N/A')} | {indicators['RSI'].get('signal','N/A')} |
| MACD | {indicators['MACD'].get('macd_line','N/A')} | {indicators['MACD'].get('signal','N/A')} |
| SMA-20 | {indicators['SMA'].get('current_value','N/A')} | {indicators['SMA'].get('signal','N/A')} |
| EMA-20 | {indicators['EMA'].get('current_value','N/A')} | {indicators['EMA'].get('signal','N/A')} |
| Bollinger | {indicators['BOLLINGER'].get('current_value','N/A')} | {indicators['BOLLINGER'].get('signal','N/A')} |

**Overall Consensus: {overall}** ({bullish}/5 bullish)

## Risk Assessment
| Metric | Value |
|--------|-------|
| Annualized Volatility | {vol['annualized_volatility_pct']}% |
| Risk Level | {vol['risk_level']} |
| Max Drawdown | {vol['max_drawdown_pct']}% |
| Best Day | {ret['best_day_pct']}% |
| Worst Day | {ret['worst_day_pct']}% |

## Conclusion

Based on the technical analysis, {symbol} presents a **{overall.lower()}** outlook over the analyzed {days}-day period.

**Key Levels to Watch:**
- **Support:** {support_str} (lower Bollinger Band / recent low)
- **Resistance:** {resist_str} (upper Bollinger Band / recent high)
- **SMA-20:** {esc(sma_val) if isinstance(sma_val,(int,float)) else 'N/A'} — a close below this level would weaken the bullish case

**Risk Factors:**
{chr(10).join(f'- {rf}' for rf in risk_factors)}

**Catalysts & Events to Monitor:**
{chr(10).join(f'- {c}' for c in catalysts)}

**Recommendation:** {"Maintain or build positions with stop-loss below SMA-20. Momentum and trend indicators are aligned to the upside." if bullish >= 3 else "Wait for clearer directional signals. Consider reducing exposure if key support levels break." if overall == "NEUTRAL" else "Consider reducing exposure or hedging. Multiple bearish signals point to further downside risk. Watch for reversal patterns near support."}

---
*Disclaimer: This report is for informational purposes only and does not constitute investment advice. Past performance does not guarantee future results.*

*Report generated by AI-Powered Stock Research Platform*
"""

    st.markdown(f'<div class="report-container">', unsafe_allow_html=True)
    st.markdown(report_md)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button("Download Markdown", data=report_md,
                           file_name=f"{symbol}_report.md", mime="text/markdown",
                           use_container_width=True)
    with dc2:
        st.download_button("Download JSON", use_container_width=True,
            data=json.dumps({"symbol": symbol, "date": datetime.now().isoformat(),
                "signal": overall, "indicators": indicators, "stats": stats,
                "rag_sources": [r["source"] for r in rag_results]}, indent=2),
            file_name=f"{symbol}_report.json", mime="application/json")


# ─────────────────────── Tab 4 ───────────────────────
with tab4:
    st.markdown("## Prompt Engineering Lab")
    st.caption("Compare how different prompt strategies shape agent behaviour.")

    from prompts.templates import PROMPT_VERSIONS
    from prompts.strategies import get_strategy_description

    compare_role = st.selectbox("Agent Role", ["analyst", "data_collector", "report_writer", "controller"])

    vcols = st.columns(3)
    version_tags = {
        "v1_basic": [],
        "v2_structured": ["Few-Shot"],
        "v3_cot_rag": ["CoT", "RAG", "Few-Shot"],
    }
    for col, ver in zip(vcols, PROMPT_VERSIONS):
        p = get_prompt_info(compare_role, ver, symbol, rag_results)
        with col:
            tags_html = "".join(f'<span class="tag">{t}</span>' for t in version_tags[ver])
            st.markdown(
                f'<div class="prompt-card">'
                f'<h4>{ver}</h4>'
                f'{tags_html}'
                f'<p style="color:#64748b;font-size:.82rem;margin:8px 0 4px">{p["description"]}</p>'
                f'<p style="margin:4px 0"><strong>Role:</strong> {p["role"]}</p>'
                f'<p style="color:#64748b;font-size:.85rem">{p["goal"][:160]}…</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.metric("Backstory", f"{len(p['backstory']):,} chars")
            with st.expander("Full backstory"):
                st.code(p["backstory"], language="text")

    # strategy matrix
    st.markdown("#### Strategy Matrix")
    st.markdown("""
| Strategy | What it does | v1 | v2 | v3 |
|----------|-------------|:--:|:--:|:--:|
| **Role Reinforcement** | Detailed persona & constraints | ✓ | ✓ | ✓ |
| **Structured Workflow** | Numbered step-by-step process | | ✓ | ✓ |
| **Few-Shot Examples** | Worked input / output pairs | | ✓ | ✓ |
| **Chain-of-Thought** | "Think step-by-step" scaffold | | | ✓ |
| **RAG Context** | Inject retrieved documents | | | ✓ |
    """)

    # live preview
    st.markdown("#### Live Prompt Preview")
    pc1, pc2 = st.columns(2)
    pv = pc1.selectbox("Version", PROMPT_VERSIONS, key="pv")
    pr = pc2.selectbox("Role", ["analyst","data_collector","report_writer"], key="pr")
    pp = get_prompt_info(pr, pv, symbol, rag_results)
    st.code(f"Role: {pp['role']}\n\nGoal:\n{pp['goal']}\n\nBackstory:\n{pp['backstory']}", language="text")


# ─────────────────────── Tab 5 ───────────────────────
with tab5:
    st.markdown("## System Metrics")

    from rag.retriever import get_retriever
    retriever = get_retriever()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Analysis Time", f"{analysis_time:.2f}s")
    m2.metric("Data Points",   data.get("data_points", 0))
    m3.metric("RAG Results",   len(rag_results))
    m4.metric("KB Chunks",     retriever.document_count)

    # RAG table
    st.markdown("#### RAG Retrieval Performance")
    if rag_results:
        scores = [r["relevance_score"] for r in rag_results]
        st.dataframe(
            {"Rank": [r["rank"] for r in rag_results],
             "Source": [r["source"] for r in rag_results],
             "Score": [f"{s:.4f}" for s in scores],
             "Type": [r["doc_type"] for r in rag_results]},
            use_container_width=True, hide_index=True,
        )
        avg_s = sum(scores)/len(scores)
        st.caption(f"Avg relevance **{avg_s:.4f}** · Max **{max(scores):.4f}** · Backend **{retriever.backend_name}**")

    # Prompt stats
    st.markdown("#### Prompt Engineering Stats")
    rows = []
    for ver in PROMPT_VERSIONS:
        for role in ["data_collector","analyst","report_writer","controller"]:
            p = get_prompt_info(role, ver, symbol, rag_results)
            rows.append({"Version": ver, "Role": role, "Backstory": f"{len(p['backstory']):,} chars"})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # architecture
    st.markdown("#### Architecture")
    st.code("""
┌─────────────────────────────────────────────────────┐
│                 Controller Agent                     │
│         (Orchestrates workflow + error handling)      │
└──────────┬────────────────┬──────────────────────────┘
           │                │                │
┌──────────▼──┐  ┌──────────▼──────┐  ┌─────▼──────────┐
│ Data        │  │  Technical      │  │ Report Writer  │
│ Collector   │  │  Analyst        │  │                │
│             │  │                 │  │ Tools:         │
│ Tools:      │  │ Tools:          │  │ • Report       │
│ • Stock     │  │ • Indicator     │  │   Formatter    │
│   Data      │  │   Calculator    │  │ • RAG Search   │
│ • Processor │  │ • Processor     │  └────────────────┘
│ • RAG       │  │ • RAG Search    │
└─────────────┘  └─────────────────┘
           │                │                │
           ▼                ▼                ▼
┌─────────────────────────────────────────────────────┐
│            RAG Knowledge Base (ChromaDB)              │
│  Financial Glossary │ SEC Filings │ Earnings │ Mkt   │
└─────────────────────────────────────────────────────┘
""", language="text")
