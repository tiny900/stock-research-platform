"""
Evaluation Metrics — Performance and Quality Assessment
========================================================
Provides metrics for evaluating RAG retrieval quality,
prompt engineering effectiveness, and system performance.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGMetrics:
    """Metrics for RAG retrieval evaluation."""
    query: str
    results_count: int
    avg_relevance_score: float
    max_relevance_score: float
    retrieval_time_ms: float
    backend: str
    ticker_filter: Optional[str] = None


@dataclass
class PipelineMetrics:
    """Metrics for the full analysis pipeline."""
    symbol: str
    data_retrieval_ms: float = 0.0
    data_processing_ms: float = 0.0
    indicator_calculation_ms: float = 0.0
    rag_retrieval_ms: float = 0.0
    report_generation_ms: float = 0.0
    total_ms: float = 0.0
    prompt_version: str = ""
    data_points: int = 0
    rag_results_count: int = 0


def evaluate_rag_retrieval(queries: list[dict]) -> list[RAGMetrics]:
    """
    Evaluate RAG retrieval quality across multiple queries.

    Args:
        queries: List of dicts with 'query', optional 'ticker', optional 'expected_sources'

    Returns:
        List of RAGMetrics for each query
    """
    from rag.retriever import get_retriever
    retriever = get_retriever()

    metrics = []
    for q in queries:
        start = time.time()
        results = retriever.retrieve(
            query=q["query"],
            ticker=q.get("ticker"),
            top_k=q.get("top_k", 5),
        )
        elapsed = (time.time() - start) * 1000

        scores = [r["relevance_score"] for r in results] if results else [0.0]
        metrics.append(RAGMetrics(
            query=q["query"],
            results_count=len(results),
            avg_relevance_score=sum(scores) / len(scores),
            max_relevance_score=max(scores),
            retrieval_time_ms=round(elapsed, 2),
            backend=retriever.backend_name,
            ticker_filter=q.get("ticker"),
        ))

    return metrics


def evaluate_pipeline(symbol: str, days: int = 60) -> PipelineMetrics:
    """
    Run the full analysis pipeline and collect timing metrics.

    Args:
        symbol: Stock ticker symbol
        days: Analysis period

    Returns:
        PipelineMetrics with timing breakdown
    """
    metrics = PipelineMetrics(symbol=symbol)

    # Data retrieval
    from tools.builtin_tools import StockDataRetrieverTool, DataProcessorTool
    start = time.time()
    retriever = StockDataRetrieverTool()
    raw_data = retriever._run(symbol, days)
    metrics.data_retrieval_ms = round((time.time() - start) * 1000, 2)

    parsed = json.loads(raw_data)
    metrics.data_points = parsed.get("data_points", 0)
    closes = [d["close"] for d in parsed["data"]]

    # Data processing
    start = time.time()
    processor = DataProcessorTool()
    processor._run(raw_data, "summary_stats")
    processor._run(raw_data, "returns")
    processor._run(raw_data, "volatility")
    metrics.data_processing_ms = round((time.time() - start) * 1000, 2)

    # Indicator calculation
    from tools.technical_indicators import TechnicalIndicatorCalculator
    start = time.time()
    indicator_tool = TechnicalIndicatorCalculator()
    closes_json = json.dumps(closes)
    for ind in ["RSI", "MACD", "SMA", "EMA", "BOLLINGER"]:
        indicator_tool._run(closes_json, ind)
    metrics.indicator_calculation_ms = round((time.time() - start) * 1000, 2)

    # RAG retrieval
    from rag.retriever import get_retriever
    start = time.time()
    rag_retriever = get_retriever()
    results = rag_retriever.retrieve(f"{symbol} analysis", ticker=symbol, top_k=3)
    metrics.rag_retrieval_ms = round((time.time() - start) * 1000, 2)
    metrics.rag_results_count = len(results)

    # Report generation
    from tools.builtin_tools import ReportFormatterTool
    start = time.time()
    formatter = ReportFormatterTool()
    sections = json.dumps([
        {"heading": "Summary", "content": f"Analysis of {symbol}"},
        {"heading": "Analysis", "content": "Technical indicators analyzed"},
    ])
    formatter._run(f"{symbol} Report", sections, "markdown")
    metrics.report_generation_ms = round((time.time() - start) * 1000, 2)

    metrics.total_ms = round(
        metrics.data_retrieval_ms + metrics.data_processing_ms +
        metrics.indicator_calculation_ms + metrics.rag_retrieval_ms +
        metrics.report_generation_ms, 2
    )

    return metrics


def generate_evaluation_report() -> str:
    """Generate a comprehensive evaluation report."""
    lines = ["# System Evaluation Report", ""]
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # RAG evaluation
    lines.append("## RAG Retrieval Quality\n")
    test_queries = [
        {"query": "AAPL revenue growth earnings", "ticker": "AAPL"},
        {"query": "Tesla electric vehicle deliveries", "ticker": "TSLA"},
        {"query": "What is RSI indicator"},
        {"query": "NVIDIA data center GPU revenue", "ticker": "NVDA"},
        {"query": "stock market risk assessment volatility"},
    ]

    rag_metrics = evaluate_rag_retrieval(test_queries)
    lines.append("| Query | Ticker | Results | Avg Score | Max Score | Time (ms) |")
    lines.append("|-------|--------|---------|-----------|-----------|-----------|")
    for m in rag_metrics:
        lines.append(f"| {m.query[:40]} | {m.ticker_filter or 'None'} | {m.results_count} | "
                     f"{m.avg_relevance_score:.4f} | {m.max_relevance_score:.4f} | {m.retrieval_time_ms:.1f} |")

    avg_score = sum(m.avg_relevance_score for m in rag_metrics) / len(rag_metrics)
    avg_time = sum(m.retrieval_time_ms for m in rag_metrics) / len(rag_metrics)
    lines.append(f"\n**Average relevance score:** {avg_score:.4f}")
    lines.append(f"**Average retrieval time:** {avg_time:.1f}ms")
    lines.append(f"**Backend:** {rag_metrics[0].backend}\n")

    # Pipeline performance
    lines.append("## Pipeline Performance\n")
    symbols = ["AAPL", "MSFT", "TSLA"]
    lines.append("| Symbol | Data (ms) | Process (ms) | Indicators (ms) | RAG (ms) | Report (ms) | Total (ms) |")
    lines.append("|--------|-----------|--------------|-----------------|----------|-------------|------------|")

    for sym in symbols:
        pm = evaluate_pipeline(sym)
        lines.append(f"| {sym} | {pm.data_retrieval_ms} | {pm.data_processing_ms} | "
                     f"{pm.indicator_calculation_ms} | {pm.rag_retrieval_ms} | "
                     f"{pm.report_generation_ms} | {pm.total_ms} |")

    # Prompt engineering stats
    lines.append("\n## Prompt Engineering\n")
    from prompts.strategies import build_agent_prompt, get_strategy_description
    from prompts.templates import PROMPT_VERSIONS

    lines.append("| Version | Description | Analyst Backstory Length |")
    lines.append("|---------|-------------|------------------------|")
    for ver in PROMPT_VERSIONS:
        p = build_agent_prompt("analyst", ver, symbol="AAPL")
        desc = get_strategy_description(ver)
        lines.append(f"| {ver} | {desc} | {len(p['backstory'])} chars |")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    report = generate_evaluation_report()
    print(report)

    import os
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_report.md", "w") as f:
        f.write(report)
    print("\nSaved to outputs/evaluation_report.md")
