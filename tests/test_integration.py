"""
Integration tests — end-to-end pipeline verification.
Tests the full workflow in demo mode without API keys.
"""

import json
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDemoPipeline:
    """Tests that the demo pipeline runs end-to-end."""

    def test_data_retrieval_synthetic(self):
        from tools.builtin_tools import StockDataRetrieverTool
        tool = StockDataRetrieverTool()
        result = json.loads(tool._run("AAPL", 60))
        assert result["data_points"] > 0
        assert len(result["data"]) > 0

    def test_full_analysis_pipeline(self):
        """Run the full demo pipeline (data → indicators → RAG → report)."""
        from tools.builtin_tools import StockDataRetrieverTool, DataProcessorTool, ReportFormatterTool
        from tools.technical_indicators import TechnicalIndicatorCalculator
        from tools.rag_tool import FinancialKnowledgeSearchTool

        # Step 1: Data
        retriever = StockDataRetrieverTool()
        raw_data = retriever._run("MSFT", 60)
        parsed = json.loads(raw_data)
        assert parsed["data_points"] > 0

        # Step 2: Process
        processor = DataProcessorTool()
        stats = json.loads(processor._run(raw_data, "summary_stats"))
        assert "price_stats" in stats

        returns = json.loads(processor._run(raw_data, "returns"))
        assert "cumulative_return_pct" in returns

        vol = json.loads(processor._run(raw_data, "volatility"))
        assert "risk_level" in vol

        # Step 3: Indicators
        closes = [d["close"] for d in parsed["data"]]
        indicator_tool = TechnicalIndicatorCalculator()
        closes_json = json.dumps(closes)

        for ind in ["RSI", "MACD", "SMA", "EMA", "BOLLINGER"]:
            result = json.loads(indicator_tool._run(closes_json, ind))
            assert "signal" in result, f"{ind} missing signal"

        # Step 4: RAG
        rag_tool = FinancialKnowledgeSearchTool()
        rag_result = json.loads(rag_tool._run("MSFT revenue cloud growth", ticker="MSFT"))
        assert rag_result["results_count"] > 0

        # Step 5: Report
        formatter = ReportFormatterTool()
        sections = json.dumps([
            {"heading": "Summary", "content": "Test report"},
            {"heading": "Analysis", "content": "Technical signals are mixed"},
        ])
        report = formatter._run("MSFT Test Report", sections, "markdown")
        assert "MSFT Test Report" in report

    def test_rag_retrieval_relevance(self):
        """Verify that RAG returns relevant results for known queries."""
        from rag.retriever import FinancialRetriever
        retriever = FinancialRetriever()

        # AAPL query should return AAPL documents with higher relevance
        results = retriever.retrieve("Apple iPhone revenue growth", ticker="AAPL", top_k=5)
        assert len(results) > 0

        # Check that at least one result mentions Apple or AAPL
        top_content = " ".join(r["content"].lower() for r in results[:3])
        assert "apple" in top_content or "aapl" in top_content or "iphone" in top_content, \
            "Top results should be relevant to Apple"

    def test_prompt_engineering_integration(self):
        """Verify prompts can be built and used for all roles and versions."""
        from prompts.strategies import build_agent_prompt
        from prompts.templates import PROMPT_VERSIONS

        rag_results = [{"source": "test.md", "content": "Test data", "relevance_score": 0.5}]

        for role in ["data_collector", "analyst", "report_writer", "controller"]:
            for version in PROMPT_VERSIONS:
                result = build_agent_prompt(role, version, symbol="TEST", rag_results=rag_results)
                assert isinstance(result["role"], str) and len(result["role"]) > 0
                assert isinstance(result["goal"], str) and len(result["goal"]) > 0
                assert isinstance(result["backstory"], str) and len(result["backstory"]) > 0


class TestMultiStockAnalysis:
    """Test that multiple stock symbols work correctly."""

    @pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"])
    def test_data_retrieval_per_symbol(self, symbol):
        from tools.builtin_tools import StockDataRetrieverTool
        tool = StockDataRetrieverTool()
        result = json.loads(tool._run(symbol, 30))
        assert result["data_points"] > 0
        assert result["symbol"] == symbol

    @pytest.mark.parametrize("symbol", ["AAPL", "TSLA"])
    def test_rag_with_different_tickers(self, symbol):
        from rag.retriever import get_retriever
        get_retriever().initialize()  # Ensure KB is loaded
        from tools.rag_tool import FinancialKnowledgeSearchTool
        tool = FinancialKnowledgeSearchTool()
        result = json.loads(tool._run(f"{symbol} analysis", ticker=symbol))
        assert isinstance(result.get("results", []), list)


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_unknown_symbol_synthetic_fallback(self):
        from tools.builtin_tools import StockDataRetrieverTool
        tool = StockDataRetrieverTool()
        result = json.loads(tool._run("ZZZZZ", 30))
        assert result["data_points"] > 0  # Should fall back to synthetic

    def test_very_short_period(self):
        from tools.builtin_tools import StockDataRetrieverTool
        tool = StockDataRetrieverTool()
        result = json.loads(tool._run("AAPL", 5))
        assert result["data_points"] > 0

    def test_indicator_insufficient_data(self):
        from tools.technical_indicators import TechnicalIndicatorCalculator
        tool = TechnicalIndicatorCalculator()
        short_data = json.dumps([100.0, 101.0, 99.0])
        result = json.loads(tool._run(short_data, "RSI"))
        assert "error" in result  # Should report insufficient data

    def test_rag_query_no_matches(self):
        from tools.rag_tool import FinancialKnowledgeSearchTool
        tool = FinancialKnowledgeSearchTool()
        result = json.loads(tool._run("completely unrelated quantum physics query"))
        # Should not crash, might return low-relevance results or empty
        assert isinstance(result, dict)
