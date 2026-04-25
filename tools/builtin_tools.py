"""
Built-in Tools for the Stock Research Agentic System
=====================================================
Integrates 3 built-in tools as required by the assignment:
1. Web Search / Data Retrieval Tool (SerperDevTool or fallback)
2. Data Processing / Transformation Tool
3. Communication / Output Formatting Tool
"""

import json
import csv
import os
from datetime import datetime, timedelta
from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


# ============================================================
# TOOL 1: Stock Data Retrieval Tool (Data Retrieval)
# ============================================================

class StockDataInput(BaseModel):
    """Input for stock data retrieval."""
    symbol: str = Field(..., description="Stock ticker symbol, e.g. 'AAPL'")
    days: int = Field(default=90, description="Number of days of historical data")


class StockDataRetrieverTool(BaseTool):
    """
    Built-in Tool #1: Retrieves stock market data.
    Uses yfinance for real data, with synthetic fallback for demo/offline mode.
    """
    name: str = "stock_data_retriever"
    description: str = (
        "Retrieves historical stock price data for a given ticker symbol. "
        "Returns OHLCV data (Open, High, Low, Close, Volume) as JSON. "
        "Input: symbol (str), days (int, default 90)."
    )
    args_schema: Type[BaseModel] = StockDataInput

    def _run(self, symbol: str, days: int = 90) -> str:
        """Fetch stock data using yfinance or generate synthetic data."""
        symbol = symbol.upper().strip()
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            end = datetime.now()
            start = end - timedelta(days=days)
            df = ticker.history(start=start, end=end)

            if df.empty:
                return self._generate_synthetic_data(symbol, days)

            records = []
            for date, row in df.iterrows():
                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(row["Open"], 2),
                    "high": round(row["High"], 2),
                    "low": round(row["Low"], 2),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"])
                })

            return json.dumps({
                "symbol": symbol,
                "days_requested": days,
                "data_points": len(records),
                "data": records,
                "source": "yfinance"
            }, indent=2)

        except Exception:
            return self._generate_synthetic_data(symbol, days)

    def _generate_synthetic_data(self, symbol: str, days: int) -> str:
        """Generate realistic synthetic stock data for demo purposes."""
        import random
        random.seed(hash(symbol) % 2**32)

        base_prices = {
            "AAPL": 185.0, "MSFT": 415.0, "GOOGL": 175.0,
            "AMZN": 185.0, "TSLA": 245.0, "NVDA": 880.0,
            "META": 500.0, "NFLX": 620.0, "AMD": 170.0,
        }
        base = base_prices.get(symbol, 100.0 + random.random() * 200)

        records = []
        price = base
        for i in range(days):
            date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
            change = random.gauss(0.0005, 0.02)
            price = max(price * (1 + change), 1.0)
            high = price * (1 + abs(random.gauss(0, 0.01)))
            low = price * (1 - abs(random.gauss(0, 0.01)))
            volume = int(random.gauss(50_000_000, 15_000_000))
            records.append({
                "date": date,
                "open": round(price * (1 + random.gauss(0, 0.005)), 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "volume": max(volume, 1_000_000)
            })

        return json.dumps({
            "symbol": symbol,
            "days_requested": days,
            "data_points": len(records),
            "data": records,
            "source": "synthetic (demo mode)"
        }, indent=2)


# ============================================================
# TOOL 2: Data Processing / Transformation Tool
# ============================================================

class DataProcessorInput(BaseModel):
    """Input for data processing."""
    data: str = Field(..., description="JSON string of stock data (from stock_data_retriever)")
    operation: str = Field(
        ...,
        description="Operation: 'summary_stats', 'returns', 'volatility', 'moving_avg', or 'export_csv'"
    )
    params: Optional[str] = Field(default=None, description="Optional JSON params, e.g. '{\"period\": 20}'")


class DataProcessorTool(BaseTool):
    """
    Built-in Tool #2: Processes and transforms stock data.
    Supports statistical summaries, return calculations, volatility analysis,
    moving averages, and CSV export.
    """
    name: str = "data_processor"
    description: str = (
        "Processes stock data with operations: 'summary_stats' (basic statistics), "
        "'returns' (daily/cumulative returns), 'volatility' (risk metrics), "
        "'moving_avg' (moving averages), 'export_csv' (save to CSV file). "
        "Input data should be JSON from the stock_data_retriever tool."
    )
    args_schema: Type[BaseModel] = DataProcessorInput

    def _run(self, data: str, operation: str, params: Optional[str] = None) -> str:
        """Process stock data."""
        try:
            parsed = json.loads(data)
            records = parsed.get("data", [])
            if not records:
                return json.dumps({"error": "No data found in input"})

            closes = [r["close"] for r in records]
            parameters = json.loads(params) if params else {}

            operations = {
                "summary_stats": self._summary_stats,
                "returns": self._calculate_returns,
                "volatility": self._calculate_volatility,
                "moving_avg": self._calculate_moving_avg,
                "export_csv": self._export_csv,
            }

            if operation not in operations:
                return json.dumps({"error": f"Unknown operation: {operation}. Supported: {list(operations.keys())}"})

            result = operations[operation](records, closes, parameters)
            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Processing error: {str(e)}"})

    def _summary_stats(self, records, closes, params) -> dict:
        n = len(closes)
        sorted_c = sorted(closes)
        mean = sum(closes) / n
        median = sorted_c[n // 2] if n % 2 else (sorted_c[n // 2 - 1] + sorted_c[n // 2]) / 2
        std = (sum((c - mean) ** 2 for c in closes) / n) ** 0.5
        volumes = [r["volume"] for r in records]

        return {
            "operation": "summary_stats",
            "period": f"{records[0]['date']} to {records[-1]['date']}",
            "data_points": n,
            "price_stats": {
                "current": closes[-1],
                "mean": round(mean, 2),
                "median": round(median, 2),
                "std_dev": round(std, 2),
                "min": round(min(closes), 2),
                "max": round(max(closes), 2),
                "range": round(max(closes) - min(closes), 2),
                "pct_from_high": round((closes[-1] - max(closes)) / max(closes) * 100, 2),
            },
            "volume_stats": {
                "avg_volume": int(sum(volumes) / len(volumes)),
                "max_volume": max(volumes),
                "min_volume": min(volumes),
            }
        }

    def _calculate_returns(self, records, closes, params) -> dict:
        daily_returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 for i in range(1, len(closes))]
        cumulative = (closes[-1] - closes[0]) / closes[0] * 100

        pos_days = sum(1 for r in daily_returns if r > 0)
        neg_days = sum(1 for r in daily_returns if r < 0)

        return {
            "operation": "returns",
            "cumulative_return_pct": round(cumulative, 2),
            "avg_daily_return_pct": round(sum(daily_returns) / len(daily_returns), 4),
            "best_day_pct": round(max(daily_returns), 2),
            "worst_day_pct": round(min(daily_returns), 2),
            "positive_days": pos_days,
            "negative_days": neg_days,
            "win_rate_pct": round(pos_days / len(daily_returns) * 100, 1),
            "last_5_daily_returns": [round(r, 2) for r in daily_returns[-5:]]
        }

    def _calculate_volatility(self, records, closes, params) -> dict:
        daily_returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        mean_ret = sum(daily_returns) / len(daily_returns)
        var = sum((r - mean_ret) ** 2 for r in daily_returns) / len(daily_returns)
        daily_vol = var ** 0.5
        annual_vol = daily_vol * (252 ** 0.5)

        # Max drawdown
        peak = closes[0]
        max_dd = 0
        for c in closes:
            if c > peak:
                peak = c
            dd = (peak - c) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "operation": "volatility",
            "daily_volatility_pct": round(daily_vol * 100, 4),
            "annualized_volatility_pct": round(annual_vol * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "risk_level": "HIGH" if annual_vol > 0.4 else "MEDIUM" if annual_vol > 0.2 else "LOW"
        }

    def _calculate_moving_avg(self, records, closes, params) -> dict:
        periods = params.get("periods", [10, 20, 50])
        result = {"operation": "moving_averages", "averages": {}}

        for p in periods:
            if len(closes) >= p:
                ma = sum(closes[-p:]) / p
                result["averages"][f"MA_{p}"] = {
                    "value": round(ma, 2),
                    "vs_current": round((closes[-1] - ma) / ma * 100, 2)
                }

        # Golden/Death cross check
        if "MA_50" in result["averages"] and "MA_10" in result["averages"]:
            if result["averages"]["MA_10"]["value"] > result["averages"]["MA_50"]["value"]:
                result["cross_signal"] = "Short MA above Long MA — potentially bullish"
            else:
                result["cross_signal"] = "Short MA below Long MA — potentially bearish"

        return result

    def _export_csv(self, records, closes, params) -> dict:
        filepath = params.get("filepath", "outputs/stock_data.csv")
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else "outputs", exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
            writer.writeheader()
            writer.writerows(records)

        return {
            "operation": "export_csv",
            "filepath": filepath,
            "rows_written": len(records),
            "status": "success"
        }


# ============================================================
# TOOL 3: Report Formatter Tool (Communication / Output)
# ============================================================

class ReportFormatterInput(BaseModel):
    """Input for report formatting."""
    title: str = Field(..., description="Report title")
    sections: str = Field(
        ...,
        description="JSON string of sections: [{\"heading\": \"...\", \"content\": \"...\"}]"
    )
    format_type: str = Field(
        default="markdown",
        description="Output format: 'markdown', 'json', or 'text'"
    )
    output_path: Optional[str] = Field(
        default=None,
        description="Optional file path to save the report"
    )


class ReportFormatterTool(BaseTool):
    """
    Built-in Tool #3: Formats analysis results into structured reports.
    Supports markdown, JSON, and plain text output formats.
    """
    name: str = "report_formatter"
    description: str = (
        "Formats stock analysis results into a professional report. "
        "Input: title (str), sections (JSON list of {heading, content}), "
        "format_type ('markdown'/'json'/'text'), optional output_path to save."
    )
    args_schema: Type[BaseModel] = ReportFormatterInput

    def _run(self, title: str, sections: str, format_type: str = "markdown",
             output_path: Optional[str] = None) -> str:
        try:
            section_list = json.loads(sections)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if format_type == "markdown":
                report = self._format_markdown(title, section_list, timestamp)
            elif format_type == "json":
                report = self._format_json(title, section_list, timestamp)
            else:
                report = self._format_text(title, section_list, timestamp)

            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "outputs", exist_ok=True)
                with open(output_path, "w") as f:
                    f.write(report)
                return json.dumps({
                    "status": "success",
                    "format": format_type,
                    "output_path": output_path,
                    "preview": report[:500] + "..." if len(report) > 500 else report
                })

            return report

        except Exception as e:
            return json.dumps({"error": f"Formatting error: {str(e)}"})

    def _format_markdown(self, title, sections, timestamp) -> str:
        lines = [
            f"# {title}",
            f"*Generated: {timestamp}*",
            "",
            "---",
            ""
        ]
        for sec in sections:
            lines.append(f"## {sec.get('heading', 'Section')}")
            lines.append("")
            lines.append(sec.get("content", ""))
            lines.append("")
        lines.append("---")
        lines.append(f"*Report generated by Stock Research Multi-Agent System*")
        return "\n".join(lines)

    def _format_json(self, title, sections, timestamp) -> str:
        return json.dumps({
            "title": title,
            "generated_at": timestamp,
            "sections": sections,
            "meta": {"generator": "Stock Research Multi-Agent System"}
        }, indent=2)

    def _format_text(self, title, sections, timestamp) -> str:
        lines = [
            "=" * 60,
            title.center(60),
            f"Generated: {timestamp}".center(60),
            "=" * 60,
            ""
        ]
        for sec in sections:
            lines.append(f"--- {sec.get('heading', 'Section')} ---")
            lines.append(sec.get("content", ""))
            lines.append("")
        return "\n".join(lines)
