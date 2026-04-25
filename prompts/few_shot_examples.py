"""
Few-Shot Examples — Worked Examples for Each Agent Role
========================================================
Provides 2-3 high-quality input/output/reasoning examples per role
to improve consistency and output quality via in-context learning.
"""


ANALYST_EXAMPLES = [
    {
        "input": (
            "RSI-14: 72.5 (OVERBOUGHT), MACD: bullish crossover with positive histogram, "
            "SMA-20: price 3.2% above, EMA-20: price 2.8% above, "
            "Bollinger: price near upper band (%B = 0.88)"
        ),
        "output": (
            "Overall Assessment: CAUTIOUSLY BULLISH (Confidence: MEDIUM)\n"
            "4 of 5 indicators show bullish momentum, but RSI at 72.5 and price near upper "
            "Bollinger Band suggest the stock is extended. The bullish MACD crossover and price "
            "above both SMAs confirm the uptrend, but profit-taking risk is elevated.\n"
            "Key Levels: Support at SMA-20 ($183.20), Resistance at upper Bollinger ($192.50).\n"
            "Recommendation: Maintain positions but tighten stop-losses below SMA-20."
        ),
        "reasoning": (
            "While RSI >70 signals overbought, in a confirmed uptrend (price above SMA+EMA, "
            "MACD bullish), overbought conditions can persist. The key is whether MACD histogram "
            "is expanding or contracting. Combined with Bollinger upper band proximity, this "
            "suggests momentum is strong but approaching resistance."
        ),
    },
    {
        "input": (
            "RSI-14: 35.2 (NEUTRAL), MACD: bearish, signal line above MACD line, "
            "SMA-20: price 1.5% below, EMA-20: price 0.8% below, "
            "Bollinger: price near lower band (%B = 0.15)"
        ),
        "output": (
            "Overall Assessment: BEARISH (Confidence: MEDIUM)\n"
            "4 of 5 indicators show bearish signals. RSI at 35.2 is declining toward oversold "
            "territory. MACD is bearish with the signal line above. Price below both moving "
            "averages confirms downtrend. Bollinger %B at 0.15 suggests price is near the lower band.\n"
            "Key Levels: Support at lower Bollinger ($172.30), Resistance at SMA-20 ($178.50).\n"
            "Recommendation: Avoid new long positions. Watch for RSI divergence as potential reversal signal."
        ),
        "reasoning": (
            "The convergence of bearish signals across momentum (MACD), trend (SMA/EMA below), "
            "and volatility (lower Bollinger) indicators creates a strong bearish case. RSI at 35 "
            "is not yet oversold, so further downside is possible before a bounce."
        ),
    },
    {
        "input": (
            "RSI-14: 52.1 (NEUTRAL), MACD: flat near zero line, "
            "SMA-20: price 0.2% above, EMA-20: price 0.1% below, "
            "Bollinger: price in middle of bands (%B = 0.48)"
        ),
        "output": (
            "Overall Assessment: NEUTRAL (Confidence: HIGH)\n"
            "All indicators show mixed or neutral signals. RSI at 52 is perfectly balanced. "
            "MACD flat near zero shows no momentum. Price is essentially at both moving averages. "
            "Bollinger %B at 0.48 confirms price is in the middle of the range.\n"
            "Key Levels: Watch for breakout above upper Bollinger or breakdown below lower band.\n"
            "Recommendation: Wait for a clear directional signal before taking positions."
        ),
        "reasoning": (
            "When all indicators converge on neutral, it typically means the stock is in a "
            "consolidation phase. The narrow Bollinger Band width (if applicable) might suggest "
            "a volatility squeeze and upcoming breakout. Direction is uncertain — stay patient."
        ),
    },
]


REPORT_WRITER_EXAMPLES = [
    {
        "input": (
            "Stock: AAPL, Price: $185.50, Period: 90 days, Cumulative Return: +8.2%, "
            "Volatility: 22.5% (MEDIUM), RSI: 62 (NEUTRAL), MACD: BULLISH, Overall: BULLISH"
        ),
        "output": (
            "## Executive Summary\n"
            "Apple Inc. (AAPL) shows a bullish technical profile over the past 90 days, with "
            "an 8.2% cumulative return and constructive momentum signals. The MACD bullish "
            "crossover, combined with a neutral RSI of 62, suggests further upside potential "
            "with manageable risk.\n\n"
            "## Risk Assessment\n"
            "Annualized volatility of 22.5% places AAPL in the MEDIUM risk category. The "
            "maximum drawdown of 6.8% during the period is within normal parameters for "
            "a large-cap technology stock. Current RSI levels do not indicate extreme conditions.\n\n"
            "*Disclaimer: This report is for informational purposes only and does not "
            "constitute investment advice.*"
        ),
        "reasoning": (
            "The report leads with the key finding (bullish outlook), supports it with "
            "specific metrics (8.2% return, MACD signal), and provides honest risk context "
            "(22.5% volatility). The disclaimer is essential for ethical compliance."
        ),
    },
    {
        "input": (
            "Stock: TSLA, Price: $245.00, Period: 90 days, Cumulative Return: -12.5%, "
            "Volatility: 52.3% (HIGH), RSI: 38 (BEARISH), MACD: BEARISH, Overall: BEARISH"
        ),
        "output": (
            "## Executive Summary\n"
            "Tesla Inc. (TSLA) presents a bearish technical picture over the analyzed 90-day "
            "period, declining 12.5% with elevated volatility. Both RSI (38) and MACD signals "
            "indicate persistent selling pressure. The high annualized volatility of 52.3% "
            "underscores the risk profile.\n\n"
            "## Risk Assessment\n"
            "With annualized volatility of 52.3%, TSLA is classified as HIGH risk. The "
            "maximum drawdown reached 22.1% during the period. While RSI at 38 is approaching "
            "oversold territory, no reversal signals are confirmed yet.\n\n"
            "*Disclaimer: This report is for informational purposes only and does not "
            "constitute investment advice.*"
        ),
        "reasoning": (
            "Even when the outlook is bearish, the report remains professional and balanced. "
            "It states facts (12.5% decline, high volatility) and notes that RSI approaching "
            "oversold could lead to a bounce — this shows analytical depth rather than one-sided bias."
        ),
    },
]


DATA_COLLECTOR_EXAMPLES = [
    {
        "input": "Collect 90 days of data for AAPL",
        "output": (
            "Data collection complete for AAPL:\n"
            "- Retrieved 90 trading days of OHLCV data (source: yfinance)\n"
            "- Price range: $175.20 - $192.80\n"
            "- Current price: $185.50\n"
            "- Average daily volume: 52.3M shares\n"
            "- Data quality: PASS (no gaps, no anomalies)\n"
            "- Summary stats, returns, and volatility computed successfully"
        ),
        "reasoning": (
            "Report data completeness and quality first, then key metrics. "
            "Flag any data issues explicitly."
        ),
    },
]


FEW_SHOT_EXAMPLES = {
    "data_collector": DATA_COLLECTOR_EXAMPLES,
    "analyst": ANALYST_EXAMPLES,
    "report_writer": REPORT_WRITER_EXAMPLES,
    "controller": [],  # Controller doesn't need few-shot examples
}


def get_examples(agent_role: str) -> list[dict]:
    """Get few-shot examples for a given agent role."""
    return FEW_SHOT_EXAMPLES.get(agent_role, [])


def format_examples_for_prompt(agent_role: str, max_examples: int = 2) -> str:
    """Format few-shot examples as a prompt string."""
    examples = get_examples(agent_role)[:max_examples]
    if not examples:
        return ""

    lines = ["\n**Reference Examples:**\n"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"*Example {i}:*")
        lines.append(f"Input: {ex['input']}")
        lines.append(f"Output: {ex['output']}")
        if ex.get("reasoning"):
            lines.append(f"Reasoning: {ex['reasoning']}")
        lines.append("")

    return "\n".join(lines)
