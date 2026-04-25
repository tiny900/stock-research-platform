"""
AI-Powered Stock Research Platform
====================================
Main entry point with RAG-enhanced analysis and prompt engineering.

Usage:
    python main.py                              # Interactive mode
    python main.py --symbol AAPL                # Analyze a specific stock
    python main.py --symbol TSLA --days 180     # Custom period
    python main.py --demo                       # Run demo (no API keys needed)
    python main.py --symbol AAPL --prompt v2_structured  # Specific prompt version

Requirements:
    pip install -r requirements.txt
"""

import argparse
import json
import os
import sys
from datetime import datetime


def setup_llm():
    """Configure the LLM based on environment variables."""
    from dotenv import load_dotenv
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", None)

    if provider == "deepseek" and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"
        from crewai import LLM
        return LLM(model="deepseek/deepseek-chat", api_key=api_key)
    elif provider == "openai" and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return None  # Use CrewAI default
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", api_key)
        if api_key:
            from crewai import LLM
            return LLM(model=f"anthropic/{model}", api_key=api_key)

    # Fallback checks
    if os.getenv("ANTHROPIC_API_KEY"):
        from crewai import LLM
        return LLM(model="anthropic/claude-sonnet-4-20250514",
                    api_key=os.environ["ANTHROPIC_API_KEY"])

    print("WARNING: No API key found. Set OPENAI_API_KEY or configure .env file.")
    print("Running in demo mode with limited functionality.\n")
    return None


def run_demo():
    """Run a demonstration of all tool capabilities including RAG, without LLM calls."""
    from tools.technical_indicators import TechnicalIndicatorCalculator
    from tools.builtin_tools import StockDataRetrieverTool, DataProcessorTool, ReportFormatterTool
    from tools.rag_tool import FinancialKnowledgeSearchTool
    from rag.retriever import get_retriever
    from prompts.strategies import build_agent_prompt, get_strategy_description
    from prompts.templates import PROMPT_VERSIONS

    print("\n" + "=" * 60)
    print("  DEMO MODE: AI-Powered Stock Research Platform")
    print("=" * 60)

    # 1. Data Retrieval
    print("\n--- Step 1: Retrieving Stock Data (AAPL) ---")
    retriever_tool = StockDataRetrieverTool()
    raw_data = retriever_tool._run("AAPL", 60)
    parsed_data = json.loads(raw_data)
    closes = [d["close"] for d in parsed_data["data"]]
    print(f"Retrieved {parsed_data['data_points']} data points (source: {parsed_data['source']})")
    print(f"Price range: ${min(closes):.2f} - ${max(closes):.2f}")

    # 2. Data Processing
    print("\n--- Step 2: Processing Data ---")
    processor = DataProcessorTool()
    stats = json.loads(processor._run(raw_data, "summary_stats"))
    returns = json.loads(processor._run(raw_data, "returns"))
    vol = json.loads(processor._run(raw_data, "volatility"))
    print(f"Mean: ${stats['price_stats']['mean']}, Std: ${stats['price_stats']['std_dev']}")
    print(f"Cumulative return: {returns['cumulative_return_pct']}%, Win rate: {returns['win_rate_pct']}%")
    print(f"Volatility: {vol['annualized_volatility_pct']}% ({vol['risk_level']})")

    # 3. Technical Indicators
    print("\n--- Step 3: Technical Analysis (Custom Tool) ---")
    indicator_tool = TechnicalIndicatorCalculator()
    price_json = json.dumps(closes)
    signals = {}
    for ind in ["RSI", "MACD", "SMA", "EMA", "BOLLINGER"]:
        result = json.loads(indicator_tool._run(price_json, ind))
        signal = result.get("signal", "N/A")
        signals[ind] = signal
        print(f"  {ind}: {signal} — {result.get('interpretation', '')[:80]}")

    # 4. RAG Knowledge Base Search
    print("\n--- Step 4: RAG Knowledge Base Search ---")
    rag_retriever = get_retriever()
    rag_retriever.initialize()
    print(f"Backend: {rag_retriever.backend_name}, Documents indexed: {rag_retriever.document_count}")

    rag_results = rag_retriever.retrieve("AAPL revenue growth earnings", ticker="AAPL", top_k=3)
    for r in rag_results:
        print(f"  [{r['rank']}] score={r['relevance_score']:.3f} | {r['source']}")
        print(f"      {r['content'][:100]}...")

    # 5. Prompt Engineering Demo
    print("\n--- Step 5: Prompt Engineering Versions ---")
    for version in PROMPT_VERSIONS:
        prompt = build_agent_prompt("analyst", version, symbol="AAPL", rag_results=rag_results)
        desc = get_strategy_description(version)
        print(f"  {version}: {desc}")
        print(f"    Role: {prompt['role']}")
        print(f"    Backstory length: {len(prompt['backstory'])} chars")

    # 6. Report Generation
    print("\n--- Step 6: Generating Report ---")
    formatter = ReportFormatterTool()
    rag_context = rag_retriever.format_context(rag_results)
    sections = json.dumps([
        {"heading": "Executive Summary",
         "content": f"AAPL shows a {signals.get('MACD', 'NEUTRAL')} technical profile."},
        {"heading": "Company Context",
         "content": f"Based on knowledge base: {rag_results[0]['content'][:200] if rag_results else 'N/A'}"},
        {"heading": "Price Overview",
         "content": f"Current: ${closes[-1]:.2f}, Range: ${min(closes):.2f}-${max(closes):.2f}"},
        {"heading": "Technical Analysis",
         "content": ", ".join(f"{k}: {v}" for k, v in signals.items())},
        {"heading": "Risk Assessment",
         "content": f"Volatility: {vol['annualized_volatility_pct']}% ({vol['risk_level']})"},
        {"heading": "Conclusion",
         "content": "Monitor key technical levels. *Disclaimer: Not investment advice.*"},
    ])

    os.makedirs("outputs", exist_ok=True)
    result = formatter._run("AAPL Stock Research Report", sections, "markdown",
                            "outputs/AAPL_demo_report.md")
    parsed_result = json.loads(result)
    print(f"Report saved to: {parsed_result.get('output_path', 'N/A')}")

    # Export CSV
    csv_result = json.loads(processor._run(raw_data, "export_csv",
                                           json.dumps({"filepath": "outputs/AAPL_data.csv"})))
    print(f"Data exported to: {csv_result.get('filepath', 'N/A')}")

    print("\n" + "=" * 60)
    print("  Demo Complete! Check 'outputs/' for results.")
    print("  Run 'streamlit run app.py' for the web interface.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Stock Research Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --demo                              # Demo without API keys
    python main.py --symbol AAPL                       # Analyze Apple
    python main.py --symbol TSLA --days 180            # Tesla, 180 days
    python main.py --symbol AAPL --prompt v2_structured  # Specific prompt version
    streamlit run app.py                               # Launch web UI
        """
    )
    parser.add_argument("--symbol", "-s", type=str, help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--days", "-d", type=int, default=90, help="Analysis period in days (default: 90)")
    parser.add_argument("--demo", action="store_true", help="Run demo mode (no API keys needed)")
    parser.add_argument("--prompt", "-p", type=str, default="v3_cot_rag",
                        choices=["v1_basic", "v2_structured", "v3_cot_rag"],
                        help="Prompt engineering version (default: v3_cot_rag)")

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.symbol:
        print("\n" + "=" * 60)
        print("  AI-Powered Stock Research Platform")
        print("=" * 60)
        print("\nOptions:")
        print("  1. Enter a stock symbol to analyze")
        print("  2. Run demo mode (no API keys needed)")
        print("  3. Launch web UI (streamlit)")
        print("  4. Exit")
        choice = input("\nYour choice (1/2/3/4): ").strip()

        if choice == "2":
            run_demo()
            return
        elif choice == "3":
            os.system("streamlit run app.py")
            return
        elif choice == "4":
            print("Goodbye!")
            return
        else:
            args.symbol = input("Enter stock symbol: ").strip().upper()
            if not args.symbol:
                print("No symbol provided. Exiting.")
                return
            days_input = input(f"Analysis period in days (default {args.days}): ").strip()
            if days_input.isdigit():
                args.days = int(days_input)

    llm = setup_llm()
    from agents.crew_orchestration import run_research
    result = run_research(args.symbol, args.days, llm=llm, prompt_version=args.prompt)
    print("\n--- Final Output ---")
    print(result)


if __name__ == "__main__":
    main()
