"""
Task Definitions & Crew Orchestration
=======================================
Defines tasks for each agent and creates the CrewAI Crew
with sequential orchestration. Integrates RAG context retrieval
and prompt engineering version selection.
"""

from typing import Optional
from crewai import Task, Crew, Process
from agents.agent_definitions import create_agents
from rag.retriever import get_retriever


def retrieve_rag_context(stock_symbol: str, top_k: int = 3) -> tuple[list[dict], str]:
    """
    Pre-retrieve RAG context for a stock analysis workflow.

    Returns:
        Tuple of (raw_results, formatted_context_string)
    """
    retriever = get_retriever()
    results = retriever.retrieve(
        query=f"{stock_symbol} financial analysis earnings revenue outlook",
        ticker=stock_symbol,
        top_k=top_k,
    )
    context_str = retriever.format_context(results)
    return results, context_str


def create_tasks(agents: dict, stock_symbol: str,
                 analysis_period: int = 90,
                 rag_context: str = "") -> list[Task]:
    """
    Create the task pipeline for stock research.

    Tasks follow a sequential workflow:
    1. Data Collection → 2. Technical Analysis → 3. Report Generation

    Args:
        agents: Dict of Agent instances
        stock_symbol: Ticker symbol
        analysis_period: Days of historical data
        rag_context: Formatted RAG context string to inject into task descriptions
    """
    # Build RAG context block for task descriptions
    context_block = ""
    if rag_context:
        context_block = (
            f"\n\n## Available Financial Context\n"
            f"The following context was retrieved from the financial knowledge base. "
            f"Use it to enrich your analysis where relevant:\n\n"
            f"{rag_context}\n"
        )

    # ---- Task 1: Data Collection ----
    data_collection_task = Task(
        description=(
            f"Collect and prepare stock market data for {stock_symbol}.\n\n"
            f"Steps:\n"
            f"1. Use the stock_data_retriever tool to fetch {analysis_period} days of "
            f"   historical data for {stock_symbol}.\n"
            f"2. Use the data_processor tool with operation='summary_stats' to compute "
            f"   basic statistics on the retrieved data.\n"
            f"3. Use the data_processor tool with operation='returns' to calculate "
            f"   return metrics.\n"
            f"4. Use the data_processor tool with operation='volatility' to assess risk.\n"
            f"5. Optionally, use the financial_knowledge_search tool to find relevant "
            f"   company fundamentals or earnings data for {stock_symbol}.\n\n"
            f"Return ALL results as a structured JSON summary."
            f"{context_block}"
        ),
        expected_output=(
            "A comprehensive JSON containing: raw price data, summary statistics, "
            "return metrics, volatility analysis, and any relevant fundamental context "
            "for the requested stock."
        ),
        agent=agents["data_collector"],
    )

    # ---- Task 2: Technical Analysis ----
    technical_analysis_task = Task(
        description=(
            f"Perform technical analysis on {stock_symbol} using the data from the "
            f"previous task.\n\n"
            f"Steps:\n"
            f"1. Extract the closing prices from the collected data.\n"
            f"2. Use the technical_indicator_calculator tool to calculate:\n"
            f"   - RSI (period=14)\n"
            f"   - MACD (fast=12, slow=26, signal=9)\n"
            f"   - SMA (period=20)\n"
            f"   - EMA (period=20)\n"
            f"   - Bollinger Bands (period=20, std=2)\n"
            f"3. Optionally, use the financial_knowledge_search tool to find context "
            f"   about how to interpret these indicators for {stock_symbol}'s sector.\n"
            f"4. Synthesize all indicator signals into an overall market assessment.\n"
            f"5. Determine overall trend: BULLISH, BEARISH, or NEUTRAL based on "
            f"   indicator consensus.\n\n"
            f"Provide the closing prices as a JSON list string to the tool."
            f"{context_block}"
        ),
        expected_output=(
            "A comprehensive technical analysis report including all indicator values, "
            "individual signals, an overall consensus assessment with reasoning, and "
            "any relevant fundamental context that supports the analysis."
        ),
        agent=agents["analyst"],
        context=[data_collection_task],
    )

    # ---- Task 3: Report Generation ----
    report_task = Task(
        description=(
            f"Generate a professional stock research report for {stock_symbol}.\n\n"
            f"Using the data collection results and technical analysis from previous tasks, "
            f"create a comprehensive report with these sections:\n"
            f"1. Executive Summary - Key findings and recommendation\n"
            f"2. Company & Market Context - Relevant fundamental data (if available)\n"
            f"3. Price Overview - Current price, range, and basic statistics\n"
            f"4. Technical Analysis - All indicator results and signals\n"
            f"5. Risk Assessment - Volatility, drawdown, and risk level\n"
            f"6. Conclusion - Overall outlook and key levels to watch\n\n"
            f"If you have fundamental context from the knowledge base, include it in "
            f"section 2 with proper source citations.\n\n"
            f"Use the report_formatter tool with format_type='markdown' and "
            f"output_path='outputs/{stock_symbol}_research_report.md' to save the report.\n\n"
            f"End the report with: *Disclaimer: This report is for informational purposes "
            f"only and does not constitute investment advice.*"
            f"{context_block}"
        ),
        expected_output=(
            f"A professional markdown research report saved to "
            f"outputs/{stock_symbol}_research_report.md with all required sections "
            f"and proper citations for any external data referenced."
        ),
        agent=agents["report_writer"],
        context=[data_collection_task, technical_analysis_task],
    )

    return [data_collection_task, technical_analysis_task, report_task]


def create_crew(stock_symbol: str, analysis_period: int = 90,
                llm=None, prompt_version: str = "v3_cot_rag") -> Crew:
    """
    Create and configure the complete research Crew.

    Args:
        stock_symbol: Ticker symbol to analyze
        analysis_period: Days of historical data
        llm: Optional LLM instance
        prompt_version: Prompt template version
    """
    # Pre-retrieve RAG context
    rag_results, rag_context = retrieve_rag_context(stock_symbol)

    # Create agents with prompt engineering and RAG context
    agents = create_agents(
        llm=llm,
        prompt_version=prompt_version,
        symbol=stock_symbol,
        rag_results=rag_results,
    )

    # Create tasks with RAG context injection
    tasks = create_tasks(agents, stock_symbol, analysis_period, rag_context)

    crew = Crew(
        agents=[
            agents["data_collector"],
            agents["analyst"],
            agents["report_writer"],
        ],
        tasks=tasks,
        process=Process.sequential,
        manager_agent=agents["controller"],
        verbose=True,
        memory=False,
        planning=False,
    )

    return crew


def run_research(stock_symbol: str, analysis_period: int = 90,
                 llm=None, prompt_version: str = "v3_cot_rag") -> str:
    """
    Main entry point to run a complete stock research workflow.

    Args:
        stock_symbol: Stock ticker (e.g. "AAPL")
        analysis_period: Days of historical data (default 90)
        llm: Optional LLM instance
        prompt_version: Prompt template version

    Returns:
        The final crew output as a string.
    """
    print(f"\n{'='*60}")
    print(f"  AI-Powered Stock Research Platform")
    print(f"  Analyzing: {stock_symbol} | Period: {analysis_period} days")
    print(f"  Prompt Version: {prompt_version}")
    print(f"{'='*60}\n")

    crew = create_crew(stock_symbol, analysis_period, llm=llm,
                       prompt_version=prompt_version)
    result = crew.kickoff()

    print(f"\n{'='*60}")
    print(f"  Research Complete for {stock_symbol}")
    print(f"{'='*60}\n")

    return result
