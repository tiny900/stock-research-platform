"""
Agent Definitions for Stock Research Multi-Agent System
========================================================
Defines all agents with prompt engineering framework integration.
Supports three prompt versions: v1_basic, v2_structured, v3_cot_rag.

Agents:
1. Controller Agent - Orchestrates the workflow
2. Data Collector Agent - Gathers stock market data
3. Analyst Agent - Performs technical and fundamental analysis
4. Report Writer Agent - Generates structured reports
"""

from typing import Optional
from crewai import Agent
from tools.technical_indicators import TechnicalIndicatorCalculator
from tools.builtin_tools import (
    StockDataRetrieverTool,
    DataProcessorTool,
    ReportFormatterTool,
)
from tools.rag_tool import FinancialKnowledgeSearchTool
from prompts.strategies import build_agent_prompt


def create_agents(llm=None,
                  prompt_version: str = "v3_cot_rag",
                  symbol: str = "",
                  rag_results: Optional[list[dict]] = None) -> dict[str, Agent]:
    """
    Create and return all agents for the stock research system.

    Args:
        llm: Optional CrewAI LLM instance
        prompt_version: Prompt template version ('v1_basic', 'v2_structured', 'v3_cot_rag')
        symbol: Stock ticker symbol for context injection
        rag_results: Retrieved documents from RAG for context injection

    Returns:
        Dictionary mapping agent names to Agent instances.
    """
    # Instantiate tools
    stock_data_tool = StockDataRetrieverTool()
    data_processor_tool = DataProcessorTool()
    report_formatter_tool = ReportFormatterTool()
    technical_indicator_tool = TechnicalIndicatorCalculator()
    rag_tool = FinancialKnowledgeSearchTool()

    common_kwargs = {}
    if llm:
        common_kwargs["llm"] = llm

    # Build prompts using the prompt engineering framework
    controller_prompt = build_agent_prompt(
        "controller", prompt_version, symbol=symbol, rag_results=rag_results)
    collector_prompt = build_agent_prompt(
        "data_collector", prompt_version, symbol=symbol, rag_results=rag_results)
    analyst_prompt = build_agent_prompt(
        "analyst", prompt_version, symbol=symbol, rag_results=rag_results)
    writer_prompt = build_agent_prompt(
        "report_writer", prompt_version, symbol=symbol, rag_results=rag_results)

    # ---- Controller Agent ----
    controller = Agent(
        role=controller_prompt["role"],
        goal=controller_prompt["goal"],
        backstory=controller_prompt["backstory"],
        tools=[stock_data_tool, data_processor_tool, report_formatter_tool,
               technical_indicator_tool, rag_tool],
        verbose=True,
        allow_delegation=True,
        memory=False,
        **common_kwargs,
    )

    # ---- Data Collector Agent ----
    data_collector = Agent(
        role=collector_prompt["role"],
        goal=collector_prompt["goal"],
        backstory=collector_prompt["backstory"],
        tools=[stock_data_tool, data_processor_tool, rag_tool],
        verbose=True,
        memory=False,
        **common_kwargs,
    )

    # ---- Analyst Agent ----
    analyst = Agent(
        role=analyst_prompt["role"],
        goal=analyst_prompt["goal"],
        backstory=analyst_prompt["backstory"],
        tools=[technical_indicator_tool, data_processor_tool, rag_tool],
        verbose=True,
        memory=False,
        **common_kwargs,
    )

    # ---- Report Writer Agent ----
    report_writer = Agent(
        role=writer_prompt["role"],
        goal=writer_prompt["goal"],
        backstory=writer_prompt["backstory"],
        tools=[report_formatter_tool, rag_tool],
        verbose=True,
        memory=False,
        **common_kwargs,
    )

    return {
        "controller": controller,
        "data_collector": data_collector,
        "analyst": analyst,
        "report_writer": report_writer,
    }
