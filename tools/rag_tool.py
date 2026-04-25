"""
RAG Tool — Financial Knowledge Search for CrewAI Agents
========================================================
CrewAI BaseTool that wraps the RAG retriever to provide agents
with access to the financial knowledge base.
"""

import json
from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from rag.retriever import get_retriever


class FinancialKnowledgeInput(BaseModel):
    """Input schema for the financial knowledge search tool."""
    query: str = Field(
        ...,
        description="Search query for the financial knowledge base, e.g. 'AAPL revenue growth' or 'what is RSI'"
    )
    ticker: Optional[str] = Field(
        default=None,
        description="Optional stock ticker to prioritize relevant documents, e.g. 'AAPL'"
    )
    top_k: int = Field(
        default=3,
        description="Number of results to return (1-10)"
    )


class FinancialKnowledgeSearchTool(BaseTool):
    """
    RAG Tool: Searches the financial knowledge base for relevant context.
    Retrieves SEC filings, earnings transcripts, financial glossary terms,
    and market analysis reports using semantic search.
    """
    name: str = "financial_knowledge_search"
    description: str = (
        "Searches a curated financial knowledge base containing SEC filings, "
        "earnings call transcripts, financial glossary terms, and market analysis reports. "
        "Use this tool to find background context, company fundamentals, financial term definitions, "
        "or market insights relevant to the stock being analyzed. "
        "Input: query (str), optional ticker (str), optional top_k (int, default 3)."
    )
    args_schema: Type[BaseModel] = FinancialKnowledgeInput

    def _run(self, query: str, ticker: Optional[str] = None, top_k: int = 3) -> str:
        """Search the financial knowledge base."""
        try:
            top_k = max(1, min(top_k, 10))
            retriever = get_retriever()
            results = retriever.retrieve(query=query, ticker=ticker, top_k=top_k)

            if not results:
                return json.dumps({
                    "query": query,
                    "results_count": 0,
                    "results": [],
                    "message": "No relevant documents found in the knowledge base."
                }, indent=2)

            output_results = []
            for r in results:
                output_results.append({
                    "rank": r["rank"],
                    "source": r["source"],
                    "section": r["section"],
                    "doc_type": r["doc_type"],
                    "relevance_score": r["relevance_score"],
                    "content": r["content"][:800],
                    "citation": r["citation"],
                })

            return json.dumps({
                "query": query,
                "ticker_filter": ticker,
                "results_count": len(output_results),
                "retrieval_backend": retriever.backend_name,
                "results": output_results,
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "error": f"Knowledge base search failed: {str(e)}",
                "query": query,
            }, indent=2)
