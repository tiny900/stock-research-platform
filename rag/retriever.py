"""
Retriever — Unified RAG Retrieval Interface
=============================================
Handles the full retrieval pipeline: query → search → re-rank → format results.
"""

import os
from typing import Optional

from rag.knowledge_base import load_and_chunk
from rag.vector_store import VectorStore


class FinancialRetriever:
    """High-level retriever that manages the knowledge base and vector store."""

    def __init__(self, knowledge_base_path: str = "data/knowledge_base",
                 persist_dir: str = "data/chroma_db",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        self.knowledge_base_path = knowledge_base_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store = VectorStore(persist_dir=persist_dir)
        self._initialized = False

    def initialize(self, force: bool = False):
        """Load and index the knowledge base. Idempotent if already indexed."""
        try:
            if not force and self._initialized and self.store.count() > 0:
                return
        except Exception:
            pass  # Collection may have been deleted; re-initialize

        chunks = load_and_chunk(
            base_path=self.knowledge_base_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        if chunks:
            self.store.reset()
            self.store.add_chunks(chunks)
        self._initialized = True

    def retrieve(self, query: str, ticker: Optional[str] = None,
                 top_k: int = 3) -> list[dict]:
        """
        Retrieve relevant financial knowledge for a query.

        Args:
            query: The search query (e.g., "AAPL revenue growth outlook")
            ticker: Optional ticker symbol to boost matching documents
            top_k: Number of results to return

        Returns:
            List of dicts with 'content', 'source', 'relevance_score', 'citation'
        """
        self.initialize()

        if self.store.count() == 0:
            return []

        raw_results = self.store.search(query, top_k=top_k, ticker_filter=ticker)

        formatted = []
        for i, r in enumerate(raw_results):
            meta = r.get("metadata", {})
            source_file = meta.get("source_file", "unknown")
            section = meta.get("section", "")
            doc_type = meta.get("doc_type", "")

            formatted.append({
                "content": r["content"],
                "source": source_file,
                "section": section,
                "doc_type": doc_type,
                "relevance_score": r.get("relevance_score", 0.0),
                "citation": f"[Source: {source_file}" + (f", Section: {section}]" if section else "]"),
                "rank": i + 1,
            })

        return formatted

    def format_context(self, results: list[dict], max_chars: int = 3000) -> str:
        """Format retrieval results as context string for LLM prompt injection."""
        if not results:
            return ""

        lines = ["### Retrieved Financial Context\n"]
        total_chars = 0
        for r in results:
            entry = (
                f"**[{r['rank']}] {r['source']}** (relevance: {r['relevance_score']:.2f})\n"
                f"> {r['content'][:500]}\n"
                f"— {r['citation']}\n"
            )
            if total_chars + len(entry) > max_chars:
                break
            lines.append(entry)
            total_chars += len(entry)

        return "\n".join(lines)

    @property
    def backend_name(self) -> str:
        return self.store.backend_name

    @property
    def document_count(self) -> int:
        self.initialize()
        return self.store.count()


# Module-level singleton for convenience
_default_retriever: Optional[FinancialRetriever] = None


def get_retriever(knowledge_base_path: str = "data/knowledge_base") -> FinancialRetriever:
    """Get or create the default retriever instance."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = FinancialRetriever(knowledge_base_path=knowledge_base_path)
    return _default_retriever
