"""
Tests for the RAG (Retrieval-Augmented Generation) pipeline.
Tests document loading, chunking, vector storage, and retrieval quality.
"""

import json
import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDocumentLoader:
    """Tests for knowledge base document loading."""

    def test_load_all_returns_documents(self):
        from rag.knowledge_base import DocumentLoader
        loader = DocumentLoader("data/knowledge_base")
        docs = loader.load_all()
        assert len(docs) > 0, "Should load at least one document"

    def test_documents_have_required_fields(self):
        from rag.knowledge_base import DocumentLoader
        loader = DocumentLoader("data/knowledge_base")
        docs = loader.load_all()
        for doc in docs:
            assert "content" in doc, "Document must have content"
            assert "source_file" in doc, "Document must have source_file"
            assert "doc_type" in doc, "Document must have doc_type"
            assert len(doc["content"]) > 0, "Document content must not be empty"

    def test_ticker_extraction_from_filename(self):
        from rag.knowledge_base import DocumentLoader
        tickers = DocumentLoader._extract_tickers("AAPL_10K_2024.md")
        assert "AAPL" in tickers

    def test_ticker_extraction_no_match(self):
        from rag.knowledge_base import DocumentLoader
        tickers = DocumentLoader._extract_tickers("market_concepts.md")
        assert len(tickers) == 0

    def test_load_nonexistent_directory(self):
        from rag.knowledge_base import DocumentLoader
        loader = DocumentLoader("/nonexistent/path")
        docs = loader.load_all()
        assert docs == []


class TestDocumentChunker:
    """Tests for document chunking."""

    def test_chunks_are_produced(self):
        from rag.knowledge_base import load_and_chunk
        chunks = load_and_chunk("data/knowledge_base")
        assert len(chunks) > 0, "Should produce at least one chunk"

    def test_chunk_size_within_bounds(self):
        from rag.knowledge_base import load_and_chunk
        chunks = load_and_chunk("data/knowledge_base", chunk_size=500)
        oversized = [c for c in chunks if len(c.content) > 1500]
        # Allow some flexibility for hard-split edge cases
        assert len(oversized) < len(chunks) * 0.1, \
            f"Too many oversized chunks: {len(oversized)}/{len(chunks)}"

    def test_chunks_have_metadata(self):
        from rag.knowledge_base import load_and_chunk
        chunks = load_and_chunk("data/knowledge_base")
        for chunk in chunks[:10]:
            assert "source_file" in chunk.metadata
            assert "doc_type" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_ids_are_unique(self):
        from rag.knowledge_base import load_and_chunk
        chunks = load_and_chunk("data/knowledge_base")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_custom_chunk_parameters(self):
        from rag.knowledge_base import load_and_chunk
        small_chunks = load_and_chunk("data/knowledge_base", chunk_size=200, chunk_overlap=20)
        large_chunks = load_and_chunk("data/knowledge_base", chunk_size=1000, chunk_overlap=100)
        assert len(small_chunks) > len(large_chunks), \
            "Smaller chunk_size should produce more chunks"


class TestVectorStore:
    """Tests for vector store (ChromaDB or TF-IDF fallback)."""

    def test_store_initialization(self):
        from rag.vector_store import VectorStore
        store = VectorStore(persist_dir="data/test_chroma", collection_name="test_collection")
        assert store.backend_name in ["ChromaDB", "TF-IDF (fallback)"]

    def test_add_and_search(self):
        from rag.vector_store import VectorStore
        from rag.knowledge_base import DocumentChunk

        store = VectorStore(persist_dir="data/test_chroma", collection_name="test_search")
        store.reset()

        chunks = [
            DocumentChunk("Apple reported strong iPhone sales growth", {"tickers": ["AAPL"], "chunk_index": 0, "source_file": "test1.md"}),
            DocumentChunk("Microsoft Azure cloud revenue grew 30 percent", {"tickers": ["MSFT"], "chunk_index": 1, "source_file": "test2.md"}),
            DocumentChunk("Tesla delivered 1.8 million electric vehicles", {"tickers": ["TSLA"], "chunk_index": 2, "source_file": "test3.md"}),
        ]
        store.add_chunks(chunks)
        assert store.count() == 3

        results = store.search("Apple iPhone sales", top_k=2)
        assert len(results) > 0
        assert results[0]["relevance_score"] > 0

    def test_ticker_filter_boost(self):
        from rag.vector_store import VectorStore
        from rag.knowledge_base import DocumentChunk

        store = VectorStore(persist_dir="data/test_chroma", collection_name="test_filter")
        store.reset()

        chunks = [
            DocumentChunk("Strong revenue growth in technology sector", {"tickers": ["AAPL"], "chunk_index": 0, "source_file": "a.md"}),
            DocumentChunk("Strong revenue growth in technology sector", {"tickers": ["MSFT"], "chunk_index": 1, "source_file": "b.md"}),
        ]
        store.add_chunks(chunks)

        results = store.search("revenue growth", top_k=2, ticker_filter="AAPL")
        aapl_result = [r for r in results if "AAPL" in str(r.get("metadata", {}).get("tickers", []))]
        assert len(aapl_result) > 0, "AAPL document should appear in results"

    def test_empty_store_search(self):
        from rag.vector_store import VectorStore
        store = VectorStore(persist_dir="data/test_chroma", collection_name="test_empty")
        store.reset()
        results = store.search("anything", top_k=3)
        assert results == []


class TestRetriever:
    """Tests for the high-level retriever."""

    def test_retriever_initialization(self):
        from rag.retriever import FinancialRetriever
        retriever = FinancialRetriever()
        retriever.initialize()
        assert retriever.document_count > 0

    def test_retrieve_returns_results(self):
        from rag.retriever import FinancialRetriever
        retriever = FinancialRetriever()
        results = retriever.retrieve("AAPL revenue growth", ticker="AAPL")
        assert len(results) > 0

    def test_result_format(self):
        from rag.retriever import FinancialRetriever
        retriever = FinancialRetriever()
        results = retriever.retrieve("stock market volatility")
        for r in results:
            assert "content" in r
            assert "source" in r
            assert "relevance_score" in r
            assert "citation" in r
            assert "rank" in r

    def test_format_context(self):
        from rag.retriever import FinancialRetriever
        retriever = FinancialRetriever()
        results = retriever.retrieve("RSI indicator", top_k=2)
        context = retriever.format_context(results)
        assert "Retrieved Financial Context" in context
        assert len(context) > 50

    def test_empty_query_graceful(self):
        from rag.retriever import FinancialRetriever
        retriever = FinancialRetriever()
        results = retriever.retrieve("")
        # Should not crash, may return results or empty
        assert isinstance(results, list)


class TestRAGTool:
    """Tests for the CrewAI RAG tool wrapper."""

    def _ensure_initialized(self):
        """Ensure the default retriever is initialized before tool tests."""
        from rag.retriever import get_retriever
        r = get_retriever()
        r.initialize()

    def test_tool_returns_json(self):
        self._ensure_initialized()
        from tools.rag_tool import FinancialKnowledgeSearchTool
        tool = FinancialKnowledgeSearchTool()
        result = tool._run("AAPL earnings revenue")
        parsed = json.loads(result)
        assert "results_count" in parsed or "error" not in parsed
        assert isinstance(parsed, dict)

    def test_tool_with_ticker(self):
        self._ensure_initialized()
        from tools.rag_tool import FinancialKnowledgeSearchTool
        tool = FinancialKnowledgeSearchTool()
        result = json.loads(tool._run("earnings growth", ticker="AAPL"))
        assert result.get("ticker_filter") == "AAPL" or "error" not in result

    def test_tool_empty_query(self):
        from tools.rag_tool import FinancialKnowledgeSearchTool
        tool = FinancialKnowledgeSearchTool()
        result = json.loads(tool._run(""))
        assert isinstance(result, dict)


# Cleanup test chroma data
@pytest.fixture(autouse=True, scope="session")
def cleanup():
    yield
    import shutil
    test_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test_chroma")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
