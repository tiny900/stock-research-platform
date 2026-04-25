"""
Vector Store — ChromaDB Wrapper with TF-IDF Fallback
=====================================================
Provides vector storage and similarity search for RAG.
Uses ChromaDB when available, falls back to TF-IDF keyword search.
"""

import json
import os
from typing import Optional

from rag.knowledge_base import DocumentChunk


class VectorStore:
    """Unified interface for vector storage with automatic backend selection."""

    def __init__(self, persist_dir: str = "data/chroma_db", collection_name: str = "financial_knowledge"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        """Try ChromaDB first, fall back to TF-IDF."""
        try:
            import chromadb
            os.makedirs(self.persist_dir, exist_ok=True)
            client = chromadb.PersistentClient(path=self.persist_dir)
            self._backend = ChromaBackend(client, self.collection_name)
        except ImportError:
            self._backend = TFIDFBackend()

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def add_chunks(self, chunks: list[DocumentChunk]):
        """Add document chunks to the store."""
        self._backend.add_chunks(chunks)

    def search(self, query: str, top_k: int = 5,
               ticker_filter: Optional[str] = None) -> list[dict]:
        """
        Search for relevant chunks with optional ticker pre-filtering.

        When a ticker is provided, uses a two-phase approach:
        1. First, search only within chunks tagged with that ticker
        2. If not enough results, fill remaining slots with general semantic search
           (excluding already-returned chunks)
        """
        if not ticker_filter:
            return self._backend.search(query, top_k=top_k)

        ticker_upper = ticker_filter.upper()

        # Phase 1: Search within ticker-specific chunks only
        ticker_results = self._backend.search_with_ticker(query, ticker_upper, top_k=top_k)

        if len(ticker_results) >= top_k:
            return ticker_results[:top_k]

        # Phase 2: Fill remaining slots with general semantic search
        remaining = top_k - len(ticker_results)
        seen_contents = {r["content"][:100] for r in ticker_results}
        general_results = self._backend.search(query, top_k=top_k + remaining)

        for r in general_results:
            if len(ticker_results) >= top_k:
                break
            if r["content"][:100] not in seen_contents:
                ticker_results.append(r)
                seen_contents.add(r["content"][:100])

        return ticker_results[:top_k]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._backend.count()

    def reset(self):
        """Clear all stored data."""
        self._backend.reset()


class ChromaBackend:
    """ChromaDB-based vector store backend."""
    name = "ChromaDB"

    def __init__(self, client, collection_name: str):
        self.client = client
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: list[DocumentChunk]):
        if not chunks:
            return
        ids = []
        documents = []
        metadatas = []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            meta = {k: (json.dumps(v) if isinstance(v, list) else str(v))
                    for k, v in chunk.metadata.items()}
            metadatas.append(meta)

        # Batch add (ChromaDB limit: 41666 per batch)
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

    def _parse_results(self, results) -> list[dict]:
        """Parse ChromaDB query results into standardized dicts."""
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                relevance = max(0.0, 1.0 - distance)
                if "tickers" in meta:
                    try:
                        meta["tickers"] = json.loads(meta["tickers"])
                    except (json.JSONDecodeError, TypeError):
                        meta["tickers"] = []
                output.append({
                    "content": doc,
                    "metadata": meta,
                    "relevance_score": round(relevance, 4),
                })
        return output

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=min(top_k, self.count() or 1))
        return self._parse_results(results)

    def search_with_ticker(self, query: str, ticker: str, top_k: int = 5) -> list[dict]:
        """Search only chunks whose tickers metadata contains the given ticker."""
        total = self.count()
        if total == 0:
            return []
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, total),
                where={"tickers": {"$contains": ticker}},
            )
            return self._parse_results(results)
        except Exception:
            # Fallback: fetch more results and filter in Python
            results = self.collection.query(query_texts=[query], n_results=min(total, top_k * 4))
            all_parsed = self._parse_results(results)
            return [r for r in all_parsed if ticker in r.get("metadata", {}).get("tickers", [])][:top_k]

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            # Collection may have been deleted externally
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            return self.collection.count()

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )


class TFIDFBackend:
    """TF-IDF keyword search fallback when ChromaDB is not available."""
    name = "TF-IDF (fallback)"

    def __init__(self):
        self._chunks: list[DocumentChunk] = []
        self._vectorizer = None
        self._matrix = None

    def add_chunks(self, chunks: list[DocumentChunk]):
        self._chunks.extend(chunks)
        self._build_index()

    def _build_index(self):
        if not self._chunks:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            texts = [c.content for c in self._chunks]
            self._matrix = self._vectorizer.fit_transform(texts)
        except ImportError:
            # Pure keyword fallback
            self._vectorizer = None
            self._matrix = None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._chunks:
            return []

        if self._vectorizer is not None and self._matrix is not None:
            return self._tfidf_search(query, top_k)
        return self._keyword_search(query, top_k)

    def search_with_ticker(self, query: str, ticker: str, top_k: int = 5) -> list[dict]:
        """Search only within chunks tagged with the given ticker."""
        all_results = self.search(query, top_k=top_k * 3)
        return [r for r in all_results
                if ticker in r.get("metadata", {}).get("tickers", [])][:top_k]

    def _tfidf_search(self, query: str, top_k: int) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix).flatten()

        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                chunk = self._chunks[idx]
                results.append({
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "relevance_score": round(float(similarities[idx]), 4),
                })
        return results

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """Pure Python keyword matching as ultimate fallback."""
        query_terms = set(query.lower().split())
        scored = []
        for chunk in self._chunks:
            content_lower = chunk.content.lower()
            matches = sum(1 for t in query_terms if t in content_lower)
            if matches > 0:
                score = matches / max(len(query_terms), 1)
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "relevance_score": round(score, 4),
            }
            for score, chunk in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._chunks)

    def reset(self):
        self._chunks = []
        self._vectorizer = None
        self._matrix = None
