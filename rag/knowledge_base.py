"""
Knowledge Base — Document Loading and Chunking
================================================
Loads financial documents from the knowledge base directory,
splits them into chunks, and prepares them for vector storage.
"""

import os
import re
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """A chunk of text with metadata."""
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        source = self.metadata.get("source_file", "unknown")
        idx = self.metadata.get("chunk_index", 0)
        return f"{source}::chunk_{idx}"


class DocumentLoader:
    """Loads markdown documents from the knowledge base directory."""

    def __init__(self, base_path: str = "data/knowledge_base"):
        self.base_path = base_path

    def load_all(self) -> list[dict]:
        """Load all documents from the knowledge base."""
        documents = []
        if not os.path.exists(self.base_path):
            return documents

        for root, _dirs, files in os.walk(self.base_path):
            for fname in sorted(files):
                if not fname.endswith(".md"):
                    continue
                filepath = os.path.join(root, fname)
                rel_path = os.path.relpath(filepath, self.base_path)
                doc_type = os.path.basename(os.path.dirname(filepath))
                tickers = self._extract_tickers(fname)

                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                documents.append({
                    "content": content,
                    "source_file": rel_path,
                    "doc_type": doc_type,
                    "tickers": tickers,
                    "filename": fname,
                })
        return documents

    @staticmethod
    def _extract_tickers(filename: str) -> list[str]:
        """Extract stock ticker symbols from filename."""
        known = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META", "NFLX", "AMD"]
        name_upper = filename.upper()
        return [t for t in known if t in name_upper]


class DocumentChunker:
    """Splits documents into chunks for vector storage."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: dict) -> list[DocumentChunk]:
        """Split a document into chunks using section-aware recursive splitting."""
        content = doc["content"]
        sections = self._split_by_sections(content)

        chunks = []
        for section_name, section_text in sections:
            section_chunks = self._recursive_split(section_text)
            for i, chunk_text in enumerate(section_chunks):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 20:
                    continue
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata={
                        "source_file": doc.get("source_file", ""),
                        "doc_type": doc.get("doc_type", ""),
                        "tickers": doc.get("tickers", []),
                        "section": section_name,
                        "chunk_index": len(chunks),
                    }
                ))
        return chunks

    def chunk_all(self, documents: list[dict]) -> list[DocumentChunk]:
        """Chunk all documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    @staticmethod
    def _split_by_sections(text: str) -> list[tuple[str, str]]:
        """Split text by markdown headers (## or ###)."""
        pattern = r"^(#{1,3})\s+(.+)$"
        lines = text.split("\n")
        sections = []
        current_name = "Introduction"
        current_lines = []

        for line in lines:
            match = re.match(pattern, line)
            if match:
                if current_lines:
                    sections.append((current_name, "\n".join(current_lines)))
                current_name = match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_name, "\n".join(current_lines)))

        return sections if sections else [("Full Document", text)]

    def _recursive_split(self, text: str) -> list[str]:
        """Recursively split text into chunks respecting natural boundaries."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separators = ["\n\n", "\n", ". ", " "]
        for sep in separators:
            parts = text.split(sep)
            if len(parts) <= 1:
                continue

            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) > self.chunk_size and current:
                    chunks.append(current)
                    # Overlap: keep tail of previous chunk
                    overlap_text = current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else ""
                    current = overlap_text + part
                else:
                    current = candidate

            if current.strip():
                chunks.append(current)

            if len(chunks) > 1:
                return chunks

        # Hard split as last resort
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks


def load_and_chunk(base_path: str = "data/knowledge_base",
                   chunk_size: int = 500,
                   chunk_overlap: int = 50) -> list[DocumentChunk]:
    """Convenience function: load all documents and chunk them."""
    loader = DocumentLoader(base_path)
    chunker = DocumentChunker(chunk_size, chunk_overlap)
    documents = loader.load_all()
    return chunker.chunk_all(documents)
