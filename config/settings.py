"""
Configuration for the AI-Powered Stock Research Platform
"""

import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "openai"),
    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "base_url": os.getenv("LLM_BASE_URL", None),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
}

# Analysis Defaults
DEFAULT_ANALYSIS_PERIOD = 90
DEFAULT_INDICATORS = ["RSI", "MACD", "SMA", "EMA", "BOLLINGER"]

# Prompt Engineering
DEFAULT_PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v3_cot_rag")
PROMPT_VERSIONS = ["v1_basic", "v2_structured", "v3_cot_rag"]

# RAG Configuration
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# Output Configuration
OUTPUT_DIR = "outputs"
REPORT_FORMAT = "markdown"

# Demo Mode
DEMO_STOCKS = ["AAPL", "TSLA", "MSFT", "NVDA", "GOOGL"]
