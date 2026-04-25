# AI-Powered Stock Research Platform

**Author:** Tianyu Zhang
**Course:** INFO 7375 — Generative AI Final Project
**University:** Northeastern University
**Date:** April 2026

An AI-powered stock research platform built with **CrewAI** multi-agent orchestration, **RAG** (Retrieval-Augmented Generation) knowledge retrieval, and a systematic **Prompt Engineering** framework. Features an interactive Streamlit web interface.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI (app.py)                    │
│  Dashboard │ RAG Context │ Report │ Prompt Lab │ System Metrics  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                  CrewAI Agent Orchestration                        │
│                                                                   │
│  ┌────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
│  │ Data       │→ │ Technical      │→ │ Report Writer          │  │
│  │ Collector  │  │ Analyst        │  │                        │  │
│  └────────────┘  └────────────────┘  └────────────────────────┘  │
│                  Controller Agent (Oversees workflow)              │
└──────────────────────────┬───────────────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    ▼                      ▼                      ▼
┌──────────┐     ┌──────────────────┐    ┌────────────────┐
│ 5 Tools  │     │ RAG Pipeline     │    │ Prompt Engine  │
│ • Stock  │     │ • ChromaDB       │    │ • 3 versions   │
│   Data   │     │ • 102 chunks     │    │ • CoT + RAG    │
│ • Process│     │ • Semantic search│    │ • Few-shot     │
│ • Indic. │     │ • TF-IDF fallbk  │    │ • Composable   │
│ • Report │     └──────────────────┘    └────────────────┘
│ • RAG    │
└──────────┘
```

## Core Components

### 1. RAG (Retrieval-Augmented Generation)
- **Knowledge Base**: 15 curated financial documents (SEC filings, earnings transcripts, glossary, market analysis)
- **Vector Store**: ChromaDB with cosine similarity search (102 indexed chunks)
- **Retrieval**: Semantic search with ticker-based re-ranking, returns top-k results with citations
- **Fallback**: TF-IDF keyword search when ChromaDB is unavailable

### 2. Prompt Engineering
- **3 versions** per agent role: `v1_basic`, `v2_structured`, `v3_cot_rag`
- **Composable strategies**: Chain-of-thought reasoning, few-shot examples, RAG context injection, output schema constraints
- **A/B testing**: Compare prompt versions side-by-side in the Prompt Engineering Lab

### 3. Multi-Agent System (CrewAI)
| Agent | Role | Tools |
|-------|------|-------|
| Controller | Orchestrates workflow | All 5 tools |
| Data Collector | Fetches & processes data | StockData, DataProcessor, RAG |
| Technical Analyst | Calculates indicators | TechIndicator, DataProcessor, RAG |
| Report Writer | Generates reports | ReportFormatter, RAG |

### 4. Technical Indicators (Custom Tool)
RSI, MACD, SMA, EMA, Bollinger Bands — with signal generation and consensus assessment.

## Setup

### 1. Install Dependencies
```bash
cd stock-research-platform
pip install -r requirements.txt
```

### 2. Configure (Optional)
```bash
cp .env.example .env
# Edit .env and add your LLM API key (OpenAI, DeepSeek, or Anthropic)
```

### 3. Run

```bash
# Demo mode — no API keys needed
python main.py --demo

# Launch web UI
streamlit run app.py

# Full analysis with LLM
python main.py --symbol AAPL
python main.py --symbol TSLA --days 180 --prompt v2_structured
```

## Testing

```bash
# Run all 60 tests
python -m pytest tests/ -v

# Generate evaluation report
python evaluation/metrics.py
```

## Project Structure

```
stock-research-platform/
├── main.py                          # CLI entry point
├── app.py                           # Streamlit web UI
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment template
├── agents/
│   ├── agent_definitions.py         # Agent roles with prompt framework
│   └── crew_orchestration.py        # Task pipeline with RAG integration
├── tools/
│   ├── builtin_tools.py             # 3 built-in tools (data, process, report)
│   ├── technical_indicators.py      # Custom tool (RSI, MACD, SMA, EMA, Bollinger)
│   └── rag_tool.py                  # RAG knowledge search tool
├── rag/
│   ├── knowledge_base.py            # Document loading & chunking
│   ├── vector_store.py              # ChromaDB wrapper + TF-IDF fallback
│   └── retriever.py                 # Semantic retrieval + re-ranking
├── prompts/
│   ├── templates.py                 # Versioned prompt templates (v1/v2/v3)
│   ├── few_shot_examples.py         # Worked examples per agent role
│   └── strategies.py                # Composable prompt strategies
├── config/
│   └── settings.py                  # Configuration
├── evaluation/
│   └── metrics.py                   # Performance & quality metrics
├── tests/
│   ├── test_standalone.py           # Tool unit tests (25+ assertions)
│   ├── test_rag.py                  # RAG pipeline tests
│   ├── test_prompts.py              # Prompt engineering tests
│   └── test_integration.py          # End-to-end integration tests
├── data/
│   ├── knowledge_base/              # 15 financial documents
│   │   ├── financial_glossary/      # Technical indicators, fundamentals, market concepts
│   │   ├── sec_filings/             # AAPL, MSFT, TSLA, NVDA, GOOGL 10-K excerpts
│   │   ├── earnings/                # Quarterly earnings call summaries
│   │   └── market_analysis/         # Sector outlook & analysis guides
│   └── demo_results/                # Pre-computed fixtures for demo mode
├── docs/
│   ├── architecture.md              # System architecture documentation
│   └── ethical_considerations.md    # Ethics & responsible AI
├── outputs/                         # Generated reports & data
└── web/
    └── index.html                   # GitHub Pages project site
```

## Evaluation Results

| Metric | Value |
|--------|-------|
| Total Tests | 60 passing |
| KB Documents | 15 files, 102 chunks |
| Avg RAG Relevance | 0.557 |
| Pipeline Latency | < 1 second |
| Prompt Versions | 3 (basic, structured, CoT+RAG) |

## Ethical Considerations

- All reports include mandatory investment disclaimer
- No personal data collection or external logging
- Knowledge base uses only public SEC filings and financial information
- Prompt Engineering Lab provides full transparency into LLM prompts
- See [docs/ethical_considerations.md](docs/ethical_considerations.md) for full details

## License

MIT — INFO 7375 Course Project, Northeastern University, Spring 2026
