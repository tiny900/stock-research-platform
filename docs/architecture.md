# System Architecture

## Overview

The AI-Powered Stock Research Platform is a multi-agent system that combines technical analysis with RAG-enhanced knowledge retrieval and systematic prompt engineering to generate comprehensive stock research reports.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         Streamlit Web UI (app.py)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐│
│  │Dashboard │ │RAG       │ │Report    │ │Prompt Lab│ │Metrics ││
│  │          │ │Context   │ │Viewer    │ │          │ │        ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘│
└───────┼────────────┼────────────┼────────────┼────────────┼──────┘
        │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     CrewAI Agent Orchestration                    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Controller Agent (PPO)                    │ │
│  │         Orchestrates workflow + error handling                │ │
│  └──────────┬──────────────────┬──────────────────┬────────────┘ │
│             │                  │                  │               │
│  ┌──────────▼─────┐ ┌─────────▼────────┐ ┌──────▼───────────┐  │
│  │ Data Collector │ │ Technical Analyst │ │ Report Writer    │  │
│  │                │ │                  │ │                  │  │
│  │ • Stock Data   │ │ • Indicator Calc │ │ • Report Format  │  │
│  │ • Data Process │ │ • Data Process   │ │ • RAG Search     │  │
│  │ • RAG Search   │ │ • RAG Search     │ │                  │  │
│  └────────────────┘ └──────────────────┘ └──────────────────┘  │
│                                                                   │
│  Sequential Pipeline: Data → Analysis → Report                    │
└──────────────────────────────────────────────────────────────────┘
        │                                              │
        ▼                                              ▼
┌────────────────────┐                    ┌─────────────────────────┐
│ Tool Layer         │                    │ Prompt Engineering      │
│                    │                    │ Framework               │
│ • StockDataRetriever│                   │                         │
│ • DataProcessor    │                    │ • Template Registry     │
│ • TechIndicator    │                    │   (v1/v2/v3 per role)  │
│ • ReportFormatter  │                    │ • Few-Shot Examples     │
│ • RAG Search Tool  │                    │ • Strategy Composition  │
└────────────────────┘                    │   (CoT, RAG, Schema)   │
        │                                 └─────────────────────────┘
        ▼
┌──────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                                    │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Knowledge    │  │ Document     │  │ Vector Store         │   │
│  │ Base (15     │→ │ Chunker      │→ │ (ChromaDB /          │   │
│  │ markdown     │  │ (500 token   │  │  TF-IDF fallback)    │   │
│  │ documents)   │  │  chunks)     │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────┬───────────┘   │
│                                                   │               │
│                                      ┌────────────▼───────────┐  │
│                                      │ Retriever              │  │
│                                      │ (semantic search +     │  │
│                                      │  ticker re-ranking)    │  │
│                                      └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Agent Layer (CrewAI)
- **Framework**: CrewAI with sequential process orchestration
- **Agents**: 4 specialized agents with configurable prompts
- **Communication**: Task context chaining (output of task N → input of task N+1)

### 2. Tool Layer
- **5 tools** total (4 built-in + 1 RAG tool)
- Each tool is a CrewAI `BaseTool` subclass with Pydantic input validation
- All tools return structured JSON for reliable parsing

### 3. RAG Pipeline
- **Knowledge Base**: 15 curated markdown documents (glossary, SEC filings, earnings, market analysis)
- **Chunking**: Section-aware recursive splitting, 500-token chunks with 50-token overlap
- **Vector Store**: ChromaDB with cosine similarity (falls back to TF-IDF)
- **Retrieval**: Semantic search with ticker-based re-ranking, returns top-k with citations

### 4. Prompt Engineering
- **3 versions** per agent role: v1_basic, v2_structured, v3_cot_rag
- **Composable strategies**: Chain-of-thought, few-shot examples, RAG context injection
- **A/B testable**: Prompt Engineering Lab in the UI allows side-by-side comparison

### 5. Web UI (Streamlit)
- **5 tabs**: Dashboard, RAG Context, Report, Prompt Lab, Metrics
- **Interactive**: Real-time analysis with configurable parameters
- **Demo mode**: Works without API keys using synthetic data
