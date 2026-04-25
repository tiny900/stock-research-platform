# System Evaluation Report

*Generated: 2026-04-15 15:44:40*

## RAG Retrieval Quality

| Query | Ticker | Results | Avg Score | Max Score | Time (ms) |
|-------|--------|---------|-----------|-----------|-----------|
| AAPL revenue growth earnings | AAPL | 5 | 0.4909 | 0.6100 | 1118.4 |
| Tesla electric vehicle deliveries | TSLA | 5 | 0.6180 | 0.7526 | 122.3 |
| What is RSI indicator | None | 5 | 0.4529 | 0.6689 | 121.8 |
| NVIDIA data center GPU revenue | NVDA | 5 | 0.7511 | 0.8688 | 138.0 |
| stock market risk assessment volatility | None | 5 | 0.4726 | 0.5804 | 122.8 |

**Average relevance score:** 0.5571
**Average retrieval time:** 324.7ms
**Backend:** ChromaDB

## Pipeline Performance

| Symbol | Data (ms) | Process (ms) | Indicators (ms) | RAG (ms) | Report (ms) | Total (ms) |
|--------|-----------|--------------|-----------------|----------|-------------|------------|
| AAPL | 634.42 | 0.55 | 0.54 | 118.53 | 0.46 | 754.5 |
| MSFT | 91.85 | 0.52 | 0.39 | 116.46 | 0.43 | 209.65 |
| TSLA | 84.6 | 0.53 | 0.38 | 120.89 | 0.45 | 206.85 |

## Prompt Engineering

| Version | Description | Analyst Backstory Length |
|---------|-------------|------------------------|
| v1_basic | Basic role/goal/backstory prompts with no enhancements | 184 chars |
| v2_structured | Structured prompts with explicit workflows and few-shot examples | 2548 chars |
| v3_cot_rag | Chain-of-thought reasoning + RAG context injection + few-shot examples | 3090 chars |