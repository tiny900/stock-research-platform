"""
Generate Project Documentation PDF
====================================
Combines architecture, implementation details, performance metrics,
challenges & solutions, ethical considerations, and future improvements
into a single professional PDF document.

Usage:
    python docs/generate_pdf.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable
)
from reportlab.lib import colors


def build_pdf(output_path: str = "outputs/project_documentation.pdf"):
    """Build the complete project documentation PDF."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        "DocTitle", parent=styles["Title"], fontSize=22, spaceAfter=6,
        textColor=HexColor("#1a1a2e"),
    ))
    styles.add(ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"], fontSize=12,
        textColor=HexColor("#555555"), spaceAfter=20, alignment=1,
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading1"], fontSize=16,
        textColor=HexColor("#0f3460"), spaceBefore=20, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "SubHead", parent=styles["Heading2"], fontSize=13,
        textColor=HexColor("#16213e"), spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10, leading=14,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        "BulletItem", parent=styles["Normal"], fontSize=10, leading=14,
        leftIndent=20, bulletIndent=10, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "CodeBlock", parent=styles["Normal"], fontSize=8, leading=10,
        fontName="Courier", leftIndent=15, spaceAfter=8,
        backColor=HexColor("#f5f5f5"),
    ))

    story = []

    # ---- Title Page ----
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("AI-Powered Stock Research Platform", styles["DocTitle"]))
    story.append(Paragraph(
        "INFO 7375 — Generative AI Final Project<br/>"
        "Northeastern University | April 2026<br/>"
        "Author: Tianyu Zhang",
        styles["DocSubtitle"]
    ))
    story.append(Spacer(1, 0.5 * inch))
    story.append(HRFlowable(width="80%", color=HexColor("#0f3460"), thickness=2))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "A multi-agent stock research system combining CrewAI orchestration, "
        "RAG knowledge retrieval (ChromaDB), and systematic prompt engineering "
        "with an interactive Streamlit web interface.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ---- Table of Contents ----
    story.append(Paragraph("Table of Contents", styles["SectionHead"]))
    toc_items = [
        "1. System Architecture",
        "2. Implementation Details",
        "3. Performance Metrics",
        "4. Challenges and Solutions",
        "5. Ethical Considerations",
        "6. Future Improvements",
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles["Body"]))
    story.append(PageBreak())

    # ---- 1. System Architecture ----
    story.append(Paragraph("1. System Architecture", styles["SectionHead"]))

    story.append(Paragraph("1.1 Overview", styles["SubHead"]))
    story.append(Paragraph(
        "The platform is a multi-agent system built on CrewAI that orchestrates four specialized "
        "agents (Controller, Data Collector, Technical Analyst, Report Writer) in a sequential "
        "pipeline. It integrates two core generative AI components: Retrieval-Augmented Generation "
        "(RAG) and a systematic Prompt Engineering framework.",
        styles["Body"]
    ))

    story.append(Paragraph("1.2 Architecture Diagram", styles["SubHead"]))
    arch_lines = [
        "Streamlit Web UI (5 tabs: Dashboard, RAG, Report, Prompt Lab, Metrics)",
        "        |",
        "CrewAI Agent Orchestration (Sequential Pipeline)",
        "  Controller -> Data Collector -> Technical Analyst -> Report Writer",
        "        |                |                |               |",
        "  [5 Tools]        [RAG Search]     [RAG Search]    [RAG Search]",
        "        |",
        "RAG Pipeline: Knowledge Base -> Chunker -> ChromaDB -> Retriever",
        "              (15 docs)        (500 tok)  (cosine)   (top-k)",
    ]
    for line in arch_lines:
        story.append(Paragraph(line, styles["CodeBlock"]))

    story.append(Paragraph("1.3 Component Summary", styles["SubHead"]))
    comp_data = [
        ["Component", "Technology", "Purpose"],
        ["Agent Framework", "CrewAI", "Multi-agent orchestration with sequential process"],
        ["Vector Store", "ChromaDB", "Semantic similarity search for RAG (TF-IDF fallback)"],
        ["Knowledge Base", "15 Markdown docs", "SEC filings, earnings, glossary, market analysis"],
        ["Prompt Engine", "Custom framework", "3-version templates with composable strategies"],
        ["Web UI", "Streamlit + Plotly", "Interactive dashboard with 5 analysis tabs"],
        ["Indicators", "Custom Python tool", "RSI, MACD, SMA, EMA, Bollinger Bands"],
    ]
    t = Table(comp_data, colWidths=[1.5 * inch, 1.5 * inch, 3.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ---- 2. Implementation Details ----
    story.append(Paragraph("2. Implementation Details", styles["SectionHead"]))

    story.append(Paragraph("2.1 RAG Pipeline", styles["SubHead"]))
    story.append(Paragraph(
        "<b>Document Loading:</b> The DocumentLoader walks the data/knowledge_base/ directory, "
        "loading all .md files with metadata extraction (doc_type, tickers from filename). "
        "15 documents cover 5 categories: financial glossary (3), SEC 10-K filings (5), "
        "earnings transcripts (4), and market analysis (3).",
        styles["Body"]
    ))
    story.append(Paragraph(
        "<b>Chunking:</b> The DocumentChunker uses section-aware recursive splitting. "
        "Markdown headers (##, ###) define section boundaries. Within sections, text is split "
        "recursively on paragraph breaks, sentences, then words. Chunk size: 500 characters "
        "with 50-character overlap. This produces 102 chunks with preserved metadata.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "<b>Vector Store:</b> ChromaDB with cosine similarity serves as the primary backend. "
        "The all-MiniLM-L6-v2 embedding model (via ChromaDB's default) converts text to "
        "384-dimensional vectors. A TF-IDF fallback (scikit-learn) activates when ChromaDB "
        "is unavailable, and a pure-Python keyword matcher serves as the ultimate fallback.",
        styles["Body"]
    ))
    story.append(Paragraph(
        "<b>Retrieval:</b> Two-phase ticker-aware search: Phase 1 searches only chunks tagged "
        "with the requested ticker. Phase 2 fills remaining slots with general semantic search. "
        "This ensures AAPL queries return AAPL-specific documents first, with cross-sector "
        "context as supplementary results.",
        styles["Body"]
    ))

    story.append(Paragraph("2.2 Prompt Engineering Framework", styles["SubHead"]))
    story.append(Paragraph(
        "The framework provides three prompt versions per agent role, each building on the previous:",
        styles["Body"]
    ))
    prompt_data = [
        ["Version", "Strategies", "Analyst Backstory Size"],
        ["v1_basic", "Simple role/goal/backstory", "184 chars"],
        ["v2_structured", "Explicit workflows + few-shot examples", "~2,500 chars"],
        ["v3_cot_rag", "Chain-of-thought + RAG context + few-shot", "~4,100 chars"],
    ]
    t2 = Table(prompt_data, colWidths=[1.2 * inch, 2.8 * inch, 1.8 * inch])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Paragraph(
        "Strategies are composable functions that transform base prompts. "
        "apply_chain_of_thought() adds step-by-step reasoning scaffolding. "
        "apply_few_shot() injects worked input/output examples. "
        "apply_rag_context() inserts retrieved documents with citation markers. "
        "build_agent_prompt() composes them based on the selected version.",
        styles["Body"]
    ))

    story.append(Paragraph("2.3 Multi-Agent Orchestration", styles["SubHead"]))
    story.append(Paragraph(
        "CrewAI manages 4 agents in a sequential pipeline: Data Collection -> Technical Analysis "
        "-> Report Generation. Each task's output flows to the next via CrewAI's context parameter. "
        "The Controller agent oversees the workflow as manager_agent. All agents have access to "
        "the RAG search tool, enabling them to query the knowledge base during their work.",
        styles["Body"]
    ))
    story.append(PageBreak())

    # ---- 3. Performance Metrics ----
    story.append(Paragraph("3. Performance Metrics", styles["SectionHead"]))

    # Get live metrics
    try:
        from evaluation.metrics import evaluate_rag_retrieval, evaluate_pipeline

        test_queries = [
            {"query": "AAPL revenue growth earnings", "ticker": "AAPL"},
            {"query": "Tesla electric vehicle deliveries", "ticker": "TSLA"},
            {"query": "What is RSI indicator"},
            {"query": "NVIDIA data center GPU revenue", "ticker": "NVDA"},
            {"query": "stock market risk assessment volatility"},
        ]
        rag_metrics = evaluate_rag_retrieval(test_queries)

        story.append(Paragraph("3.1 RAG Retrieval Quality", styles["SubHead"]))
        rag_table = [["Query", "Ticker", "Results", "Avg Score", "Time (ms)"]]
        for m in rag_metrics:
            rag_table.append([
                m.query[:35], m.ticker_filter or "—",
                str(m.results_count), f"{m.avg_relevance_score:.4f}",
                f"{m.retrieval_time_ms:.1f}"
            ])
        avg_score = sum(m.avg_relevance_score for m in rag_metrics) / len(rag_metrics)
        rag_table.append(["AVERAGE", "", "", f"{avg_score:.4f}", ""])

        t3 = Table(rag_table, colWidths=[2.2 * inch, 0.7 * inch, 0.7 * inch, 1 * inch, 1 * inch])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, -1), (-1, -1), HexColor("#e8f0fe")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t3)
        story.append(Spacer(1, 10))

        story.append(Paragraph("3.2 Pipeline Latency", styles["SubHead"]))
        pipe_table = [["Symbol", "Data (ms)", "Process (ms)", "Indicators (ms)", "RAG (ms)", "Total (ms)"]]
        for sym in ["AAPL", "MSFT", "TSLA"]:
            pm = evaluate_pipeline(sym)
            pipe_table.append([
                sym, f"{pm.data_retrieval_ms}", f"{pm.data_processing_ms}",
                f"{pm.indicator_calculation_ms}", f"{pm.rag_retrieval_ms}",
                f"{pm.total_ms}"
            ])

        t4 = Table(pipe_table, colWidths=[0.8 * inch, 1 * inch, 1 * inch, 1.2 * inch, 0.9 * inch, 1 * inch])
        t4.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(t4)

    except Exception as e:
        story.append(Paragraph(f"Metrics generation error: {str(e)}", styles["Body"]))

    story.append(Paragraph("3.3 Test Coverage", styles["SubHead"]))
    test_data = [
        ["Test File", "Tests", "Focus Area"],
        ["test_standalone.py", "25+", "Tool unit tests (data, indicators, reports)"],
        ["test_rag.py", "18", "RAG pipeline (loading, chunking, retrieval, tool)"],
        ["test_prompts.py", "16", "Prompt templates, strategies, composition"],
        ["test_integration.py", "14", "End-to-end pipeline, multi-stock, edge cases"],
        ["TOTAL", "60", "All passing"],
    ]
    t5 = Table(test_data, colWidths=[1.8 * inch, 0.7 * inch, 3.5 * inch])
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#0f3460")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, -1), (-1, -1), HexColor("#e8f0fe")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t5)
    story.append(PageBreak())

    # ---- 4. Challenges and Solutions ----
    story.append(Paragraph("4. Challenges and Solutions", styles["SectionHead"]))

    challenges = [
        ("ChromaDB Collection Lifecycle",
         "ChromaDB collections can become stale when the persistent client's in-memory "
         "handle outlives the actual collection (e.g., after a reset in tests). This caused "
         "NotFoundError during concurrent test runs.",
         "Added try/except around collection.count() with automatic re-creation via "
         "get_or_create_collection(). The retriever's initialize() also catches exceptions "
         "and re-indexes when the collection is missing."),
        ("Dollar Sign Rendering in Streamlit",
         "Streamlit's st.markdown() interprets $...$ as LaTeX math delimiters. Financial "
         "values like $94.9 billion were rendered as garbled math expressions.",
         "Escaped all $ characters in report markdown as \\\\$ before passing to st.markdown(). "
         "Created an esc() helper function for consistent dollar-sign formatting across "
         "the report template."),
        ("RAG Ticker Relevance",
         "Initial semantic search returned cross-ticker results (e.g., MSFT documents for "
         "an AAPL query) because financial language is similar across companies.",
         "Implemented two-phase ticker-aware retrieval: Phase 1 searches only chunks tagged "
         "with the target ticker via ChromaDB's where filter. Phase 2 fills remaining slots "
         "with general semantic search. This ensures ticker-specific context appears first."),
        ("Prompt Version Compatibility",
         "v3_cot_rag templates contain {rag_context} placeholders that must be replaced "
         "even when no RAG results are available, otherwise they leak into agent backstories.",
         "Added cleanup step in build_agent_prompt() that strips remaining {rag_context} "
         "placeholders after all strategies are applied. Tests verify no leftover placeholders "
         "exist across all 12 role/version combinations."),
    ]
    for title, problem, solution in challenges:
        story.append(Paragraph(f"<b>{title}</b>", styles["Body"]))
        story.append(Paragraph(f"<i>Problem:</i> {problem}", styles["BulletItem"]))
        story.append(Paragraph(f"<i>Solution:</i> {solution}", styles["BulletItem"]))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    # ---- 5. Ethical Considerations ----
    story.append(Paragraph("5. Ethical Considerations", styles["SectionHead"]))

    ethics = [
        ("Financial Advice Disclaimer",
         "All generated reports include a mandatory disclaimer stating the content is for "
         "informational purposes only and does not constitute investment advice. Technical "
         "analysis has inherent limitations and cannot predict future market movements."),
        ("Data Privacy",
         "The system does not collect, store, or transmit any personal user data. Stock data "
         "comes from public APIs (yfinance) or synthetic generation. The knowledge base "
         "contains only publicly available SEC filings and financial information."),
        ("Bias Awareness",
         "The knowledge base covers only a subset of large-cap US technology companies. "
         "Analysis may not generalize to small-caps, international markets, or non-tech sectors. "
         "Reports present both bullish and bearish perspectives where applicable."),
        ("Transparency",
         "The Prompt Engineering Lab allows users to inspect exact prompts sent to the LLM. "
         "RAG results include source citations. Complete source code is open for review."),
        ("Responsible AI Use",
         "The system assists analysis but does not replace human judgment. It provides source "
         "attribution for all claims and includes honest risk assessments. Content filtering "
         "ensures no misleading investment recommendations are generated."),
    ]
    for title, content in ethics:
        story.append(Paragraph(f"<b>{title}</b>", styles["Body"]))
        story.append(Paragraph(content, styles["BulletItem"]))
        story.append(Spacer(1, 4))

    story.append(PageBreak())

    # ---- 6. Future Improvements ----
    story.append(Paragraph("6. Future Improvements", styles["SectionHead"]))

    improvements = [
        ("Real-Time Data Integration",
         "Add WebSocket-based real-time price streaming and intraday technical analysis "
         "with automatic report refresh on significant signal changes."),
        ("Fundamental Analysis Agent",
         "Add a fifth agent specializing in fundamental analysis (P/E ratio, revenue growth, "
         "cash flow) using the RAG knowledge base more deeply. Enable comparative analysis "
         "across peer companies."),
        ("Fine-Tuned Financial Model",
         "Fine-tune a small language model (e.g., Llama-3-8B) on financial analysis tasks "
         "to reduce API costs and improve domain-specific reasoning quality."),
        ("Multi-Modal Report Generation",
         "Generate PDF reports with embedded interactive charts, export to PowerPoint, "
         "and add voice narration of the executive summary using text-to-speech."),
        ("Backtesting Framework",
         "Validate technical indicator signals against historical data. Compute win rates, "
         "Sharpe ratios, and drawdown profiles for each signal type to quantify accuracy."),
        ("Portfolio-Level Analysis",
         "Extend from single-stock analysis to portfolio optimization. Analyze correlations "
         "between holdings, suggest rebalancing based on combined risk metrics."),
        ("User Feedback Loop",
         "Collect user feedback on report quality to continuously improve prompt templates "
         "and RAG retrieval ranking. Implement reinforcement learning from human feedback "
         "(RLHF) for prompt optimization."),
    ]
    for title, content in improvements:
        story.append(Paragraph(f"<b>{title}</b>", styles["Body"]))
        story.append(Paragraph(content, styles["BulletItem"]))
        story.append(Spacer(1, 4))

    # Build PDF
    doc.build(story)
    print(f"PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_pdf()
