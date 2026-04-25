"""
Prompt Strategies — Composable Prompt Engineering Functions
============================================================
Each strategy is a function that transforms or augments a base prompt.
Strategies can be composed together via build_agent_prompt().
"""

from typing import Optional
from prompts.templates import get_template
from prompts.few_shot_examples import format_examples_for_prompt


def apply_chain_of_thought(text: str) -> str:
    """Add chain-of-thought reasoning scaffold to a prompt."""
    cot_suffix = (
        "\n\n**Important: Think through your analysis step-by-step.** "
        "Before reaching a conclusion:\n"
        "1. List the key data points you have\n"
        "2. Analyze each piece of evidence independently\n"
        "3. Identify any contradictions or confirmations between signals\n"
        "4. Synthesize into a final assessment with stated confidence level\n"
        "Show your reasoning process."
    )
    return text + cot_suffix


def apply_few_shot(text: str, agent_role: str, max_examples: int = 2) -> str:
    """Inject few-shot examples into the prompt."""
    examples_text = format_examples_for_prompt(agent_role, max_examples)
    if examples_text:
        return text + "\n" + examples_text
    return text


def apply_rag_context(text: str, rag_results: list[dict]) -> str:
    """Inject RAG retrieval results with citation markers into the prompt."""
    if not rag_results:
        return text.replace("{rag_context}", "")

    context_lines = ["**Relevant Financial Context from Knowledge Base:**\n"]
    for r in rag_results:
        source = r.get("source", "unknown")
        content = r.get("content", "")[:600]
        score = r.get("relevance_score", 0)
        context_lines.append(
            f"[Source: {source} | Relevance: {score:.2f}]\n"
            f"> {content}\n"
        )

    context_block = "\n".join(context_lines)
    if "{rag_context}" in text:
        return text.replace("{rag_context}", context_block)
    return text + "\n\n" + context_block


def apply_output_schema(text: str, schema_description: str) -> str:
    """Add output format constraints to the prompt."""
    schema_block = (
        f"\n\n**Required Output Format:**\n"
        f"{schema_description}\n"
        f"Strictly follow this format in your response."
    )
    return text + schema_block


def apply_symbol_context(text: str, symbol: str) -> str:
    """Replace {symbol} placeholder with actual stock symbol."""
    return text.replace("{symbol}", symbol)


def build_agent_prompt(agent_role: str,
                       version: str = "v3_cot_rag",
                       symbol: str = "",
                       rag_results: Optional[list[dict]] = None,
                       strategies: Optional[list[str]] = None) -> dict:
    """
    Build a complete agent prompt by composing template + strategies.

    Args:
        agent_role: One of 'data_collector', 'analyst', 'report_writer', 'controller'
        version: Template version ('v1_basic', 'v2_structured', 'v3_cot_rag')
        symbol: Stock ticker symbol for context injection
        rag_results: Retrieved documents from RAG for context injection
        strategies: List of strategy names to apply. Defaults based on version.

    Returns:
        Dict with 'role', 'goal', 'backstory' strings ready for CrewAI Agent.
    """
    template = get_template(agent_role, version)
    role = template["role"]
    goal = template["goal"]
    backstory = template["backstory"]

    # Default strategies based on version
    if strategies is None:
        if version == "v1_basic":
            strategies = []
        elif version == "v2_structured":
            strategies = ["few_shot"]
        elif version == "v3_cot_rag":
            strategies = ["rag_context", "few_shot", "chain_of_thought"]

    # Apply symbol context first
    if symbol:
        goal = apply_symbol_context(goal, symbol)
        backstory = apply_symbol_context(backstory, symbol)

    # Apply strategies in order
    for strategy in strategies:
        if strategy == "chain_of_thought":
            backstory = apply_chain_of_thought(backstory)
        elif strategy == "few_shot":
            backstory = apply_few_shot(backstory, agent_role)
        elif strategy == "rag_context":
            backstory = apply_rag_context(backstory, rag_results or [])
        elif strategy == "output_schema":
            pass  # Applied at task level, not backstory

    # Clean up any remaining placeholders
    backstory = backstory.replace("{rag_context}", "")

    return {
        "role": role,
        "goal": goal,
        "backstory": backstory,
    }


def get_strategy_description(version: str) -> str:
    """Get a human-readable description of strategies used by a prompt version."""
    descriptions = {
        "v1_basic": "Basic role/goal/backstory prompts with no enhancements",
        "v2_structured": "Structured prompts with explicit workflows and few-shot examples",
        "v3_cot_rag": "Chain-of-thought reasoning + RAG context injection + few-shot examples",
    }
    return descriptions.get(version, "Unknown version")
