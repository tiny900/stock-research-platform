"""
Tests for the Prompt Engineering framework.
Tests templates, strategies, few-shot examples, and composition.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPromptTemplates:
    """Tests for the prompt template registry."""

    def test_all_roles_have_templates(self):
        from prompts.templates import PROMPT_TEMPLATES
        expected_roles = ["data_collector", "analyst", "report_writer", "controller"]
        for role in expected_roles:
            assert role in PROMPT_TEMPLATES, f"Missing templates for role: {role}"

    def test_all_versions_exist(self):
        from prompts.templates import PROMPT_TEMPLATES, PROMPT_VERSIONS
        for role, templates in PROMPT_TEMPLATES.items():
            for version in PROMPT_VERSIONS:
                assert version in templates, f"Missing {version} for role: {role}"

    def test_templates_have_required_fields(self):
        from prompts.templates import PROMPT_TEMPLATES
        for role, versions in PROMPT_TEMPLATES.items():
            for ver_name, template in versions.items():
                assert "role" in template, f"{role}/{ver_name} missing 'role'"
                assert "goal" in template, f"{role}/{ver_name} missing 'goal'"
                assert "backstory" in template, f"{role}/{ver_name} missing 'backstory'"

    def test_get_template_valid(self):
        from prompts.templates import get_template
        t = get_template("analyst", "v2_structured")
        assert "role" in t
        assert "goal" in t

    def test_get_template_invalid_role(self):
        from prompts.templates import get_template
        with pytest.raises(ValueError, match="Unknown agent role"):
            get_template("invalid_role", "v1_basic")

    def test_get_template_invalid_version(self):
        from prompts.templates import get_template
        with pytest.raises(ValueError, match="Unknown version"):
            get_template("analyst", "v999_nonexistent")

    def test_v3_templates_have_rag_placeholder(self):
        from prompts.templates import PROMPT_TEMPLATES
        for role in ["data_collector", "analyst", "report_writer"]:
            backstory = PROMPT_TEMPLATES[role]["v3_cot_rag"]["backstory"]
            assert "{rag_context}" in backstory, \
                f"v3_cot_rag backstory for {role} must contain {{rag_context}} placeholder"

    def test_v2_templates_have_symbol_placeholder(self):
        from prompts.templates import PROMPT_TEMPLATES
        for role in ["data_collector", "analyst", "report_writer"]:
            goal = PROMPT_TEMPLATES[role]["v2_structured"]["goal"]
            assert "{symbol}" in goal, \
                f"v2_structured goal for {role} should contain {{symbol}} placeholder"


class TestFewShotExamples:
    """Tests for few-shot examples."""

    def test_analyst_has_examples(self):
        from prompts.few_shot_examples import get_examples
        examples = get_examples("analyst")
        assert len(examples) >= 2, "Analyst should have at least 2 examples"

    def test_report_writer_has_examples(self):
        from prompts.few_shot_examples import get_examples
        examples = get_examples("report_writer")
        assert len(examples) >= 1

    def test_examples_have_required_fields(self):
        from prompts.few_shot_examples import FEW_SHOT_EXAMPLES
        for role, examples in FEW_SHOT_EXAMPLES.items():
            for i, ex in enumerate(examples):
                assert "input" in ex, f"{role} example {i} missing 'input'"
                assert "output" in ex, f"{role} example {i} missing 'output'"

    def test_format_examples_non_empty(self):
        from prompts.few_shot_examples import format_examples_for_prompt
        text = format_examples_for_prompt("analyst", max_examples=2)
        assert "Example 1" in text
        assert len(text) > 100

    def test_format_examples_empty_role(self):
        from prompts.few_shot_examples import format_examples_for_prompt
        text = format_examples_for_prompt("controller")
        assert text == ""  # Controller has no examples


class TestPromptStrategies:
    """Tests for prompt strategy composition."""

    def test_chain_of_thought_adds_scaffold(self):
        from prompts.strategies import apply_chain_of_thought
        result = apply_chain_of_thought("Base prompt")
        assert "step-by-step" in result.lower()
        assert "Base prompt" in result

    def test_few_shot_injects_examples(self):
        from prompts.strategies import apply_few_shot
        result = apply_few_shot("Base prompt", "analyst")
        assert "Example" in result
        assert "Base prompt" in result

    def test_rag_context_replaces_placeholder(self):
        from prompts.strategies import apply_rag_context
        rag_results = [{"source": "test.md", "content": "Test content", "relevance_score": 0.8}]
        text = "Before {rag_context} After"
        result = apply_rag_context(text, rag_results)
        assert "{rag_context}" not in result
        assert "Test content" in result
        assert "test.md" in result

    def test_rag_context_empty_results(self):
        from prompts.strategies import apply_rag_context
        result = apply_rag_context("Text {rag_context} end", [])
        assert "{rag_context}" not in result
        assert "Text" in result

    def test_symbol_context_replacement(self):
        from prompts.strategies import apply_symbol_context
        result = apply_symbol_context("Analyze {symbol} stock", "AAPL")
        assert result == "Analyze AAPL stock"

    def test_build_agent_prompt_v1(self):
        from prompts.strategies import build_agent_prompt
        result = build_agent_prompt("analyst", "v1_basic", symbol="AAPL")
        assert "role" in result
        assert "goal" in result
        assert "backstory" in result

    def test_build_agent_prompt_v3_with_rag(self):
        from prompts.strategies import build_agent_prompt
        rag = [{"source": "s.md", "content": "Revenue grew 10%", "relevance_score": 0.9}]
        result = build_agent_prompt("analyst", "v3_cot_rag", symbol="TSLA", rag_results=rag)
        assert "TSLA" in result["goal"]
        assert "Revenue grew 10%" in result["backstory"]
        assert "step-by-step" in result["backstory"].lower()

    def test_build_agent_prompt_no_leftover_placeholders(self):
        from prompts.strategies import build_agent_prompt
        for role in ["data_collector", "analyst", "report_writer", "controller"]:
            for ver in ["v1_basic", "v2_structured", "v3_cot_rag"]:
                result = build_agent_prompt(role, ver, symbol="TEST")
                assert "{rag_context}" not in result["backstory"], \
                    f"Leftover {{rag_context}} in {role}/{ver}"
                assert "{symbol}" not in result["goal"], \
                    f"Leftover {{symbol}} in {role}/{ver}"

    def test_version_backstory_length_increases(self):
        from prompts.strategies import build_agent_prompt
        v1 = build_agent_prompt("analyst", "v1_basic")
        v2 = build_agent_prompt("analyst", "v2_structured")
        v3 = build_agent_prompt("analyst", "v3_cot_rag")
        assert len(v1["backstory"]) < len(v2["backstory"]) < len(v3["backstory"]), \
            "More advanced versions should have longer backstories"

    def test_get_strategy_description(self):
        from prompts.strategies import get_strategy_description
        assert "Basic" in get_strategy_description("v1_basic")
        assert "RAG" in get_strategy_description("v3_cot_rag")
