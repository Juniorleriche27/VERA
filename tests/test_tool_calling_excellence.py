"""Comprehensive tests for VERA tool-calling excellence, schemas, and argument validation."""
from __future__ import annotations

import pytest

from agent.analytics_executor import AnalyticsToolExecutor
from agent.meta_planner import (
    MetaModelError,
    _validate_semantic_tool_contract,
    build_meta_tools,
)
from agent.tool_catalog_selector import ToolCatalogSelector


def _catalog():
    return AnalyticsToolExecutor().available_tools()


def test_strategic_tools_expose_rich_schemas():
    tools = build_meta_tools(_catalog())
    by_name = {t["function"]["name"]: t["function"] for t in tools}

    # 1. linear_trend
    assert "linear_trend" in by_name
    trend_props = by_name["linear_trend"]["parameters"]["properties"]["bindings"]["properties"]
    assert "time_column" in trend_props
    assert "value_column" in trend_props
    assert "aggregation" in trend_props

    # 2. mean_shift_change_point
    change_params = by_name["mean_shift_change_point"]["parameters"]["properties"]["parameters"]["properties"]
    assert "min_segment_size" in change_params

    # 3. one_way_anova
    anova_props = by_name["one_way_anova"]["parameters"]["properties"]["bindings"]["properties"]
    assert "grouping_column" in anova_props
    assert "value_column" in anova_props

    # 4. bayesian_structural_causal_impact
    causal_props = by_name["bayesian_structural_causal_impact"]["parameters"]["properties"]["bindings"]["properties"]
    assert "intervention_date" in causal_props
    assert "control_columns" in causal_props


def test_forbidden_raw_data_in_parameters_is_rejected():
    with pytest.raises(MetaModelError, match="contains forbidden raw data key"):
        _validate_semantic_tool_contract(
            "linear_trend",
            parameters={"raw_data": [1, 2, 3]},
            bindings={"time_column": "date", "value_column": "sales"},
            call_id="call-fail-1",
        )


def test_forbidden_raw_data_in_bindings_is_rejected():
    with pytest.raises(MetaModelError, match="contains forbidden raw data key"):
        _validate_semantic_tool_contract(
            "one_way_anova",
            parameters={},
            bindings={"rows": ["a", "b"]},
            call_id="call-fail-2",
        )


def test_invalid_aggregation_is_rejected():
    with pytest.raises(MetaModelError, match="invalid aggregation 'average'"):
        _validate_semantic_tool_contract(
            "linear_trend",
            parameters={},
            bindings={"aggregation": "average"},
            call_id="call-fail-3",
        )


def test_causal_impact_missing_intervention_date_is_rejected():
    with pytest.raises(MetaModelError, match="requires an 'intervention_date' binding"):
        _validate_semantic_tool_contract(
            "bayesian_structural_causal_impact",
            parameters={},
            bindings={"time_column": "date", "value_column": "sales"},
            call_id="call-fail-4",
        )


def test_invalid_alpha_bounds_are_rejected():
    with pytest.raises(MetaModelError, match="must be a number strictly between 0 and 1"):
        _validate_semantic_tool_contract(
            "one_way_anova",
            parameters={"alpha": 1.5},
            bindings={"grouping_column": "region", "value_column": "sales"},
            call_id="call-fail-5",
        )


def test_all_official_questions_from_section_x():
    selector = ToolCatalogSelector()
    dataset_context = {
        "columns": [
            {"name": "date", "role": "time"},
            {"name": "sales", "role": "measure"},
            {"name": "region", "role": "dimension"},
            {"name": "control_sales", "role": "measure"},
        ]
    }
    catalog = _catalog()

    # 1. Multi-part official hackathon question -> all 4 families
    q1 = "Why did our sales decline, and did the September campaign actually help?"
    sel1 = selector.select(q1, dataset_context=dataset_context, tool_catalog=catalog)
    top1 = sel1.ranked_names[:4]
    assert "linear_trend" in top1
    assert "mean_shift_change_point" in top1
    assert "one_way_anova" in top1
    assert "bayesian_structural_causal_impact" in top1

    # 2. Timing of decline
    q2 = "When did sales start declining?"
    intents2 = selector._investigation_intents(q2.lower(), {"time", "measure", "dimension"})
    assert "change" in intents2
    assert "temporal" in intents2
    assert "causal" not in intents2

    # 3. Regional difference
    q3 = "Are sales significantly different across regions?"
    intents3 = selector._investigation_intents(q3.lower(), {"time", "measure", "dimension"})
    assert intents3 == {"groups"}

    # 4. Campaign impact only
    q4 = "Did the September campaign improve sales?"
    intents4 = selector._investigation_intents(q4.lower(), {"time", "measure", "dimension"})
    assert "causal" in intents4
    assert "groups" not in intents4

    # 5. Temporal evolution
    q5 = "How are sales evolving over time?"
    intents5 = selector._investigation_intents(q5.lower(), {"time", "measure", "dimension"})
    assert intents5 == {"temporal"}

    # 6. Which region performs differently
    q6 = "Which region performs differently?"
    intents6 = selector._investigation_intents(q6.lower(), {"time", "measure", "dimension"})
    assert intents6 == {"groups"}

    # 7. Two group comparison
    q7 = "Compare North and South sales."
    intents7 = selector._investigation_intents(q7.lower(), {"time", "measure", "dimension"})
    assert "groups" in intents7

    # 8. Structural shift
    q8 = "Was there a structural shift in sales?"
    intents8 = selector._investigation_intents(q8.lower(), {"time", "measure", "dimension"})
    assert intents8 == {"change"}
