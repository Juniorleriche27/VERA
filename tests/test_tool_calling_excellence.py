"""Comprehensive tests for VERA tool-calling excellence, schemas, and argument validation."""
from __future__ import annotations

import json
from types import SimpleNamespace
import pandas as pd
import pytest

from agent.analytics_executor import AnalyticsToolExecutor
from agent.data_binding import DatasetBinder
from agent.meta_planner import (
    MetaModelError,
    MetaModelPlanner,
    _validate_semantic_tool_contract,
    build_meta_tools,
)
from agent.tool_catalog_selector import ToolCatalogSelector


def _catalog():
    return AnalyticsToolExecutor().available_tools()


class _FakeCompletions:
    def __init__(self, response):
        self.response = response

    def create(self, **kwargs):
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_FakeCompletions(response))


def _mock_model_response(tool_name: str, parameters: dict, bindings: dict, call_id: str = "call-1"):
    arguments = {
        "objective": f"Run {tool_name}",
        "reason": f"Testing contract for {tool_name}",
        "parameters": parameters,
        "bindings": bindings,
        "source_call_ids": [],
    }
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=tool_name,
            arguments=json.dumps(arguments),
        ),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="muse-spark-1.3",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


@pytest.fixture
def sample_eval_df():
    return pd.DataFrame({
        "date": [
            "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05",
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05"
        ],
        "sales": [100.0, 110.0, 105.0, 108.0, 102.0, 150.0, 160.0, 155.0, 165.0, 170.0],
        "region": ["North", "North", "South", "South", "West", "West", "North", "South", "West", "North"],
        "control_sales": [95.0, 100.0, 98.0, 99.0, 97.0, 102.0, 104.0, 101.0, 105.0, 106.0],
    })


def test_strategic_tools_expose_rich_schemas():
    tools = build_meta_tools(_catalog())
    by_name = {t["function"]["name"]: t["function"] for t in tools}

    # 1. linear_trend: bindings have time_column, value_column (required), aggregation; parameters is empty {}
    assert "linear_trend" in by_name
    trend_spec = by_name["linear_trend"]["parameters"]["properties"]
    assert trend_spec["parameters"]["properties"] == {}
    assert trend_spec["parameters"]["additionalProperties"] is False
    assert trend_spec["bindings"]["required"] == ["time_column", "value_column"]
    assert trend_spec["bindings"]["additionalProperties"] is False

    # 2. mean_shift_change_point: parameters have min_segment_size, threshold
    change_params = by_name["mean_shift_change_point"]["parameters"]["properties"]["parameters"]["properties"]
    assert "min_segment_size" in change_params
    assert "threshold" in change_params

    # 3. variance_shift_change_point: parameters have min_segment_size, variance_ratio_threshold
    var_params = by_name["variance_shift_change_point"]["parameters"]["properties"]["parameters"]["properties"]
    assert "min_segment_size" in var_params
    assert "variance_ratio_threshold" in var_params

    # 4. one_way_anova: parameters empty {}, required grouping_column and value_column
    anova_spec = by_name["one_way_anova"]["parameters"]["properties"]
    assert anova_spec["parameters"]["properties"] == {}
    assert anova_spec["bindings"]["required"] == ["grouping_column", "value_column"]

    # 5. kruskal_wallis_test: parameters empty {}, required grouping_column and value_column
    kruskal_spec = by_name["kruskal_wallis_test"]["parameters"]["properties"]
    assert kruskal_spec["parameters"]["properties"] == {}
    assert kruskal_spec["bindings"]["required"] == ["grouping_column", "value_column"]

    # 6. mann_whitney_u_test: alternative in parameters, required grouping_column and value_column
    mw_spec = by_name["mann_whitney_u_test"]["parameters"]["properties"]
    assert "alternative" in mw_spec["parameters"]["properties"]
    assert mw_spec["bindings"]["required"] == ["grouping_column", "value_column"]

    # 7. bayesian_structural_causal_impact: intervention_date in bindings (required), credibility & prior_scale in parameters
    causal_spec = by_name["bayesian_structural_causal_impact"]["parameters"]["properties"]
    assert "intervention_date" in causal_spec["bindings"]["required"]
    assert "credibility" in causal_spec["parameters"]["properties"]
    assert "prior_scale" in causal_spec["parameters"]["properties"]


def test_forbidden_raw_data_in_parameters_is_rejected():
    for forbidden_key in ["data", "rows", "records", "values", "df", "raw_data"]:
        with pytest.raises(MetaModelError, match="contains forbidden raw data key"):
            _validate_semantic_tool_contract(
                "linear_trend",
                parameters={forbidden_key: [1, 2, 3]},
                bindings={"time_column": "date", "value_column": "sales"},
                call_id="call-fail-raw-param",
            )


def test_forbidden_raw_data_in_bindings_is_rejected():
    for forbidden_key in ["data", "rows", "records", "values", "df", "raw_data"]:
        with pytest.raises(MetaModelError, match="contains forbidden raw data key"):
            _validate_semantic_tool_contract(
                "one_way_anova",
                parameters={},
                bindings={"grouping_column": "region", "value_column": "sales", forbidden_key: ["a", "b"]},
                call_id="call-fail-raw-binding",
            )


def test_aggregation_in_parameters_is_rejected():
    with pytest.raises(MetaModelError, match="placed 'aggregation' in 'parameters'"):
        _validate_semantic_tool_contract(
            "linear_trend",
            parameters={"aggregation": "sum"},
            bindings={"time_column": "date", "value_column": "sales"},
            call_id="call-fail-agg-param",
        )


def test_invalid_aggregation_value_is_rejected():
    with pytest.raises(MetaModelError, match="invalid aggregation 'median'"):
        _validate_semantic_tool_contract(
            "linear_trend",
            parameters={},
            bindings={"time_column": "date", "value_column": "sales", "aggregation": "median"},
            call_id="call-fail-agg-val",
        )


def test_causal_intervention_date_in_parameters_is_rejected():
    with pytest.raises(MetaModelError, match="placed 'intervention_date' in 'parameters'"):
        _validate_semantic_tool_contract(
            "bayesian_structural_causal_impact",
            parameters={"intervention_date": "2026-09-01"},
            bindings={"time_column": "date", "value_column": "sales"},
            call_id="call-fail-causal-date-param",
        )


def test_causal_missing_intervention_date_is_rejected():
    with pytest.raises(MetaModelError, match="requires an 'intervention_date' binding"):
        _validate_semantic_tool_contract(
            "bayesian_structural_causal_impact",
            parameters={},
            bindings={"time_column": "date", "value_column": "sales"},
            call_id="call-fail-causal-no-date",
        )


def test_missing_required_bindings_are_rejected():
    # missing time_column for linear trend
    with pytest.raises(MetaModelError, match="missing required binding 'time_column'"):
        _validate_semantic_tool_contract(
            "linear_trend",
            parameters={},
            bindings={"value_column": "sales"},
            call_id="call-fail-no-time",
        )
    # missing value_column for linear trend
    with pytest.raises(MetaModelError, match="missing required binding 'value_column'"):
        _validate_semantic_tool_contract(
            "linear_trend",
            parameters={},
            bindings={"time_column": "date"},
            call_id="call-fail-no-val",
        )
    # missing grouping_column for ANOVA
    with pytest.raises(MetaModelError, match="missing required binding 'grouping_column'"):
        _validate_semantic_tool_contract(
            "one_way_anova",
            parameters={},
            bindings={"value_column": "sales"},
            call_id="call-fail-no-group",
        )


def test_unsupported_parameters_are_rejected():
    # linear_trend has no parameters
    with pytest.raises(MetaModelError, match="unsupported parameter keys"):
        _validate_semantic_tool_contract(
            "linear_trend",
            parameters={"min_periods": 5},
            bindings={"time_column": "date", "value_column": "sales"},
            call_id="call-fail-unsupported-param",
        )
    # ANOVA has no alpha parameter
    with pytest.raises(MetaModelError, match="unsupported parameter keys"):
        _validate_semantic_tool_contract(
            "one_way_anova",
            parameters={"alpha": 0.05},
            bindings={"grouping_column": "region", "value_column": "sales"},
            call_id="call-fail-anova-alpha",
        )


def test_unsupported_bindings_are_rejected():
    with pytest.raises(MetaModelError, match="unsupported binding keys"):
        _validate_semantic_tool_contract(
            "linear_trend",
            parameters={},
            bindings={"time_column": "date", "value_column": "sales", "date_range": "2026"},
            call_id="call-fail-unsupported-binding",
        )


def test_parameter_bounds_validation():
    # mean_shift: threshold <= 0
    with pytest.raises(MetaModelError, match="parameter 'threshold' must be a finite number > 0"):
        _validate_semantic_tool_contract(
            "mean_shift_change_point",
            parameters={"threshold": -0.5},
            bindings={"time_column": "date", "value_column": "sales"},
            call_id="call-fail-threshold",
        )
    # variance_shift: variance_ratio_threshold <= 1.0
    with pytest.raises(MetaModelError, match="parameter 'variance_ratio_threshold' must be a finite number > 1"):
        _validate_semantic_tool_contract(
            "variance_shift_change_point",
            parameters={"variance_ratio_threshold": 1.0},
            bindings={"time_column": "date", "value_column": "sales"},
            call_id="call-fail-var-threshold",
        )
    # mann_whitney: invalid alternative
    with pytest.raises(MetaModelError, match="parameter 'alternative' must be 'two-sided', 'greater', or 'less'"):
        _validate_semantic_tool_contract(
            "mann_whitney_u_test",
            parameters={"alternative": "invalid"},
            bindings={"grouping_column": "region", "value_column": "sales"},
            call_id="call-fail-alt",
        )
    # causal: credibility <= 0 or >= 1
    with pytest.raises(MetaModelError, match="parameter 'credibility' must be a number strictly between 0 and 1"):
        _validate_semantic_tool_contract(
            "bayesian_structural_causal_impact",
            parameters={"credibility": 1.5},
            bindings={"time_column": "date", "value_column": "sales", "intervention_date": "2026-09-01"},
            call_id="call-fail-cred",
        )


def test_end_to_end_chain_for_all_seven_strategic_tools(sample_eval_df):
    """Test full pipeline chain: model response -> parser -> semantic validation -> trace -> binder -> executor."""
    catalog = _catalog()
    binder = DatasetBinder()
    executor = AnalyticsToolExecutor()

    test_cases = [
        ("linear_trend", {}, {"time_column": "date", "value_column": "sales", "aggregation": "sum"}),
        ("mean_shift_change_point", {"min_segment_size": 2, "threshold": 1.0}, {"time_column": "date", "value_column": "sales", "aggregation": "sum"}),
        ("variance_shift_change_point", {"min_segment_size": 3, "variance_ratio_threshold": 2.0}, {"time_column": "date", "value_column": "sales", "aggregation": "sum"}),
        ("one_way_anova", {}, {"grouping_column": "region", "value_column": "sales"}),
        ("kruskal_wallis_test", {}, {"grouping_column": "region", "value_column": "sales"}),
        ("mann_whitney_u_test", {"alternative": "two-sided"}, {"grouping_column": "region", "value_column": "sales", "group_a": "North", "group_b": "South"}),
        ("bayesian_structural_causal_impact", {"credibility": 0.95, "prior_scale": 10.0}, {"time_column": "date", "value_column": "sales", "intervention_date": "2026-09-01", "control_columns": ["control_sales"], "aggregation": "sum"}),
    ]

    for tool_name, params, bindings in test_cases:
        resp = _mock_model_response(tool_name, params, bindings)
        planner = MetaModelPlanner(client=_FakeClient(resp))
        planning_result = planner.plan(
            f"Run analysis with {tool_name}",
            dataset_context={"columns": [{"name": c} for c in sample_eval_df.columns]},
            tool_catalog=catalog,
        )
        assert len(planning_result.traces) == 1
        trace = planning_result.traces[0]
        assert trace.tool == tool_name

        # Bind with DatasetBinder
        bound = binder.bind(trace, sample_eval_df)
        assert bound.tool == tool_name

        # Execute with AnalyticsToolExecutor
        exec_result = executor.execute(trace.tool, bound.parameters, bound.bindings)
        assert exec_result is not None
        assert isinstance(exec_result, dict)


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
