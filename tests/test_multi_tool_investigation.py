"""Deterministic hackathon multi-tool planning and execution contracts."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from agent.analytics_executor import AnalyticsToolExecutor
from agent.meta_planner import MetaModelError, MetaModelPlanner, MetaPlanningResult, build_meta_tools
from agent.pipeline import PipelineError, VerifiedAnalyticsPipeline
from agent.tool_call_trace import ToolCallTrace, TraceStatus, VerificationStatus
from agent.tool_catalog_selector import ToolCatalogSelector
from backend.data_preparation import prepare_dataset
from tests.test_meta_planner import _FakeClient, _response, _tool_call

QUESTION = "Why did our sales decline, and did the September campaign actually help?"
PROFILE = {
    "row_count": 104,
    "column_count": 4,
    "columns": [
        {"name": "date", "dtype": "datetime64[ns]", "role": "time"},
        {"name": "sales", "dtype": "float64", "role": "measure"},
        {"name": "region", "dtype": "object", "role": "dimension"},
        {"name": "campaign_spend", "dtype": "float64", "role": "measure"},
    ],
}
TOOLS = (
    "linear_trend", "mean_shift_change_point", "one_way_anova",
    "bayesian_structural_causal_impact",
)
BINDINGS = (
    {"time_column": "date", "value_column": "sales", "aggregation": "sum"},
    {"time_column": "date", "value_column": "sales", "aggregation": "sum"},
    {"grouping_column": "region", "value_column": "sales"},
    {"time_column": "date", "value_column": "sales", "aggregation": "sum", "intervention_date": "2026-09-01"},
)


def demo_dataset():
    rows = []
    for index, date in enumerate(pd.date_range("2026-08-01", periods=52, freq="D")):
        for region, offset in (("North", 0), ("South", 19)):
            rows.append({
                "date": date.isoformat()[:10],
                "region": region,
                "sales": float(350 + offset - index + (index >= 31) * 3),
                "campaign_spend": float(index >= 31) * 14.0,
            })
    return prepare_dataset(pd.DataFrame(rows)).frame


def plan_traces():
    return tuple(ToolCallTrace.planned(
        call_id=f"demo-{i+1}", tool=tool,
        objective=f"Business question {tool}", reason="Verified investigation",
        parameters={}, bindings=dict(bindings), source_call_ids=(),
    ) for i, (tool, bindings) in enumerate(zip(TOOLS, BINDINGS)))


@pytest.mark.parametrize("question", [
    QUESTION,
    "Sales have been falling since August. What changed and did our promotion help?",
    "Which regions drove the revenue decline, and was the marketing intervention effective?",
    "When did revenue begin to fall and did the September treatment reverse it?",
    "Explain the sales drop across regions and estimate the campaign effect.",
])
def test_actual_catalog_covers_detected_intentions(question):
    result = ToolCatalogSelector().select(
        question, dataset_context=PROFILE,
        tool_catalog=AnalyticsToolExecutor().available_tools(),
    )
    assert len(result.tools) <= 10
    assert {"linear_trend", "bayesian_structural_causal_impact"} <= set(result.ranked_names)
    if any(phrase in question.lower() for phrase in ("why", "which regions", "what changed", "when", "explain")):
        assert "mean_shift_change_point" in result.ranked_names
    if any(phrase in question.lower() for phrase in ("why", "regions", "across")):
        assert "one_way_anova" in result.ranked_names


def test_question_demo_exact_four_tools_and_alternative():
    selected = ToolCatalogSelector().select(
        QUESTION, dataset_context=PROFILE,
        tool_catalog=AnalyticsToolExecutor().available_tools(),
    )
    assert selected.ranked_names[:4] == TOOLS
    assert "kruskal_wallis_test" in selected.ranked_names
    assert selected.ranked_names == ToolCatalogSelector().select(
        QUESTION, dataset_context=PROFILE,
        tool_catalog=AnalyticsToolExecutor().available_tools(),
    ).ranked_names


def test_missing_roles_do_not_reserve_incompatible_families():
    result = ToolCatalogSelector(max_tools=2).select(
        QUESTION, dataset_context={"columns": [{"name": "sales", "role": "measure"}]},
        tool_catalog=AnalyticsToolExecutor().available_tools(),
    )
    assert result.ranked_names[:2] != ("linear_trend", "mean_shift_change_point")


def test_planner_mock_four_calls_safe_context_and_correct_bindings():
    arguments = [
        {"objective": f"Analyze {tool}", "reason": "Needed to answer the question",
         "parameters": {}, "bindings": dict(binding), "source_call_ids": []}
        for tool, binding in zip(TOOLS, BINDINGS)
    ]
    client = _FakeClient(_response(tool_calls=[
        _tool_call(f"call-{i}", name, arguments[i]) for i, name in enumerate(TOOLS)
    ]))
    planner = MetaModelPlanner(client=client)
    catalog = [t for t in AnalyticsToolExecutor().available_tools() if t["name"] in TOOLS]
    result = planner.plan(QUESTION, dataset_context=PROFILE, tool_catalog=catalog)
    assert len(result.traces) == 4
    assert tuple(t.tool for t in result.traces) == TOOLS
    assert len({t.call_id for t in result.traces}) == 4
    assert [t.bindings for t in result.traces] == list(BINDINGS)
    assert all(t.source_call_ids == () for t in result.traces)
    request = client.chat.completions.calls[0]
    assert 'diagnosis' not in request['messages'][0]['content'].lower() or 'causal' in request['messages'][0]['content'].lower()
    assert '104' in request['messages'][1]['content']
    assert '350' not in request['messages'][1]['content']
    assert 'raw_data' not in request['messages'][1]['content']
    assert len(request['tools']) == 4
    causal_description = next(t['function']['description'] for t in request['tools'] if t['function']['name'] == 'bayesian_structural_causal_impact')
    assert 'value_column' in causal_description
    assert 'intervention_date' in causal_description


def test_planner_rejects_five_calls_without_silent_truncation():
    calls = [_tool_call(f"call-{i}", "linear_trend") for i in range(5)]
    with pytest.raises(MetaModelError, match="maximum is 4"):
        MetaModelPlanner(client=_FakeClient(_response(tool_calls=calls))).plan(
            QUESTION, dataset_context=PROFILE,
            tool_catalog=[t for t in AnalyticsToolExecutor().available_tools() if t["name"] == "linear_trend"],
        )


def test_pipeline_rejects_five_calls_from_custom_planner():
    class BadPlanner:
        def plan(self, question, *, dataset_context, tool_catalog):
            return SimpleNamespace(traces=plan_traces() + (plan_traces()[0],))
    with pytest.raises(PipelineError, match="four initial"):
        VerifiedAnalyticsPipeline(BadPlanner()).run(
            QUESTION, dataset=demo_dataset(), dataset_context=PROFILE,
        )


def test_real_four_tool_pipeline_full_execution():
    class StubPlanner:
        def __init__(self):
            self.context = None
            self.catalog = None
        def plan(self, question, *, dataset_context, tool_catalog):
            self.context = dataset_context
            self.catalog = tool_catalog
            return MetaPlanningResult(traces=plan_traces(), assistant_content="", finish_reason="tool_calls", model="stub", usage={})

    planner = StubPlanner()
    result = VerifiedAnalyticsPipeline(planner).run(QUESTION, dataset=demo_dataset(), dataset_context=PROFILE)
    assert [step.trace.tool for step in result.steps] == list(TOOLS)
    assert len(result.steps) == 4
    assert all(step.trace.status is TraceStatus.COMPLETED for step in result.steps)
    assert all(step.trace.deterministic_result is not None for step in result.steps)
    assert all(step.report is not None for step in result.steps)
    assert all(step.trace.trace_hash == step.report.trace_hash for step in result.steps)
    assert result.succeeded
    assert len(planner.catalog) <= 10
    assert 'sales' in json.dumps(planner.context)
    assert '350.0' not in json.dumps(planner.context)


def test_four_calls_allow_fifth_kruskal_replan():
    from verification.verifier import DeterministicVerifier

    class RequestedRobustnessVerifier:
        def verify(self, trace):
            verified, report = DeterministicVerifier().verify(trace)
            if trace.tool == "one_way_anova":
                return trace.verify(VerificationStatus.NEEDS_MORE_ANALYSIS, next_decision="run_robustness_check"), report
            return verified, report
    class StubPlanner:
        def plan(self, question, *, dataset_context, tool_catalog):
            return SimpleNamespace(traces=plan_traces())

    result = VerifiedAnalyticsPipeline(StubPlanner(), verifier=RequestedRobustnessVerifier()).run(
        QUESTION, dataset=demo_dataset(), dataset_context=PROFILE,
    )
    assert len(result.steps) == 5
    assert [step.trace.tool for step in result.steps] == [
        "linear_trend", "mean_shift_change_point", "one_way_anova",
        "kruskal_wallis_test", "bayesian_structural_causal_impact",
    ]
    assert result.steps[2].replanning.status == "scheduled"
    assert result.steps[3].trace.source_call_ids == ("demo-3",)
    assert result.succeeded
    assert not result.exhausted
