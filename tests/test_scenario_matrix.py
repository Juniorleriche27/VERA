"""Matrix of backend & pipeline scenarios for hackathon assessment (Section M)."""
from __future__ import annotations

from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient

from agent.analytics_executor import AnalyticsToolExecutor
from agent.data_binding import DatasetBinder
from agent.pipeline import PipelineError, VerifiedAnalyticsPipeline
from agent.replanning import SelfCorrectionPlanner
from agent.tool_call_trace import ToolCallTrace, TraceStatus, VerificationStatus
from backend.app import create_app
from backend.data_preparation import prepare_dataset
import pandas as pd


class MockPlanner:
    def __init__(self, traces):
        self.traces = traces

    def plan(self, question, *, dataset_context, tool_catalog):
        return SimpleNamespace(traces=self.traces)


class ControlledVerifier:
    def __init__(self, status_map=None):
        self.status_map = status_map or {}

    def verify(self, trace):
        from verification.verifier import DeterministicVerifier
        real = DeterministicVerifier()
        verified, report = real.verify(trace)
        if trace.call_id in self.status_map:
            st, dec = self.status_map[trace.call_id]
            return trace.verify(st, next_decision=dec), report
        return verified, report


@pytest.fixture
def sample_sales_df():
    return pd.DataFrame({
        "date": [
            "2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05",
            "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04", "2026-09-05"
        ],
        "sales": [100.0, 110.0, 105.0, 108.0, 102.0, 150.0, 160.0, 155.0, 165.0, 170.0],
        "region": ["North", "North", "South", "South", "West", "West", "North", "South", "West", "North"],
        "control_sales": [95.0, 100.0, 98.0, 99.0, 97.0, 102.0, 104.0, 101.0, 105.0, 106.0],
    })


def test_scenario_2_robustness_replan(sample_sales_df):
    """Scenario 2: ANOVA -> NEEDS_MORE_ANALYSIS -> Kruskal-Wallis -> VERIFIED."""
    anova_trace = ToolCallTrace.planned(
        call_id="call-anova-1",
        tool="one_way_anova",
        objective="Compare regional sales",
        reason="Check if mean sales differ across regions",
        parameters={},
        bindings={"grouping_column": "region", "value_column": "sales"},
    )
    pipeline = VerifiedAnalyticsPipeline(
        planner=MockPlanner([anova_trace]),
        verifier=ControlledVerifier({"call-anova-1": (VerificationStatus.NEEDS_MORE_ANALYSIS, "run_robustness_check")}),
    )
    result = pipeline.run("Do sales differ across regions?", dataset=sample_sales_df, dataset_context={"row_count": 8})
    assert len(result.steps) == 2
    assert result.steps[0].trace.tool == "one_way_anova"
    assert result.steps[0].trace.verification_status is VerificationStatus.NEEDS_MORE_ANALYSIS
    assert result.steps[0].replanning.status == "scheduled"
    assert result.steps[1].trace.tool == "kruskal_wallis_test"
    assert result.steps[1].trace.verification_status is VerificationStatus.VERIFIED
    assert result.succeeded is True
    brief = result.build_decision_brief()
    assert brief.headline != ""


def test_scenario_3_unresolved_causal_evidence(sample_sales_df):
    """Scenario 3: causal impact -> NEEDS_MORE_ANALYSIS, no fallback available -> no decision brief."""
    causal_trace = ToolCallTrace.planned(
        call_id="call-causal-1",
        tool="bayesian_structural_causal_impact",
        objective="Assess campaign lift",
        reason="Evaluate September campaign",
        parameters={},
        bindings={
            "time_column": "date",
            "value_column": "sales",
            "aggregation": "sum",
            "intervention_date": "2026-09-01",
            "control_columns": ["control_sales"],
        },
    )
    pipeline = VerifiedAnalyticsPipeline(
        planner=MockPlanner([causal_trace]),
        verifier=ControlledVerifier({"call-causal-1": (VerificationStatus.NEEDS_MORE_ANALYSIS, "run_robustness_check")}),
    )
    result = pipeline.run("Did the campaign help?", dataset=sample_sales_df, dataset_context={"row_count": 8})
    assert result.succeeded is False
    assert result.unresolved_call_ids == ("call-causal-1",)
    assert result.steps[0].replanning.status == "no_fallback"


def test_scenario_4_tool_execution_failure(sample_sales_df):
    """Scenario 4: tool execution failure marks trace FAILED with execution_error."""
    bad_trace = ToolCallTrace.planned(
        call_id="call-bad-col",
        tool="one_way_anova",
        objective="Analyze non-existent column",
        reason="Test error handling",
        parameters={},
        bindings={"grouping_column": "non_existent_column", "value_column": "sales"},
    )
    pipeline = VerifiedAnalyticsPipeline(
        planner=MockPlanner([bad_trace]),
    )
    result = pipeline.run("Analyze column", dataset=sample_sales_df, dataset_context={"row_count": 8})
    assert result.succeeded is False
    assert len(result.steps) == 1
    assert result.steps[0].trace.status is TraceStatus.FAILED
    assert "Could not deterministically resolve grouped analysis columns" in result.steps[0].trace.execution_error


def test_scenario_7_invalid_csv_upload():
    """Scenario 7: invalid CSV rejects with clear 400 error."""
    client = TestClient(create_app(planner_factory=lambda: MockPlanner([])))
    response = client.post(
        "/datasets",
        files={"file": ("corrupt.csv", b"This is not a,valid\n\x00\xffbinary", "text/csv")},
    )
    # Binary null bytes or unparseable CSV is rejected with HTTP 400
    assert response.status_code in (400, 422)


def test_scenario_8_dataset_incompatible_temporal(sample_sales_df):
    """Scenario 8: calling temporal tool on dataset with missing time column fails gracefully."""
    non_temporal_df = sample_sales_df[["region", "sales"]]
    trend_trace = ToolCallTrace.planned(
        call_id="call-trend-fail",
        tool="linear_trend",
        objective="Analyze trend",
        reason="Attempt temporal analysis without date",
        parameters={},
        bindings={"time_column": "date", "value_column": "sales"},
    )
    pipeline = VerifiedAnalyticsPipeline(
        planner=MockPlanner([trend_trace]),
    )
    result = pipeline.run("What is trend?", dataset=non_temporal_df, dataset_context={"row_count": 8})
    assert result.succeeded is False
    assert result.steps[0].trace.status is TraceStatus.FAILED
    assert "date" in result.steps[0].trace.execution_error


def test_scenario_9_one_tool_investigation(sample_sales_df):
    """Scenario 9: Single tool question produces valid verified pipeline without forcing 4 tools."""
    anova_trace = ToolCallTrace.planned(
        call_id="call-anova-only",
        tool="one_way_anova",
        objective="Compare regional sales",
        reason="Targeted group comparison",
        parameters={"alpha": 0.05},
        bindings={"grouping_column": "region", "value_column": "sales"},
    )
    pipeline = VerifiedAnalyticsPipeline(
        planner=MockPlanner([anova_trace]),
    )
    result = pipeline.run("Are sales different across regions?", dataset=sample_sales_df, dataset_context={"row_count": 8})
    assert result.succeeded is True
    assert len(result.steps) == 1
    brief = result.build_decision_brief()
    assert brief.headline != ""
    assert len(brief.evidence.items) == 1
    visuals = result.build_visualizations()
    assert len(visuals) == 1
