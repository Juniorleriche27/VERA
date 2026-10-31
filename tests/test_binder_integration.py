"""Local deterministic trace -> binder -> executor integration tests."""
from __future__ import annotations

import pandas as pd
import pytest

from agent.analytics_executor import AnalyticsToolExecutor
from agent.data_binding import DataBindingError, DatasetBinder
from agent.tool_call_trace import ToolCallTrace
from backend.data_preparation import prepare_dataset


def dataset(n=24):
    rows = []
    for day in range(n):
        date = (pd.Timestamp("2026-08-01") + pd.Timedelta(days=day)).date().isoformat()
        for group, extra in [("North", 0), ("South", 8)]:
            rows.append({
                "date": date, "region": group,
                "sales": float(120 + day * 2 + extra + (8 if day >= 12 else 0)),
                "control": float(80 + day + extra),
            })
    return prepare_dataset(pd.DataFrame(rows)).frame


def bind_run(tool, frame, bindings, parameters=None):
    trace = ToolCallTrace.planned(
        call_id="integration-1", tool=tool, objective="Test analytics",
        reason="Deterministic integration", bindings=bindings,
        parameters=parameters or {},
    )
    original = trace.as_dict()
    bound = DatasetBinder().bind(trace, frame)
    assert trace.as_dict() == original
    assert set(trace.bindings) == set(bindings)
    result = AnalyticsToolExecutor().execute(tool, bound.parameters, bound.bindings)
    return bound, result


@pytest.mark.parametrize("tool,parameters", [
    ("linear_trend", {}),
    ("mean_shift_change_point", {"threshold": 0.5}),
    ("variance_shift_change_point", {}),
    ("moving_average_trend", {"window": 3}),
    ("additive_seasonal_decomposition", {"period": 7}),
])
def test_each_temporal_tool_real_executor(tool, parameters):
    frame = dataset()
    bound, result = bind_run(
        tool, frame,
        {"time_column": "date", "value_column": "sales", "aggregation": "sum"},
        parameters,
    )
    assert set(bound.bindings) == {"time", "values"}
    assert len(bound.bindings["time"]) == 24
    assert bound.bindings["values"][0] == 248.0
    assert bound.parameters == parameters
    assert result["status"] == "success"
    assert result["result"]


def test_duplicate_dates_require_aggregation():
    with pytest.raises(DataBindingError, match="Repeated timestamps"):
        DatasetBinder().bind(ToolCallTrace.planned(
            call_id="duplicate", tool="linear_trend", objective="Trend",
            reason="Duplicate timestamps", parameters={},
            bindings={"time_column": "date", "value_column": "sales"},
        ), dataset())


def test_unique_dates_allow_identity_without_aggregation():
    frame = dataset().query("region == 'North'")
    bound, result = bind_run("linear_trend", frame, {
        "time_column": "date", "value_column": "sales",
    })
    assert bound.bindings["values"][0] == 120.0
    assert result["status"] == "success"


def test_mean_aggregation_and_invalid_aggregation():
    frame = dataset()
    bound, _ = bind_run("linear_trend", frame, {
        "time_column": "date", "value_column": "sales", "aggregation": "mean",
    })
    assert bound.bindings["values"][0] == 124.0
    for value in ("median", "first", 9):
        with pytest.raises(DataBindingError, match="aggregation"):
            DatasetBinder().bind(ToolCallTrace.planned(
                call_id="invalid", tool="linear_trend", objective="test", reason="test",
                parameters={}, bindings={"time_column": "date", "value_column": "sales", "aggregation": value},
            ), frame)


def test_mann_whitney_executed_from_grouped_data():
    bound, result = bind_run("mann_whitney_u_test", dataset(), {
        "grouping_column": "region", "value_column": "sales",
        "group_a": "North", "group_b": "South",
    })
    assert set(bound.bindings) == {"data"}
    assert set(bound.bindings["data"]) == {"North", "South"}
    assert "p_value" in result


def test_causal_impact_with_aligned_control_real_result():
    frame = dataset()
    bound, result = bind_run("bayesian_structural_causal_impact", frame, {
        "time_column": "date", "value_column": "sales", "aggregation": "sum",
        "intervention_date": "2026-08-13", "control_columns": ["control"],
    })
    assert bound.bindings["intervention_index"] == 12
    assert set(bound.bindings["controls"]) == {"control"}
    assert len(bound.bindings["outcome"]) == 24
    assert len(bound.bindings["controls"]["control"]) == 24
    assert "absolute_effect" in result


def test_causal_impact_without_controls():
    bound, result = bind_run("bayesian_structural_causal_impact", dataset(), {
        "time_column": "date", "value_column": "sales", "aggregation": "sum",
        "intervention_date": "2026-08-13",
    })
    assert bound.bindings["controls"] == {}
    assert "counterfactual_post_mean" in result


def test_causal_invalid_intervention_rejected_no_nearest_date():
    frame = dataset()
    trace = ToolCallTrace.planned(
        call_id="invalid", tool="bayesian_structural_causal_impact", objective="test", reason="test",
        parameters={}, bindings={"time_column": "date", "value_column": "sales", "aggregation": "sum", "intervention_date": "2026-09-30"},
    )
    with pytest.raises(DataBindingError, match="exactly match"):
        DatasetBinder().bind(trace, frame)


def test_causal_control_must_be_numeric_and_not_outcome():
    frame = dataset()
    for bad in (["region"], ["sales"], ["control", "control"]):
        trace = ToolCallTrace.planned(
            call_id="invalid", tool="bayesian_structural_causal_impact", objective="test", reason="test",
            parameters={}, bindings={"time_column": "date", "value_column": "sales", "aggregation": "sum", "intervention_date": "2026-08-13", "control_columns": bad},
        )
        with pytest.raises(DataBindingError):
            DatasetBinder().bind(trace, frame)


def test_time_series_does_not_mutate_raw_dataset():
    frame = dataset()
    copy = frame.copy(deep=True)
    bind_run("linear_trend", frame, {"time_column": "date", "value_column": "sales", "aggregation": "sum"})
    pd.testing.assert_frame_equal(frame, copy)
