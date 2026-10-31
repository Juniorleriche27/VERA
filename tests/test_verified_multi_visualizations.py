"""Verified-only multi-evidence visualizations and API integration."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
import math

import pytest
from fastapi.testclient import TestClient

from agent.decision_brief import EvidenceItem, EvidencePackage
from agent.pipeline import VerifiedAnalyticsPipeline
from backend.app import create_app
from tests.test_multi_tool_investigation import QUESTION, PROFILE, demo_dataset, plan_traces
from tests.test_replanning_pipeline_resilience import run, planned
from agent.tool_call_trace import VerificationStatus
from visualization.spec import VerifiedVisualizationBuilder, VisualizationSpecError


def item(tool, result, call_id="call-x"):
    return EvidenceItem(call_id, tool, "f" * 64, "Verified calculation", result)


def build(*items):
    return VerifiedVisualizationBuilder().build(EvidencePackage(QUESTION, tuple(items)))


def trend(timestamps=None, fitted=None):
    return {"status": "success", "method": "linear_trend", "result": {
        "timestamps": ["2026-08-01T00:00:00", "2026-08-02T00:00:00"] if timestamps is None else timestamps,
        "fitted_values": [20.0, 18.0] if fitted is None else fitted,
        "slope": -2.0,
    }}


def shift(*, kind="mean", detected=True):
    key = "mean" if kind == "mean" else "variance"
    return {"status": "success", "method": f"{kind}_shift_change_point", "result": {
        "detected": detected, "change_timestamp": "2026-08-02T00:00:00",
        "strongest_candidate": {f"left_{key}": 20, f"right_{key}": 10},
    }}


def causal(observed=None, counterfactual=None, intervals=None, credibility=.9):
    return {"observed_post": [120, 130, 125] if observed is None else observed,
            "counterfactual_post_mean": [110, 115, 114] if counterfactual is None else counterfactual,
            "counterfactual_post_interval": [[105, 115], [110, 120], [109, 119]] if intervals is None else intervals,
            "credibility": credibility}


def test_trend_is_fitted_only_from_engine_envelope():
    original = trend()
    copy = deepcopy(original)
    spec = build(item("linear_trend", original))[0]
    assert spec.chart_type == "line"
    assert spec.x_key == "timestamp"
    assert spec.data == (
        {"timestamp": "2026-08-01T00:00:00", "fitted": 20.0},
        {"timestamp": "2026-08-02T00:00:00", "fitted": 18.0},
    )
    assert set(spec.data[0]) == {"timestamp", "fitted"}
    assert spec.series[0].role == "fitted"
    assert original == copy


@pytest.mark.parametrize("times,values", [([], []), (["2026-08-01"], [1, 2]), ([""], [1]), ([2], [1]), (["x"], [float("inf")]), (["x"], [True])])
def test_bad_trend_is_rejected(times, values):
    with pytest.raises(VisualizationSpecError):
        build(item("linear_trend", trend(times, values)))


def test_verified_mean_shift_chart_with_timestamp_annotation():
    s = build(item("mean_shift_change_point", shift()))[0]
    assert s.chart_type == "bar" and s.title == "Verified mean shift"
    assert s.data == ({"segment": "Before shift", "mean": 20.0}, {"segment": "After shift", "mean": 10.0})
    assert s.annotations[0]["value"] == "2026-08-02T00:00:00"
    assert build(item("mean_shift_change_point", shift(detected=False))) == ()


def test_invalid_shift_candidate_rejected():
    with pytest.raises(VisualizationSpecError):
        build(item("mean_shift_change_point", {"detected": True, "strongest_candidate": {}}))


def test_variance_shift_and_not_detected():
    s = build(item("variance_shift_change_point", shift(kind="variance")))[0]
    assert s.title == "Verified variance shift"
    assert s.data[0]["variance"] == 20.0 and s.data[1]["variance"] == 10.0
    assert build(item("variance_shift_change_point", shift(kind="variance", detected=False))) == ()


def test_group_mean_and_kruskal_median_separate():
    anova = item("one_way_anova", {"group_statistics": {"North": {"mean": 20, "median": 18}}}, "a")
    kruskal = item("kruskal_wallis_test", {"group_statistics": {"North": {"mean": 20, "median": 18}}}, "k")
    charts = build(anova, kruskal)
    assert [s.title for s in charts] == ["Verified group means", "Verified group medians"]
    assert charts[0].data[0] == {"group": "North", "mean": 20.0}
    assert charts[1].data[0] == {"group": "North", "median": 18.0}
    assert charts[1].call_id == "k"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "4"])
def test_group_rejects_nonfinite_non_numeric(value):
    with pytest.raises(VisualizationSpecError):
        build(item("kruskal_wallis_test", {"group_statistics": {"North": {"median": value}}}))


def test_mann_whitney_and_unknown_tool_do_not_invent_charts():
    assert build(item("mann_whitney_u_test", {"p_value": 0.02}), item("other_tool", {})) == ()


def test_causal_data_bands_and_frontend_payload():
    original = causal()
    copy = deepcopy(original)
    s = build(item("bayesian_structural_causal_impact", original, "cause-1"))[0]
    assert s.chart_type == "line" and s.x_key == "post_period"
    assert s.data[0] == {"post_period": 1, "observed": 120.0, "counterfactual": 110.0, "counterfactual_lower": 105.0, "counterfactual_upper": 115.0}
    assert s.data[2]["post_period"] == 3
    assert "90%" in s.description
    assert len(s.bands) == 1
    payload = s.as_dict()
    assert payload["bands"] == [{"lowerKey": "counterfactual_lower", "upperKey": "counterfactual_upper", "label": "Counterfactual credible interval"}]
    assert payload["chartType"] == "line"
    assert {x["role"] for x in payload["series"]} == {"observed", "counterfactual", "interval_lower", "interval_upper"}
    assert payload["evidence"] == {"call_id": "cause-1", "trace_hash": "f" * 64, "tool": "bayesian_structural_causal_impact"}
    assert original == copy


@pytest.mark.parametrize("bad", [
    {"observed": [1]},
    {"intervals": [[1, 2]]},
    {"intervals": [[2, 1], [1, 2], [1, 2]]},
    {"intervals": [[1, 2], [1], [1, 2]]},
    {"intervals": [[1, 2], [float("nan"), 3], [1, 2]]},
    {"observed": [True, 2, 3]},
    {"counterfactual": [1, float("inf"), 3]},
])
def test_bad_causal_series_is_rejected(bad):
    values = causal()
    for key, value in bad.items():
        values[{"intervals": "counterfactual_post_interval"}.get(key, {"observed": "observed_post", "counterfactual": "counterfactual_post_mean"}.get(key, key))] = value
    with pytest.raises(VisualizationSpecError):
        build(item("bayesian_structural_causal_impact", values))


def test_multi_evidence_order_is_independent_of_input_order():
    items = (
        item("bayesian_structural_causal_impact", causal(), "c"),
        item("one_way_anova", {"group_statistics": {"A": {"mean": 11}}}, "a"),
        item("mean_shift_change_point", shift(), "s"),
        item("linear_trend", trend(), "t"),
        item("mann_whitney_u_test", {"p_value": .5}, "m"),
    )
    expected = ("linear_trend", "mean_shift_change_point", "one_way_anova", "bayesian_structural_causal_impact")
    assert tuple(s.evidence_tool for s in build(*items)) == expected
    assert tuple(s.evidence_tool for s in build(*reversed(items))) == expected


def test_resolved_anova_evidence_generates_only_kruskal_medians():
    result = run([planned()], statuses={
        "call-a": (VerificationStatus.NEEDS_MORE_ANALYSIS, "run_robustness_check"),
    })
    brief = result.build_decision_brief()
    charts = result.build_visualizations()
    assert len(charts) == 1
    assert charts[0].title == "Verified group medians"
    assert charts[0].trace_hash == brief.evidence.items[0].trace_hash
    assert charts[0].call_id == brief.evidence.items[0].call_id


def _real_pipeline():
    class Planner:
        def plan(self, question, *, dataset_context, tool_catalog):
            return SimpleNamespace(traces=plan_traces())
    return VerifiedAnalyticsPipeline(Planner()).run(QUESTION, dataset=demo_dataset(), dataset_context=PROFILE)


def test_real_pipeline_four_charts_and_verified_hashes():
    result = _real_pipeline()
    assert result.succeeded
    brief = result.build_decision_brief()
    charts = result.build_visualizations()
    assert len(charts) == 4
    assert [s.chart_type for s in charts] == ["line", "bar", "bar", "line"]
    assert [s.evidence_tool for s in charts] == ["linear_trend", "mean_shift_change_point", "one_way_anova", "bayesian_structural_causal_impact"]
    by_id = {e.call_id: e for e in brief.evidence.items}
    for s in charts:
        assert s.trace_hash == by_id[s.call_id].trace_hash
        assert s.evidence_tool == by_id[s.call_id].tool
        assert s.as_dict()["evidence"]["trace_hash"] == by_id[s.call_id].trace_hash
    assert len(charts[-1].bands) == 1
    print("CHARTS", [(s.chart_type, s.title, len(s.data), s.call_id, s.trace_hash[:12]) for s in charts])
    print("CAUSAL_FIRST_THREE", list(charts[-1].data[:3]))


def test_api_returns_multi_charts_without_visualization_mocks():
    class Planner:
        def plan(self, question, *, dataset_context, tool_catalog):
            return SimpleNamespace(traces=plan_traces())
    app = create_app(planner_factory=lambda: Planner())
    client = TestClient(app)
    frame = demo_dataset()
    csv = frame.to_csv(index=False).encode()
    upload = client.post("/datasets", files={"file": ("sales.csv", csv, "text/csv")})
    assert upload.status_code == 201
    response = client.post("/analyses", json={"dataset_id": upload.json()["dataset_id"], "question": QUESTION})
    assert response.status_code == 201
    payload = response.json()
    assert payload["succeeded"]
    assert len(payload["visualizations"]) == 4
    assert payload["decision_brief"]
    for visual in payload["visualizations"]:
        assert visual["evidence"]["trace_hash"] in {x["trace_hash"] for x in payload["decision_brief"]["evidence"]["items"]}
    assert payload["visualizations"][-1]["bands"]
