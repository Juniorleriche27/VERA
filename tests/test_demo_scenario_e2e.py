"""Official synthetic VERA hackathon scenario, from versioned CSV through API."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from agent.analytics_executor import AnalyticsToolExecutor
from agent.tool_call_trace import TraceStatus, VerificationStatus
from agent.tool_catalog_selector import ToolCatalogSelector
from backend.app import create_app, _safe_dataset_profile
from backend.data_preparation import prepare_dataset
from scripts.run_demo_scenario import (
    CSV_PATH, DemoPlanner, INTERVENTION_DATE, QUESTION, load_demo_dataset, run_demo,
)

EXPECTED_TOOLS = (
    "linear_trend", "mean_shift_change_point", "one_way_anova",
    "bayesian_structural_causal_impact",
)
EXPECTED_IDS = ("demo-trend", "demo-change", "demo-regions", "demo-campaign")


def _payload(step):
    result = step.trace.deterministic_result
    return result.get("result", result)


def test_csv_is_versioned_synthetic_daily_multiregion_and_preserves_grain():
    assert CSV_PATH.exists()
    frame = load_demo_dataset()
    assert len(frame) == 61 * 3
    assert tuple(frame.columns) == ("date", "region", "sales", "control_sales", "campaign_spend")
    assert frame.date.min().strftime("%Y-%m-%d") == "2026-08-01"
    assert frame.date.max().strftime("%Y-%m-%d") == "2026-09-30"
    assert set(frame.region) == {"North", "South", "West"}
    assert frame.groupby("date").size().eq(3).all()
    assert frame.loc[frame.date < INTERVENTION_DATE, "campaign_spend"].eq(0).all()
    assert frame.loc[frame.date >= INTERVENTION_DATE, "campaign_spend"].gt(0).all()
    assert "synthetic" in (CSV_PATH.parent / "README.md").read_text().lower()


def test_upload_profile_roles_are_safe_and_complete():
    profile = _safe_dataset_profile(load_demo_dataset())
    assert profile["row_count"] == 183
    assert profile["column_count"] == 5
    assert {item["name"]: item["role"] for item in profile["columns"]} == {
        "date": "time", "region": "dimension", "sales": "measure",
        "control_sales": "measure", "campaign_spend": "measure",
    }
    assert all({"name", "dtype", "role", "missing_count", "missing_rate", "unique_count"} <= set(col) for col in profile["columns"])
    assert all(col["missing_count"] == 0 for col in profile["columns"])
    assert [c["unique_count"] for c in profile["columns"] if c["name"] == "date"] == [61]
    assert "North" not in str(profile) and "South" not in str(profile)
    assert "330.000" not in str(profile)


def test_official_tool_selector_top_ten_is_multidimensional():
    profile = _safe_dataset_profile(load_demo_dataset())
    selection = ToolCatalogSelector().select(
        QUESTION, dataset_context=profile, tool_catalog=AnalyticsToolExecutor().available_tools(),
    )
    assert len(selection.ranked_names) == 10
    assert set(EXPECTED_TOOLS) <= set(selection.ranked_names)
    assert "kruskal_wallis_test" in selection.ranked_names
    assert selection.ranked_names[:4] == EXPECTED_TOOLS


def test_direct_real_pipeline_and_provenance_and_brief():
    frame, profile, selected, result, brief, visuals = run_demo()
    assert result.succeeded is True
    assert result.exhausted is False
    assert result.unresolved_call_ids == ()
    assert len(result.steps) == 4
    assert tuple(step.trace.tool for step in result.steps) == EXPECTED_TOOLS
    assert tuple(step.trace.call_id for step in result.steps) == EXPECTED_IDS
    assert all(step.trace.status is TraceStatus.COMPLETED for step in result.steps)
    assert all(step.trace.verification_status is VerificationStatus.VERIFIED for step in result.steps)
    assert all(step.trace.trace_hash == step.report.trace_hash for step in result.steps)
    trend, change, anova, causal = map(_payload, result.steps)
    assert trend["slope"] < 0
    assert trend["r_squared_defined"]
    assert change["detected"] is True
    assert "2026-08-12" <= change["change_timestamp"][:10] <= "2026-08-20"
    assert change["strongest_candidate"]["left_mean"] > change["strongest_candidate"]["right_mean"]
    assert anova["p_value"] < .05
    assert anova["group_statistics"]["North"]["mean"] > anova["group_statistics"]["South"]["mean"] > anova["group_statistics"]["West"]["mean"]
    assert causal["intervention_index"] == 31
    assert causal["absolute_effect"] > 0
    assert causal["absolute_effect_interval"][0] > 0
    assert causal["relative_effect_percent"] > 0
    assert brief.headline != "VERA produced a verified analytical result."
    assert "downward" in brief.headline
    assert "change" in brief.headline
    assert "significant" in brief.headline
    assert "positive conditional" in brief.headline
    assert len(brief.evidence.items) == 4
    assert any("counterfactual assumptions" in caveat for caveat in brief.caveats)
    for forbidden in ("causality proven", "definitely caused", "proved that", "guaranteed causal"):
        assert forbidden not in str(brief.as_dict()).lower()
    assert tuple(chart.evidence_tool for chart in visuals) == EXPECTED_TOOLS
    assert tuple(chart.chart_type for chart in visuals) == ("line", "bar", "bar", "line")
    assert len(visuals) == 4
    by_id = {evidence.call_id: evidence for evidence in brief.evidence.items}
    for chart in visuals:
        assert chart.trace_hash == by_id[chart.call_id].trace_hash
        assert chart.evidence_tool == by_id[chart.call_id].tool
    assert visuals[-1].bands and visuals[-1].data[0]["post_period"] == 1
    assert len(visuals[-1].data) == 30


def test_reproducible_analytics_and_provenance_hashes_from_same_csv():
    # Re-read the versioned artifact from disk, not a separately generated fixture.
    pd.testing.assert_frame_equal(load_demo_dataset(), load_demo_dataset())
    _, _, selected_1, result_1, brief_1, visuals_1 = run_demo()
    _, _, selected_2, result_2, brief_2, visuals_2 = run_demo()
    assert selected_1.ranked_names == selected_2.ranked_names
    assert [step.trace.call_id for step in result_1.steps] == [step.trace.call_id for step in result_2.steps]
    assert [step.trace.deterministic_result for step in result_1.steps] == [step.trace.deterministic_result for step in result_2.steps]
    assert [step.trace.trace_hash for step in result_1.steps] == [step.trace.trace_hash for step in result_2.steps]
    assert brief_1.as_dict() == brief_2.as_dict()
    assert [spec.as_dict() for spec in visuals_1] == [spec.as_dict() for spec in visuals_2]


def test_fastapi_upload_analysis_get_full_payload_roundtrip():
    app = create_app(planner_factory=DemoPlanner)
    client = TestClient(app)
    raw = CSV_PATH.read_bytes()
    uploaded = client.post("/datasets", files={"file": (CSV_PATH.name, raw, "text/csv")})
    assert uploaded.status_code == 201, uploaded.text
    received = uploaded.json()
    assert set(received) == {"dataset_id", "filename", "profile"}
    assert received["filename"] == CSV_PATH.name
    assert received["profile"]["row_count"] == 183
    assert received["profile"]["column_count"] == 5
    assert {entry["name"]: entry["role"] for entry in received["profile"]["columns"]}["date"] == "time"
    assert "North" not in str(received["profile"])
    result = client.post("/analyses", json={"dataset_id": received["dataset_id"], "question": QUESTION})
    assert result.status_code == 201, result.text
    body = result.json()
    assert body["succeeded"] is True
    assert body["exhausted"] is False
    assert len(body["steps"]) == 4
    assert body["unresolved_call_ids"] == [] if "unresolved_call_ids" in body else True
    assert body["decision_brief"] is not None
    assert len(body["visualizations"]) == 4
    assert [item["trace"]["tool"] for item in body["steps"]] == list(EXPECTED_TOOLS)
    for step in body["steps"]:
        assert step["trace"]["trace_hash"] == step["report"]["trace_hash"]
    evidence = {entry["call_id"]: entry["trace_hash"] for entry in body["decision_brief"]["evidence"]["items"]}
    for spec in body["visualizations"]:
        assert spec["evidence"]["trace_hash"] == evidence[spec["evidence"]["call_id"]]
    fetched = client.get(f"/analyses/{body['analysis_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == body
