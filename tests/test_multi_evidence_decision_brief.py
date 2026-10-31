"""Business synthesis, cautious causal language, and preserved verified evidence."""
from __future__ import annotations

from types import SimpleNamespace
import pytest

from agent.decision_brief import DecisionBriefError, VerifiedDecisionBriefBuilder
from agent.pipeline import PipelineResult, PipelineStep, VerifiedAnalyticsPipeline
from agent.tool_call_trace import ToolCallTrace, VerificationStatus
from tests.test_multi_tool_investigation import QUESTION, PROFILE, demo_dataset, plan_traces, TOOLS


def finished(tool, result, call_id, *, sources=(), status=VerificationStatus.VERIFIED):
    return ToolCallTrace.planned(
        call_id=call_id, tool=tool, objective=f"Study {tool}", reason="Evidence",
        parameters={}, bindings={}, source_call_ids=sources,
    ).complete(result).verify(status, next_decision="interpret_result")


def pipeline(*traces):
    return PipelineResult(QUESTION, None, tuple(PipelineStep(i + 1, trace) for i, trace in enumerate(traces)))


def brief_for(tool, result):
    return VerifiedDecisionBriefBuilder().build(pipeline(finished(tool, result, "e1")))


@pytest.mark.parametrize("slope,direction", [(-3.2, "downward"), (2.5, "upward"), (0.0, "flat")])
def test_trend_direction_and_r_squared(slope, direction):
    payload = {"status": "success", "method": "linear_trend", "result": {"slope": slope, "r_squared": 0.8, "r_squared_defined": True}, "warnings": [], "diagnostics": {}}
    b = brief_for("linear_trend", payload)
    assert direction in b.findings[0]
    assert "R-squared" in b.findings[0]
    assert b.evidence.items[0].result == payload
    assert b.evidence.items[0].trace_hash == pipeline(finished("linear_trend", payload, "e1")).steps[0].trace.trace_hash


def test_undefined_r_squared_not_misreported():
    b = brief_for("linear_trend", {"status": "success", "method": "linear_trend", "result": {"slope": -1.0, "r_squared_defined": False}})
    assert not any("R-squared" in f for f in b.findings)


@pytest.mark.parametrize("detected", [True, False])
def test_change_point_presence_and_absence(detected):
    b = brief_for("mean_shift_change_point", {"status": "success", "method": "mean_shift_change_point", "result": {
        "detected": detected, "change_timestamp": "2026-08-18",
        "strongest_candidate": {"left_mean": 150, "right_mean": 120},
    }})
    joined = " ".join(b.findings)
    if detected:
        assert "2026-08-18" in joined and "below" in joined
    else:
        assert "did not identify" in joined


def test_variance_shift():
    assert "variance" in brief_for("variance_shift_change_point", {"status": "success", "method": "variance_shift_change_point", "result": {"detected": True, "direction": "increase"}}).findings[0]


@pytest.mark.parametrize("tool,significant", [
    ("one_way_anova", True), ("one_way_anova", False),
    ("kruskal_wallis_test", True), ("kruskal_wallis_test", False),
    ("mann_whitney_u_test", True), ("mann_whitney_u_test", False),
])
def test_group_tests_are_significance_aware(tool, significant):
    p = 0.01 if significant else 0.38
    b = brief_for(tool, {"p_value": p, "group_statistics": {"North": {"median": 8}, "South": {"median": 12}},
                         "h_statistic": 5.1, "epsilon_squared": 0.3, "first_sample": "North", "second_sample": "South", "alternative": "two-sided"})
    content = " ".join(b.findings).lower()
    assert "0.01" in content if significant else "0.38" in content
    if tool == "kruskal_wallis_test":
        assert "rank" in content and "pairwise" in content
    if tool == "mann_whitney_u_test":
        assert "two-sided" in content


@pytest.mark.parametrize("bounds,expected", [([10, 30], "positive"), ([-30, -10], "negative"), ([-10, 20], "uncertain")])
def test_causal_intervals_and_conditional_language(bounds, expected):
    b = brief_for("bayesian_structural_causal_impact", {
        "absolute_effect": 12, "absolute_effect_interval": bounds,
        "relative_effect_percent": 2.0, "relative_effect_interval_percent": [-1, 3],
        "causal_interpretation": "conditional_on_model_specification_and_counterfactual_assumptions",
    })
    assert expected in b.headline
    assert expected in " ".join(b.findings)
    assert any("counterfactual assumptions" in c for c in b.caveats)
    text = str(b.as_dict()).lower()
    for forbidden in ("causality proven", "definitely caused", "proved that", "guaranteed causal"):
        assert forbidden not in text


def test_verified_descendant_excludes_unverified_ancestor():
    first = finished("one_way_anova", {"p_value": 0.3}, "a", status=VerificationStatus.NEEDS_MORE_ANALYSIS)
    second = finished("kruskal_wallis_test", {"p_value": 0.02, "h_statistic": 4.5}, "k", sources=("a",))
    result = pipeline(first, second)
    assert result.succeeded
    b = VerifiedDecisionBriefBuilder().build(result)
    assert len(b.evidence.items) == 1
    assert b.evidence.items[0].tool == "kruskal_wallis_test"
    assert b.evidence.items[0].source_call_ids == ("a",)
    assert b.evidence.as_dict()["items"][0]["source_call_ids"] == ["a"]
    assert b.evidence.items[0].trace_hash == second.trace_hash


def test_real_anova_kruskal_resolution_produces_final_brief():
    from tests.test_replanning_pipeline_resilience import run, planned
    result = run([planned()], statuses={
        "call-a": (VerificationStatus.NEEDS_MORE_ANALYSIS, "run_robustness_check"),
    })
    assert result.succeeded
    assert len(result.steps) == 2
    brief = result.build_decision_brief()
    assert [item.tool for item in brief.evidence.items] == ["kruskal_wallis_test"]
    assert brief.evidence.items[0].source_call_ids == ("call-a",)
    assert brief.evidence.items[0].trace_hash == result.steps[-1].trace.trace_hash
    assert not any(item.call_id == "call-a" for item in brief.evidence.items)


def test_unresolved_pipeline_does_not_generate_final_brief():
    rejected = finished("one_way_anova", {"p_value": 0.3}, "a", status=VerificationStatus.NEEDS_MORE_ANALYSIS)
    with pytest.raises(DecisionBriefError):
        VerifiedDecisionBriefBuilder().build(pipeline(rejected))


def test_findings_sorted_independent_of_execution_order_and_caveats_deduplicated():
    causal = finished("bayesian_structural_causal_impact", {"absolute_effect": 2, "absolute_effect_interval": [-1, 3], "causal_interpretation": "conditional_on_model_specification_and_counterfactual_assumptions"}, "c")
    group = finished("one_way_anova", {"p_value": .8}, "g")
    change = finished("mean_shift_change_point", {"status": "success", "method": "mean_shift_change_point", "result": {"detected": False, "limitations": ["Short sample.", "Short sample."]}}, "m")
    trend = finished("linear_trend", {"status": "success", "method": "linear_trend", "result": {"slope": -1.2, "limitations": ["Short sample."]}}, "t")
    b = VerifiedDecisionBriefBuilder().build(pipeline(causal, group, change, trend))
    assert "trend" in b.findings[0].lower()
    assert "mean level" in b.findings[1].lower()
    assert "ANOVA" in " ".join(b.findings[2:5])
    assert "counterfactual" in " ".join(b.findings[-1:]).lower()
    assert b.caveats.count("Short sample.") == 1
    assert list(item.call_id for item in b.evidence.items) == ["c", "g", "m", "t"]
    assert "downward" in b.headline and "uncertain" in b.headline


def test_real_four_tool_pipeline_brief_has_original_engine_envelopes():
    class Planner:
        def plan(self, question, *, dataset_context, tool_catalog):
            return SimpleNamespace(traces=plan_traces())
    result = VerifiedAnalyticsPipeline(Planner()).run(QUESTION, dataset=demo_dataset(), dataset_context=PROFILE)
    assert result.succeeded
    brief = result.build_decision_brief()
    assert tuple(item.tool for item in brief.evidence.items) == TOOLS
    assert len(brief.findings) >= 4
    assert brief.headline != "VERA produced a verified analytical result."
    for item in brief.evidence.items:
        original = next(step.trace for step in result.steps if step.trace.call_id == item.call_id)
        assert item.trace_hash == original.trace_hash
        assert item.result == original.deterministic_result
    assert brief.evidence.items[0].result["status"] == "success"
    assert "slope" in brief.evidence.items[0].result["result"]
    assert any("counterfactual assumptions" in c for c in brief.caveats)
    assert brief.as_dict()["evidence_count"] == 4
    assert brief.as_dict()["tools_used"] == list(TOOLS)
    print("DEMO_HEADLINE", brief.headline)
    print("DEMO_FINDINGS", list(brief.findings))
    print("DEMO_CAVEATS", list(brief.caveats))
    print("DEMO_NEXT_ACTION", brief.next_action)
    print("DEMO_PROVENANCE", [(i.call_id, i.trace_hash) for i in brief.evidence.items])
