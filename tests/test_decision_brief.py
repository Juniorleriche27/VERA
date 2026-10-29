from types import SimpleNamespace

import pytest

from agent.decision_brief import DecisionBriefError, VerifiedDecisionBriefBuilder
from agent.pipeline import PipelineResult, PipelineStep
from agent.tool_call_trace import ToolCallTrace, VerificationStatus


def _verified_trace(tool="one_way_anova", result=None):
    if result is None:
        result = {
            "p_value": 6.105260231223826e-10,
            "f_statistic": 200.0,
            "eta_squared": 0.970873786407767,
            "sample_size": 15,
            "group_statistics": {
                "North": {"mean": 12.0},
                "South": {"mean": 22.0},
                "West": {"mean": 32.0},
            },
        }
    trace = ToolCallTrace(
        call_id="call-brief-1",
        tool=tool,
        objective="Compare regional sales.",
        reason="Use verified deterministic evidence.",
        _parameters_json='{"alpha":0.05}',
        _bindings_json='{"group_col":"region","value_col":"sales"}',
    )
    return (
        trace.mark_running()
        .complete(result)
        .verify(VerificationStatus.VERIFIED, next_decision="interpret_result")
    )


def _pipeline(trace=None, report=None):
    trace = trace or _verified_trace()
    return PipelineResult(
        question="Are average sales different across regions?",
        planning=SimpleNamespace(),
        steps=(PipelineStep(index=1, trace=trace, report=report),),
    )


def test_decision_brief_requires_pipeline_result():
    with pytest.raises(DecisionBriefError):
        VerifiedDecisionBriefBuilder().build(object())


def test_decision_brief_rejects_empty_pipeline():
    result = PipelineResult(question="Q", planning=None, steps=())
    with pytest.raises(DecisionBriefError):
        VerifiedDecisionBriefBuilder().build(result)


def test_decision_brief_rejects_unverified_pipeline():
    trace = ToolCallTrace(
        call_id="unverified",
        tool="one_way_anova",
        objective="Compare groups.",
        reason="test",
        _parameters_json="{}",
        _bindings_json="{}",
    ).mark_running().complete({"p_value": 0.01})
    with pytest.raises(DecisionBriefError):
        VerifiedDecisionBriefBuilder().build(_pipeline(trace=trace))


def test_anova_brief_interprets_only_verified_result():
    brief = VerifiedDecisionBriefBuilder().build(_pipeline())
    assert "statistically significant" in brief.headline
    assert any("6.105e-10" in item for item in brief.findings)
    assert any("F statistic is 200" in item for item in brief.findings)
    assert any("eta-squared" in item for item in brief.findings)
    assert any("North=12" in item and "West=32" in item for item in brief.findings)


def test_evidence_preserves_trace_hash_and_result():
    pipeline = _pipeline()
    brief = VerifiedDecisionBriefBuilder().build(pipeline)
    evidence = brief.evidence.items[0]
    assert evidence.trace_hash == pipeline.final_trace.trace_hash
    assert evidence.call_id == "call-brief-1"
    assert evidence.result["sample_size"] == 15


def test_decision_brief_does_not_expose_parameters_or_bindings():
    payload = VerifiedDecisionBriefBuilder().build(_pipeline()).as_dict()
    evidence = payload["evidence"]["items"][0]
    assert "parameters" not in evidence
    assert "bindings" not in evidence
    assert "alpha" not in repr(payload)
    assert "region" not in evidence.keys()


def test_generic_verified_result_has_safe_fallback_interpretation():
    trace = _verified_trace(
        tool="descriptive_statistics",
        result={"sample_size": 42, "estimate": 12.5},
    )
    brief = VerifiedDecisionBriefBuilder().build(_pipeline(trace=trace))
    assert any("estimate=12.5" in item for item in brief.findings)
    assert any("sample_size=42" in item for item in brief.findings)
    assert brief.headline == "VERA produced a verified analytical result."


def test_failed_report_messages_become_caveats_without_raw_data_access():
    report = SimpleNamespace(
        checks=(
            SimpleNamespace(passed=False, message="Small subgroup requires caution."),
            SimpleNamespace(passed=True, message="This should not appear."),
        )
    )
    brief = VerifiedDecisionBriefBuilder().build(_pipeline(report=report))
    assert brief.caveats == ("Small subgroup requires caution.",)


def test_anova_significant_result_recommends_post_hoc_before_pair_claim():
    brief = VerifiedDecisionBriefBuilder().build(_pipeline())
    assert "post-hoc" in brief.next_action
    assert "specific pair" in brief.next_action


def test_anova_non_significant_result_avoids_difference_claim():
    trace = _verified_trace(
        result={
            "p_value": 0.42,
            "f_statistic": 0.9,
            "eta_squared": 0.02,
            "sample_size": 30,
            "group_statistics": {"A": {"mean": 10}, "B": {"mean": 10.5}},
        }
    )
    brief = VerifiedDecisionBriefBuilder().build(_pipeline(trace=trace))
    assert "does not find statistically significant" in brief.headline
    assert "Do not claim" in brief.next_action
