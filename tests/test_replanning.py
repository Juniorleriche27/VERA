from dataclasses import FrozenInstanceError

import pytest

from agent.replanning import ReplanningError, SelfCorrectionPlanner
from agent.tool_call_trace import (
    ToolCallTrace,
    TraceStatus,
    VerificationStatus,
)


def _source(
    *,
    tool="one_way_anova",
    verification_status=VerificationStatus.NEEDS_MORE_ANALYSIS,
    next_decision="run_kruskal_wallis_test",
):
    return (
        ToolCallTrace.planned(
            call_id="call-1",
            tool=tool,
            objective="Compare regional sales",
            reason="Compare independent regional groups",
            parameters={"alpha": 0.05},
            bindings={"data": ["north", "south", "west"]},
            source_call_ids=["call-0"],
        )
        .complete({"p_value": 0.02})
        .verify(
            verification_status,
            next_decision=next_decision,
        )
    )


def test_propose_creates_immutable_replan_directive():
    directive = SelfCorrectionPlanner().propose(
        _source(),
        next_call_id="call-2",
    )

    assert directive.source_call_id == "call-1"
    assert directive.next_call_id == "call-2"
    assert directive.tool == "kruskal_wallis_test"
    assert directive.source_call_ids == ("call-1", "call-0")

    with pytest.raises(FrozenInstanceError):
        directive.tool = "changed"


def test_directive_materializes_new_planned_trace():
    source = _source()
    trace = SelfCorrectionPlanner().propose(
        source,
        next_call_id="call-2",
    ).to_trace()

    assert trace.status is TraceStatus.PLANNED
    assert trace.call_id == "call-2"
    assert trace.tool == "kruskal_wallis_test"
    assert trace.source_call_ids == ("call-1", "call-0")
    assert source.call_id == "call-1"
    assert source.verification_status is VerificationStatus.NEEDS_MORE_ANALYSIS


def test_replanning_does_not_mutate_source_trace():
    source = _source()
    before = source.trace_hash

    SelfCorrectionPlanner().propose(source, next_call_id="call-2")

    assert source.trace_hash == before
    assert source.next_decision == "run_kruskal_wallis_test"


def test_explicit_tool_overrides_decision_inference():
    directive = SelfCorrectionPlanner().propose(
        _source(next_decision="perform a stronger comparison"),
        next_call_id="call-2",
        tool="mann_whitney_u_test",
    )

    assert directive.tool == "mann_whitney_u_test"


def test_custom_parameters_and_bindings_are_used():
    directive = SelfCorrectionPlanner().propose(
        _source(),
        next_call_id="call-2",
        parameters={"alpha": 0.01},
        bindings={"data": ["east", "west"]},
    )

    assert directive.parameters == {"alpha": 0.01}
    assert directive.bindings == {"data": ["east", "west"]}


def test_additional_sources_are_deduplicated():
    directive = SelfCorrectionPlanner().propose(
        _source(),
        next_call_id="call-2",
        additional_source_call_ids=["call-0", "call-extra"],
    )

    assert directive.source_call_ids == ("call-1", "call-0", "call-extra")


def test_verified_trace_cannot_be_replanned():
    trace = _source(
        verification_status=VerificationStatus.VERIFIED,
        next_decision="interpret_result",
    )

    with pytest.raises(ReplanningError):
        SelfCorrectionPlanner().propose(trace, next_call_id="call-2")


def test_uncompleted_trace_cannot_be_replanned():
    trace = ToolCallTrace.planned(
        call_id="call-1",
        tool="one_way_anova",
        objective="Compare groups",
        reason="Need group comparison",
    )

    with pytest.raises(ReplanningError):
        SelfCorrectionPlanner().propose(trace, next_call_id="call-2")


def test_replanned_call_requires_new_call_id():
    with pytest.raises(ReplanningError):
        SelfCorrectionPlanner().propose(
            _source(),
            next_call_id="call-1",
        )


def test_unknown_decision_requires_explicit_tool():
    with pytest.raises(ReplanningError):
        SelfCorrectionPlanner().propose(
            _source(next_decision="inspect assumptions"),
            next_call_id="call-2",
        )


def test_automatic_robustness_anova_to_kruskal():
    source = _source(next_decision="run_robustness_check")

    directive = SelfCorrectionPlanner().automatic_robustness_check(
        source,
        next_call_id="call-2",
    )

    assert directive.tool == "kruskal_wallis_test"
    assert "robustness check" in directive.reason


def test_automatic_robustness_pearson_to_spearman():
    source = _source(
        tool="pearson_correlation",
        next_decision="run_robustness_check",
    )

    directive = SelfCorrectionPlanner().automatic_robustness_check(
        source,
        next_call_id="call-2",
    )

    assert directive.tool == "spearman_correlation"


def test_unknown_automatic_fallback_is_rejected():
    source = _source(
        tool="custom_method",
        next_decision="run_robustness_check",
    )

    with pytest.raises(ReplanningError):
        SelfCorrectionPlanner().automatic_robustness_check(
            source,
            next_call_id="call-2",
        )


def test_as_dict_contains_audit_link_to_source_hash():
    source = _source()
    directive = SelfCorrectionPlanner().propose(
        source,
        next_call_id="call-2",
    )

    payload = directive.as_dict()

    assert payload["source_call_id"] == "call-1"
    assert payload["source_trace_hash"] == source.trace_hash
    assert payload["next_call_id"] == "call-2"
    assert payload["tool"] == "kruskal_wallis_test"
