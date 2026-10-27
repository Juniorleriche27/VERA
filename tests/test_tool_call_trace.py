from dataclasses import FrozenInstanceError

import pytest

from agent.tool_call_trace import (
    ToolCallTrace,
    ToolCallTraceError,
    TraceStatus,
    VerificationStatus,
)


def _planned(**overrides):
    values = {
        "call_id": "call-1",
        "tool": "one_way_anova",
        "objective": "Compare regional sales",
        "reason": "Compare several independent group means",
        "parameters": {"alpha": 0.05},
        "bindings": {"data": ["north", "south", "west"]},
    }
    values.update(overrides)
    return ToolCallTrace.planned(**values)


def test_trace_is_frozen_and_inputs_are_defensively_copied():
    parameters = {"alpha": 0.05, "nested": {"x": [1, 2]}}
    trace = _planned(parameters=parameters)

    parameters["alpha"] = 0.99
    parameters["nested"]["x"].append(3)

    assert trace.parameters == {"alpha": 0.05, "nested": {"x": [1, 2]}}

    with pytest.raises(FrozenInstanceError):
        trace.call_id = "changed"


def test_returned_payload_cannot_mutate_trace():
    trace = _planned()
    payload = trace.parameters
    payload["alpha"] = 0.99

    assert trace.parameters["alpha"] == pytest.approx(0.05)


def test_hash_is_stable_for_equivalent_mapping_order():
    first = _planned(parameters={"b": 2, "a": 1})
    second = _planned(parameters={"a": 1, "b": 2})

    assert first.trace_hash == second.trace_hash
    assert len(first.trace_hash) == 64


def test_running_and_completion_return_new_immutable_snapshots():
    planned = _planned()
    running = planned.mark_running()
    completed = running.complete({"p_value": 0.012, "f_statistic": 7.4})

    assert planned.status is TraceStatus.PLANNED
    assert running.status is TraceStatus.RUNNING
    assert completed.status is TraceStatus.COMPLETED
    assert completed.deterministic_result["p_value"] == pytest.approx(0.012)
    assert len({planned.trace_hash, running.trace_hash, completed.trace_hash}) == 3


def test_completed_trace_can_be_verified():
    completed = _planned().complete({"p_value": 0.2})
    verified = completed.verify(
        VerificationStatus.VERIFIED,
        next_decision="interpret_result",
    )

    assert verified.verification_status is VerificationStatus.VERIFIED
    assert verified.next_decision == "interpret_result"
    assert completed.verification_status is VerificationStatus.NOT_VERIFIED


def test_needs_more_analysis_requires_next_decision():
    completed = _planned().complete({"p_value": 0.2})

    with pytest.raises(ToolCallTraceError):
        completed.verify(VerificationStatus.NEEDS_MORE_ANALYSIS)

    replanned = completed.verify(
        VerificationStatus.NEEDS_MORE_ANALYSIS,
        next_decision="run_spearman_correlation",
    )
    assert replanned.next_decision == "run_spearman_correlation"


def test_verification_before_completion_is_rejected():
    with pytest.raises(ToolCallTraceError):
        _planned().verify(VerificationStatus.VERIFIED)


def test_failed_trace_records_error_without_fake_result():
    failed = _planned().mark_running().fail("insufficient observations")

    assert failed.status is TraceStatus.FAILED
    assert failed.execution_error == "insufficient observations"
    assert failed.deterministic_result is None
    assert failed.verification_status is VerificationStatus.NOT_VERIFIED


def test_sources_are_unique_and_cannot_reference_self():
    with pytest.raises(ToolCallTraceError):
        _planned(source_call_ids=["upstream", "upstream"])

    with pytest.raises(ToolCallTraceError):
        _planned(source_call_ids=["call-1"])


def test_non_finite_or_non_json_payloads_are_rejected():
    with pytest.raises(ToolCallTraceError):
        _planned(parameters={"alpha": float("nan")})

    with pytest.raises(ToolCallTraceError):
        _planned(parameters={"bad": object()})


def test_as_dict_contains_complete_audit_contract():
    trace = _planned(source_call_ids=["call-0"]).complete(
        {"effect": 12.5}
    ).verify(
        VerificationStatus.NEEDS_MORE_ANALYSIS,
        next_decision="run_robustness_check",
    )

    payload = trace.as_dict()

    assert payload["call_id"] == "call-1"
    assert payload["tool"] == "one_way_anova"
    assert payload["objective"] == "Compare regional sales"
    assert payload["reason"] == "Compare several independent group means"
    assert payload["parameters"] == {"alpha": 0.05}
    assert payload["bindings"] == {"data": ["north", "south", "west"]}
    assert payload["status"] == "completed"
    assert payload["deterministic_result"] == {"effect": 12.5}
    assert payload["verification_status"] == "needs_more_analysis"
    assert payload["next_decision"] == "run_robustness_check"
    assert payload["source_call_ids"] == ["call-0"]
    assert len(payload["trace_hash"]) == 64
