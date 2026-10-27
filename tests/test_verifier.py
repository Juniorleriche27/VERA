from dataclasses import FrozenInstanceError

import pytest

from agent.tool_call_trace import (
    ToolCallTrace,
    TraceStatus,
    VerificationStatus,
)
from verification.verifier import (
    DeterministicVerifier,
    VerificationReport,
    VerifierError,
)


def _trace(result, **overrides):
    values = {
        "call_id": "call-1",
        "tool": "one_way_anova",
        "objective": "Compare regional sales",
        "reason": "Test whether regional means differ",
        "parameters": {},
        "bindings": {"data": ["north", "south", "west"]},
    }
    values.update(overrides)
    return ToolCallTrace.planned(**values).complete(result)


def test_valid_completed_result_is_verified():
    trace, report = DeterministicVerifier().verify(
        _trace({"sample_size": 30, "p_value": 0.012})
    )
    assert trace.verification_status is VerificationStatus.VERIFIED
    assert trace.next_decision == "interpret_result"
    assert report.passed is True


def test_verifier_requires_completed_trace():
    trace = ToolCallTrace.planned(
        call_id="call-1",
        tool="one_way_anova",
        objective="Compare groups",
        reason="Mean comparison",
    )
    with pytest.raises(VerifierError):
        DeterministicVerifier().evaluate(trace)


def test_missing_result_is_rejected():
    verified, report = DeterministicVerifier().verify(_trace(None))
    assert verified.verification_status is VerificationStatus.REJECTED
    assert verified.next_decision == "rerun_tool_with_valid_output"
    assert report.failed_checks[0].code == "result_present"


def test_invalid_probability_is_rejected():
    verified, report = DeterministicVerifier().verify(
        _trace({"sample_size": 10, "p_value": 1.2})
    )
    assert verified.verification_status is VerificationStatus.REJECTED
    assert verified.next_decision == "rerun_tool_with_valid_output"
    assert any(check.code == "probability_bounds" for check in report.failed_checks)


def test_invalid_interval_order_is_rejected():
    verified, report = DeterministicVerifier().verify(
        _trace(
            {
                "sample_size": 20,
                "credible_interval_lower": 5.0,
                "credible_interval_upper": 2.0,
            }
        )
    )
    assert verified.verification_status is VerificationStatus.REJECTED
    assert verified.next_decision == "rerun_tool_with_valid_output"
    assert any(check.code == "interval_order" for check in report.failed_checks)


def test_high_missingness_requests_more_analysis():
    verified, report = DeterministicVerifier().verify(
        _trace({"sample_size": 60, "missing_count": 40, "total_count": 100})
    )
    assert verified.verification_status is VerificationStatus.NEEDS_MORE_ANALYSIS
    assert verified.next_decision == "run_data_quality_and_missingness_analysis"
    assert report.passed is False


def test_small_sample_requests_more_data():
    verified, _ = DeterministicVerifier(minimum_sample_size=5).verify(
        _trace({"sample_size": 3})
    )
    assert verified.verification_status is VerificationStatus.NEEDS_MORE_ANALYSIS
    assert verified.next_decision == "collect_more_data_or_reduce_scope"


def test_failed_explicit_diagnostic_requests_robustness_check():
    verified, report = DeterministicVerifier().verify(
        _trace({"sample_size": 30, "diagnostics_passed": False})
    )
    assert verified.verification_status is VerificationStatus.NEEDS_MORE_ANALYSIS
    assert verified.next_decision == "run_robustness_check"
    assert any(
        check.code == "explicit_flag:diagnostics_passed"
        for check in report.failed_checks
    )


def test_adjusted_p_values_are_checked():
    verified, _ = DeterministicVerifier().verify(
        _trace(
            {
                "sample_size": 10,
                "adjusted_p_values": [0.01, 0.2, 0.9],
            }
        )
    )
    assert verified.verification_status is VerificationStatus.VERIFIED


def test_non_mapping_json_result_can_pass_generic_verification():
    verified, report = DeterministicVerifier().verify(_trace([1, 2, 3]))
    assert verified.verification_status is VerificationStatus.VERIFIED
    assert report.passed is True


def test_report_is_frozen_and_serializable():
    _, report = DeterministicVerifier().verify(
        _trace({"sample_size": 10, "p_value": 0.4})
    )
    assert isinstance(report, VerificationReport)
    payload = report.as_dict()
    assert payload["status"] == "verified"
    assert payload["next_decision"] == "interpret_result"
    assert payload["passed"] is True
    assert payload["checks"]

    with pytest.raises(FrozenInstanceError):
        report.next_decision = "changed"


def test_constructor_thresholds_are_validated():
    with pytest.raises(VerifierError):
        DeterministicVerifier(missingness_warning_threshold=1.5)

    with pytest.raises(VerifierError):
        DeterministicVerifier(minimum_sample_size=0)


def test_evaluate_does_not_mutate_original_trace():
    original = _trace({"sample_size": 10})
    report = DeterministicVerifier().evaluate(original)
    assert original.verification_status is VerificationStatus.NOT_VERIFIED
    assert original.next_decision == ""
    assert report.status is VerificationStatus.VERIFIED
    assert original.status is TraceStatus.COMPLETED
