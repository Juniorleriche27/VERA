"""Regression coverage for core AnalysisResult envelopes and VERA extensions."""
from copy import deepcopy
import math

import pytest

from agent.analytics_executor import AnalyticsToolExecutor
from agent.data_binding import DatasetBinder
from agent.tool_call_trace import ToolCallTrace, VerificationStatus
from verification.verifier import DeterministicVerifier
from tests.test_binder_integration import dataset


def trace(result, tool="linear_trend"):
    return ToolCallTrace.planned(
        call_id="normalized-verification", tool=tool, objective="Verify analytics",
        reason="Protect evidence", bindings={}, parameters={},
    ).complete(result)


def engine(payload=None, **overrides):
    data = dict(status="success", method="linear_trend", category="timeseries",
                version="1.0", result=payload or {"sample_size": 20},
                warnings=[], diagnostics={}, metadata={})
    data.update(overrides)
    return data


def check(report, code):
    return next(c for c in report.checks if c.code == code)


def verified(result, tool="linear_trend", **verifier_kwargs):
    original = trace(result, tool=tool)
    before = deepcopy(original.deterministic_result)
    final_trace, report = DeterministicVerifier(**verifier_kwargs).verify(original)
    assert original.deterministic_result == before
    assert final_trace.deterministic_result == before
    assert report.trace_hash == final_trace.trace_hash
    return report


def test_top_level_extension_is_still_verified():
    assert verified({"sample_size": 20, "p_value": 0.08}, "one_way_anova").passed


def test_engine_valid_envelope():
    report = verified(engine())
    assert report.passed and check(report, "engine_status").passed
    assert check(report, "engine_method").passed
    assert check(report, "engine_result").passed


@pytest.mark.parametrize("changes,code", [
    ({"status": "error"}, "engine_status"),
    ({"method": "unrelated"}, "engine_method"),
    ({"result": None}, "engine_result"),
    ({"result": []}, "engine_result"),
])
def test_bad_engine_envelope_is_rejected(changes, code):
    report = verified(engine(**changes))
    assert report.status is VerificationStatus.REJECTED
    assert check(report, code).passed is False


def test_engine_result_absent_is_rejected():
    result = engine()
    del result["result"]
    assert verified(result).status is VerificationStatus.REJECTED


def test_nested_small_sample_warns_and_requests_more_analysis():
    report = verified(engine({"sample_size": 2}), minimum_sample_size=3)
    assert report.status is VerificationStatus.NEEDS_MORE_ANALYSIS
    assert not check(report, "sample_size:sample_size").passed


@pytest.mark.parametrize("p,expected", [(0.03, True), (1.4, False)])
def test_nested_probability(p, expected):
    report = verified(engine({"sample_size": 20, "p_value": p}))
    assert check(report, "probability_bounds").passed is expected
    assert report.passed is expected


def test_nested_missingness():
    report = verified(engine({"sample_size": 40, "missing_count": 30, "total_count": 40}))
    assert report.status is VerificationStatus.NEEDS_MORE_ANALYSIS
    assert not check(report, "missingness").passed


def test_engine_warnings_are_recorded_but_not_rejected():
    report = verified(engine(warnings=["Limited evidence"]))
    assert report.passed
    assert not check(report, "engine_warnings").passed
    assert check(report, "engine_warnings").severity.value == "warning"


@pytest.mark.parametrize("location", ["result", "diagnostics"])
def test_negative_diagnostic_in_payload_or_envelope(location):
    data = engine({"sample_size": 30})
    data[location]["diagnostics_passed"] = False
    report = verified(data)
    assert report.status is VerificationStatus.NEEDS_MORE_ANALYSIS
    assert not check(report, "explicit_flag:diagnostics_passed").passed


def test_unrelated_false_diagnostic_does_not_fail():
    assert verified(engine({"sample_size": 30, "detected": False}, diagnostics={"stable": False})).passed


@pytest.mark.parametrize("interval,expected", [([2.0, 4.0], True), ([4.0, 2.0], False)])
def test_scalar_causal_interval(interval, expected):
    report = verified({"sample_size": 20, "absolute_effect_interval": interval})
    assert check(report, "interval_order").passed is expected
    assert report.passed is expected


def test_nan_interval_rejected_by_check():
    # Non-finite JSON is already forbidden by ToolCallTrace; exercise verifier's own defense.
    assert not DeterministicVerifier._check_intervals({"absolute_effect_interval": [math.nan, 2]}).passed


@pytest.mark.parametrize("pairs,expected", [([[1, 2], [3, 4]], True), ([[1, 2], [5, 3]], False)])
def test_causal_interval_collections(pairs, expected):
    result = {"sample_size": 20, "counterfactual_post_interval": pairs,
              "point_effect_interval": [[0, 1]]}
    report = verified(result)
    assert check(report, "interval_order").passed is expected


def test_malformed_interval_pair_is_rejected():
    report = verified({"sample_size": 20, "relative_effect_interval_percent": [1]})
    assert report.status is VerificationStatus.REJECTED


def causal_payload(**overrides):
    payload = {"sample_size": 20, "pre_intervention_count": 12,
               "post_intervention_count": 8, "intervention_index": 12,
               "absolute_effect_interval": [-5, 10],
               "relative_effect_interval_percent": [-2, 4],
               "counterfactual_post_interval": [[100, 120]] * 8,
               "point_effect_interval": [[-1, 2]] * 8,
               "causal_interpretation": "conditional_on_model_specification_and_counterfactual_assumptions"}
    payload.update(overrides)
    return payload


def test_causal_consistent_counts_and_scope():
    report = verified(causal_payload(), "bayesian_structural_causal_impact")
    assert report.passed
    assert check(report, "causal_sample_structure").passed
    assert check(report, "causal_assumptions_scope").severity.value == "info"


@pytest.mark.parametrize("changes", [
    {"sample_size": 21}, {"pre_intervention_count": 11},
    {"intervention_index": 13},
])
def test_causal_inconsistent_counts_are_rejected(changes):
    report = verified(causal_payload(**changes), "bayesian_structural_causal_impact")
    assert report.status is VerificationStatus.REJECTED
    assert not check(report, "causal_sample_structure").passed


def test_change_not_detected_remains_verified():
    report = verified(engine({"sample_size": 30, "detected": False}), "linear_trend")
    assert report.passed


def test_detected_out_of_bounds_change_rejected():
    report = verified(engine({"sample_size": 30, "detected": True, "change_index": 31}))
    assert report.status is VerificationStatus.REJECTED
    assert not check(report, "change_index_bounds").passed


def integration(tool, bindings, parameters=None):
    planned = ToolCallTrace.planned(
        call_id="real-analytics", tool=tool, objective="Real analytical execution",
        reason="Integration test", parameters=parameters or {}, bindings=bindings,
    )
    bound = DatasetBinder().bind(planned, dataset())
    result = AnalyticsToolExecutor().execute(tool, bound.parameters, bound.bindings)
    completed = planned.complete(result)
    final, report = DeterministicVerifier().verify(completed)
    assert final.deterministic_result == result
    assert final.trace_hash == report.trace_hash
    return result, report


def test_real_linear_trend_nested_sample_checked():
    result, report = integration("linear_trend", {
        "time_column": "date", "value_column": "sales", "aggregation": "sum",
    })
    assert result["status"] == "success"
    assert result["result"]["sample_size"] == 24
    assert check(report, "sample_size:sample_size").passed
    assert report.status is VerificationStatus.VERIFIED


def test_real_mean_shift_verification():
    _, report = integration("mean_shift_change_point", {
        "time_column": "date", "value_column": "sales", "aggregation": "sum",
    })
    assert check(report, "sample_size:sample_size").passed


def test_real_mann_whitney_verification():
    _, report = integration("mann_whitney_u_test", {
        "grouping_column": "region", "value_column": "sales",
        "group_a": "North", "group_b": "South",
    })
    assert report.status is VerificationStatus.VERIFIED
    assert check(report, "probability_bounds").passed


def test_real_causal_impact_verification():
    result, report = integration("bayesian_structural_causal_impact", {
        "time_column": "date", "value_column": "sales", "aggregation": "sum",
        "intervention_date": "2026-08-13", "control_columns": ["control"],
    })
    assert check(report, "interval_order").passed
    assert check(report, "causal_sample_structure").passed
    assert check(report, "causal_assumptions_scope").passed
    assert result["sample_size"] == 24
    assert report.status is VerificationStatus.VERIFIED
